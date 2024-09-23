from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Union
from types import MappingProxyType # immutable dictionary
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt
import torch


diffusion_coefficients = MappingProxyType({
    'CO2': 1.91e-09,
    'OH': 5.293e-09,
    'CO3': 9.23e-10,
    'HCO3': 1.18e-09,
    'H': 9.311e-09,
    'K': 1.96e-09,
    'CO': 2.03e-09,
    'H2': 4.5e-09
}) # [m^2/s]

salting_out_exponents = MappingProxyType({
    'h_OH': 6.67e-05,
    'h_CO3': 0.0001251,
    'h_HCO3': 7.95e-05,
    'h_K': 7.5e-05,
    'h_CO2': 0.0
}) # [m^3/mol]

electrode_reaction_kinetics = MappingProxyType({
    'i_0_CO': 0.00471, # [A/m^2]
    'i_0_H2a': 0.00979, # [A/m^2]
    'i_0_H2b': 1.16e-05, # [A/m^2] not used in the calculation
    'alpha_CO': 0.44,
    'alpha_H2a': 0.27,
    'alpha_H2b': 0.36,
    'E_0_CO2': -0.11, # [V]
    'E_0_H2a': 0.0, # [V]
    'E_0_H2b': 0.0 # [V]
})

chemical_reaction_rates = MappingProxyType({
    'k1f': 2.23, # [m^3/(mol s)]
    # 'k1r': 0.000840308, # [1/s]
    # 'k2f': 6000000.0, # [m^3/(mol s)]
    # 'k2r': 345200.0, # [1/s]
    'c_ref': 1000.0 # [mol/m^3]
})

@dataclass
class InputParameters:
    T: float # temperature [K]
    p0: float # CO2 partial pressure [atm]
    Q: float # Flow rate [mL/min]
    flow_chan_length: float # Flow channel active length [m]
    flow_chan_height: float # Flow channel height (plane parallel) [m]
    flow_chan_width: float # Flow channel width (plane perpendicular) [m]
    L: float # Catalyst layer length [m]
    eps: float # Catalyst layer porosity
    r : float # Catalyst layer average particle radius [m]
    c_khco3: float # KHCO3 concentration [mol/m^3]
    c_k: float # K+ concentration [mol/m^3]
    dic: float # ?
    method: str # Method for calculating CO2 concentration (DIC or CO2 eql)

    @property
    def v(self): # flow velocity [m/s]
        return self.Q / (self.flow_chan_height * self.flow_chan_width) * 1e-6 / 60
    
    @property
    def volumetric_surface_area(self):
        return 3 * (1 - self.eps) / self.r # m^-1
    
def detach_dict(d: dict) -> dict:
    out = {}
    for k,v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.item()
        else:
            out[k] = v
    return out

def copy_input_parameters(ipt: InputParameters, keep_grad: bool = True):
    if keep_grad:
        return InputParameters(
            T=ipt.T,
            p0=ipt.p0,
            Q=ipt.Q,
            flow_chan_length=ipt.flow_chan_length,
            flow_chan_height=ipt.flow_chan_height,
            flow_chan_width=ipt.flow_chan_width,
            L=ipt.L,
            eps=ipt.eps,
            r=ipt.r,
            c_khco3=ipt.c_khco3,
            c_k=ipt.c_k,
            dic=ipt.dic,
            method=ipt.method
        )
    d = vars(ipt)
    d = detach_dict(d)
    return InputParameters(**d)
    
class System:
    diffusion_coefficients: Dict[str, float]
    salting_out_exponents: Dict[str, float]
    electrode_reaction_kinetics: Dict[str, float]
    chemical_reaction_rates: Dict[str, float]

    T: float # temperature [K]
    R: float = 8.3145 # Universal gas constant [J/(mol*K)]
    F: float = 96485 # Faraday constant [C/mol]
    
    @property
    def T(self) -> float:
        return self.input_parameters.T

    @property
    def Hnr(self) -> float:
        T = self.T
        return 1000*self.mod.exp(93.4517*100/T - 60.2409 + 23.3585*self.mod.log(T/100)) # Henry's constant for CO2 [mol/(L*atm)]
    
    def __init__(
        self, 
        input_parameters: InputParameters, 
        diffusion_coefficients: Dict[str, float]=diffusion_coefficients,
        salting_out_exponents: Dict[str, float]=salting_out_exponents,
        electrode_reaction_kinetics: Dict[str, float]=electrode_reaction_kinetics,
        chemical_reaction_rates: Dict[str, float]=chemical_reaction_rates,
        mod=np
    ):
        self.mod = mod

        self.diffusion_coefficients = diffusion_coefficients | {}
        self.salting_out_exponents = salting_out_exponents | {}
        self.electrode_reaction_kinetics = electrode_reaction_kinetics | {}
        self.chemical_reaction_rates = chemical_reaction_rates | {}
        self.input_parameters = input_parameters

        self.update_diffusion_coefficients()
        self.set_butler_volmer_coefficients()
        self.set_flow_channel_characteristics()
        self.set_initial_carbonate_equilibria()

    def update_diffusion_coefficients(self):
        substances = list(self.diffusion_coefficients.keys())
        for substance in substances:
            self.diffusion_coefficients[f'D{substance}'] = self.diffusion_coefficients[substance]*self.input_parameters.eps**1.5 # Bruggeman correction for diffusion coefficient [m^2/s]
        self.diffusion_coefficients['Gas diffusion layer mass transfer coefficient'] = self.diffusion_coefficients['DCO2'] / self.input_parameters.r # [m/s]

    def set_butler_volmer_coefficients(self):
        alpha_to_b = lambda alpha: self.T*self.R/(alpha*self.F)
        self.electrode_reaction_kinetics['b_CO2'] = alpha_to_b(self.electrode_reaction_kinetics['alpha_CO'])
        self.electrode_reaction_kinetics['b_H2a'] = alpha_to_b(self.electrode_reaction_kinetics['alpha_H2a'])
        self.electrode_reaction_kinetics['b_H2b'] = alpha_to_b(self.electrode_reaction_kinetics['alpha_H2b'])
    
    def set_flow_channel_characteristics(self):
        input_parameters = self.input_parameters
        flow_channel_characteristics = {
            'mu': 0.00000093944, # [m^2/s]
        }
        flow_channel_characteristics['Reynolds Number'] = input_parameters.v*input_parameters.flow_chan_length/flow_channel_characteristics['mu']
        flow_channel_characteristics['Hydrodynamic Entrance length'] = 0.0099*flow_channel_characteristics['Reynolds Number']*input_parameters.flow_chan_width
        flow_channel_characteristics['Parallel plate effective boundary layer'] = 13/35*input_parameters.flow_chan_width
        diff_coeff_to_bl_thickness = lambda x: 3*1.607/4*(input_parameters.flow_chan_width*x*input_parameters.flow_chan_length/input_parameters.v)**(1/3)
        diff_coeff_to_K_L = lambda D,L: D*self.mod.sqrt(L**-2+flow_channel_characteristics['Parallel plate effective boundary layer']**-2)/np.sqrt(2)
        for substance in ['CO2', 'OH', 'CO3', 'HCO3', 'H', 'K', 'CO', 'H2']:
            flow_channel_characteristics[f'Developing boundary layer thickness {substance} (average)'] = diff_coeff_to_bl_thickness(self.diffusion_coefficients[substance])
        for substance in ['CO2', 'OH', 'CO3', 'HCO3', 'H', 'K', 'CO', 'H2']:
            flow_channel_characteristics[f'K_L_{substance}'] = diff_coeff_to_K_L(self.diffusion_coefficients[substance], flow_channel_characteristics[f'Developing boundary layer thickness {substance} (average)'])

        self.flow_channel_characteristics = flow_channel_characteristics

    def set_initial_carbonate_equilibria(self):
        T = self.T
        kco3_to_salinity = lambda x: x/(1.005*x/1000+19.924)
        initial_carbonate_equilibria = {
            'Salinity': kco3_to_salinity(self.input_parameters.c_khco3),
            'pK1_0': -126.34048 + 6320.813/T + 19.568224*self.mod.log(T),
            'pK2_0': -90.18333 + 5143.692/T + 14.613358*self.mod.log(T),
        }
        salinity_to_A1 = lambda x: 13.4191*x**0.5 + 0.0331*x - 5.33e-5*x**2
        initial_carbonate_equilibria['A1'] = salinity_to_A1(initial_carbonate_equilibria['Salinity'])
        salinity_to_A2 = lambda x: 21.0894*x**0.5 + 0.1248*x - 3.687e-4*x**2
        initial_carbonate_equilibria['A2'] = salinity_to_A2(initial_carbonate_equilibria['Salinity'])
        salinity_to_B1 = lambda x: -530.123*x**0.5 - 6.103*x
        initial_carbonate_equilibria['B1'] = salinity_to_B1(initial_carbonate_equilibria['Salinity'])
        salinity_to_B2 = lambda x: -772.483*x**0.5 - 20.051*x
        initial_carbonate_equilibria['B2'] = salinity_to_B2(initial_carbonate_equilibria['Salinity'])
        initial_carbonate_equilibria['C1'] = -2.0695*initial_carbonate_equilibria['Salinity']**0.5
        initial_carbonate_equilibria['C2'] = -3.3336*initial_carbonate_equilibria['Salinity']**0.5
        initial_carbonate_equilibria['pK1'] = initial_carbonate_equilibria['pK1_0'] + initial_carbonate_equilibria['A1'] + initial_carbonate_equilibria['B1']/T + initial_carbonate_equilibria['C1']*self.mod.log(T)
        initial_carbonate_equilibria['pK2'] = initial_carbonate_equilibria['pK2_0'] + initial_carbonate_equilibria['A2'] + initial_carbonate_equilibria['B2']/T + initial_carbonate_equilibria['C2']*self.mod.log(T)
        initial_carbonate_equilibria['K1'] = 10**-initial_carbonate_equilibria['pK1']
        initial_carbonate_equilibria['K2'] = 10**-initial_carbonate_equilibria['pK2']
        initial_carbonate_equilibria['Kw'] = self.mod.exp(148.96502 - 13847.26/T - 23.6521*self.mod.log(T) + (-5.977 + 118.67/T + 1.0495*self.mod.log(T))*initial_carbonate_equilibria['Salinity']**0.5 - 0.01615*initial_carbonate_equilibria['Salinity'])

        self.initial_carbonate_equilibria = initial_carbonate_equilibria

    def calculate_co2_equilibrium(self, max_iter=50):
        ice = self.initial_carbonate_equilibria
        input_parameters = self.input_parameters

        co2_init = self.Hnr*input_parameters.dic
        a = 1
        b = input_parameters.c_khco3/1000
        c = lambda co2: -(ice['Kw'] + ice['K1']*co2/1000)
        d = lambda co2: -2*co2/1000*ice['K1']*ice['K2']
        f_xn = lambda x, a,b,c,d: a*x**3 + b*x**2 + c*x + d
        df_xn = lambda x, a,b,c,d: 3*a*x**2 + 2*b*x + c 
        xn_to_co2 = lambda xn, co2: self.Hnr*input_parameters.dic*self.mod.exp(
            -input_parameters.c_k*self.salting_out_exponents['h_K'] -
            self.salting_out_exponents['h_HCO3']*co2*ice['K1']/xn -
            co2*self.salting_out_exponents['h_CO3']*ice['K1']*ice['K2']/xn**2 -
            self.salting_out_exponents['h_OH']*ice['Kw']/xn*1000
        )
        # find root of f(x) = 0
        co2 = co2_init
        for i in range(3):
            x_n = opt.newton(f_xn, x0=1e-5, fprime=df_xn, tol=1e-36, args=(a,b,c(co2),d(co2)), maxiter=max_iter)
            co2 = xn_to_co2(x_n, co2)

        co2_equilibrium_sol = {
            'final_pH': -self.mod.log10(x_n),
            'CO2': co2,
            'HCO3': co2*ice['K1']/x_n,
            'OH': ice['Kw']/x_n * 1000,
            'K': input_parameters.c_k,
        }
        co2_equilibrium_sol['CO3'] = co2_equilibrium_sol['HCO3']*ice['K2']/x_n
        co2_equilibrium_sol
        return co2_equilibrium_sol

    def calculate_dic_solution(self):
        ice = self.initial_carbonate_equilibria
        dic_electrolyte_solution = {
            'initial_pH': 7,
        }
        a = 1
        b = ice['K1'] + self.input_parameters.c_khco3/1000
        c = ice['K1']*ice['K2'] - ice['Kw']
        d = -ice['K1']*(ice['Kw'] + ice['K2']*self.input_parameters.c_khco3/1000)
        e = -ice['Kw']*ice['K2']*ice['K1']
        x_n = 1e-4
        f_xn = lambda x: a*x**4 + b*x**3 + c*x**2 + d*x + e
        df_xn = lambda x: 4*a*x**3 + 3*b*x**2 + 2*c*x + d
        # find root of f(x) = 0
        x_n = opt.newton(f_xn, x_n, fprime=df_xn, tol=1e-35)
        dic_electrolyte_solution['final_pH'] = -self.mod.log10(x_n)
        dic_electrolyte_solution['CO2'] = self.input_parameters.c_khco3/(1 + ice['K1']/x_n + ice['K1']*ice['K2']/x_n**2)
        dic_electrolyte_solution['HCO3'] = dic_electrolyte_solution['CO2']*ice['K1']/x_n
        dic_electrolyte_solution['CO3'] = dic_electrolyte_solution['HCO3']*ice['K2']/x_n
        dic_electrolyte_solution['OH'] = ice['Kw']/x_n * 1000
        dic_electrolyte_solution['K'] = self.input_parameters.c_khco3
        dic_electrolyte_solution
        return dic_electrolyte_solution

    def solve(self, phi_ext: Union[float, np.ndarray, torch.Tensor]):
        A = self.input_parameters.volumetric_surface_area
        F = self.F
        R = self.R
        T = self.T
        L = self.input_parameters.L
        Hnr = self.Hnr
        Hnr_c = lambda c1, c2: Hnr*self.mod.exp(-self.salting_out_exponents['h_OH']*c1 - self.salting_out_exponents['h_CO3']*c2 - self.salting_out_exponents['h_K']*(c1+2*c2))
        E_CO = self.electrode_reaction_kinetics['E_0_CO2']
        if self.input_parameters.method=='DIC':
            dic_electrolyte_solution = self.calculate_dic_solution()
            OH_neg = dic_electrolyte_solution['OH']
            HCO3_neg = dic_electrolyte_solution['HCO3']
            CO3_2neg = dic_electrolyte_solution['CO3']
        elif self.input_parameters.method=='CO2 eql':
            co2_equilibrium_sol = self.calculate_co2_equilibrium()
            OH_neg = co2_equilibrium_sol['OH']
            HCO3_neg = co2_equilibrium_sol['HCO3']
            CO3_2neg = co2_equilibrium_sol['CO3']

        overpotential = phi_ext - E_CO
        M = lambda k: self.mod.sqrt(k*L**2/self.diffusion_coefficients['DCO2'])

        # solve without equilibrium reactions
        k0 = A/(2*F) * self.electrode_reaction_kinetics['i_0_CO']/self.chemical_reaction_rates['c_ref'] * self.mod.exp(
            -overpotential/(self.electrode_reaction_kinetics['b_CO2']))
        eff_0 = 1/(M(k0)/self.mod.tanh(M(k0)) + k0*L/self.diffusion_coefficients['Gas diffusion layer mass transfer coefficient'])
        c00 = Hnr*self.input_parameters.p0*eff_0

        # estimating OH- concentration
        r_H2 = A*self.electrode_reaction_kinetics['i_0_H2b']/F * self.mod.exp(-phi_ext/self.electrode_reaction_kinetics['b_H2b']) # phi_ext is with respect to SHE
        c10 = OH_neg+(
            self.flow_channel_characteristics['K_L_OH']*OH_neg - 
            1/(
                1/(L*(r_H2+2*k0*c00)) +
                1/(self.flow_channel_characteristics['K_L_HCO3']*HCO3_neg)
            )
            + L*(r_H2+2*k0*c00)
        ) / (
            self.flow_channel_characteristics['K_L_OH'] + 2*self.input_parameters.eps*L*self.chemical_reaction_rates['k1f']*c00
        )

        # update CO2 concentration 
        k1 = k0 + self.input_parameters.eps*self.chemical_reaction_rates['k1f']*c10
        eff_1 = 1/(M(k1)/self.mod.tanh(M(k1)) + k1*L/self.diffusion_coefficients['Gas diffusion layer mass transfer coefficient'])
        c01 = eff_1*self.input_parameters.p0*Hnr
        # update OH- concentration
        c11 = OH_neg + (
            self.flow_channel_characteristics['K_L_OH']*OH_neg - 1/(
                1/(L*(r_H2+2*k0*c01)) + 1/(self.flow_channel_characteristics['K_L_HCO3']*HCO3_neg)
            ) + L*(r_H2+2*k0*c01) 
            ) / (
            self.flow_channel_characteristics['K_L_OH']+2*self.chemical_reaction_rates['k1f']*L*self.input_parameters.eps*c01
        )
        # solve for CO3--
        A_1 = (
            2*self.flow_channel_characteristics['K_L_CO3']*CO3_2neg + self.flow_channel_characteristics['K_L_HCO3']*HCO3_neg + self.flow_channel_characteristics['K_L_OH']*OH_neg + L*r_H2 - self.flow_channel_characteristics['K_L_OH']*c11
        ) / (2*self.flow_channel_characteristics['K_L_CO3'])
        B_2 = L*2*k0*c01 / (2*self.flow_channel_characteristics['K_L_CO3']) * self.mod.exp(-c11*(self.salting_out_exponents['h_OH']+self.salting_out_exponents['h_K']))
        C = self.salting_out_exponents['h_CO3']+2*self.salting_out_exponents['h_K']
        c20 = A_1+self.mod.log(
            1+(B_2*C * self.mod.exp(-A_1*C)) / (1+self.mod.log(self.mod.sqrt(1+B_2*C*self.mod.exp(-A_1*C))))
            ) / C
        # salting out corrected CO2 concentration
        c02 = eff_1*self.input_parameters.p0*Hnr_c(c11,c20)
        # corrected OH- concentration
        c12 = OH_neg + (
            self.flow_channel_characteristics['K_L_OH']*OH_neg - 1/(
                1/(L*(r_H2+2*k0*c02)) + 1/(self.flow_channel_characteristics['K_L_HCO3']*HCO3_neg)
            ) + L*(r_H2+2*k0*c02)
            ) / (
            self.flow_channel_characteristics['K_L_OH']+2*self.chemical_reaction_rates['k1f']*L*self.input_parameters.eps*c02
        )

        k2 = k0 + self.input_parameters.eps*self.chemical_reaction_rates['k1f']*c12
        eff_2 = 1/(
            self.mod.sqrt(
                k2*L**2/self.diffusion_coefficients['DCO2']
            ) / self.mod.tanh(self.mod.sqrt(
                k2*L**2/self.diffusion_coefficients['DCO2']
            )) + k2*L/self.diffusion_coefficients['Gas diffusion layer mass transfer coefficient']
        )
        A_1_1 = (
            2*self.flow_channel_characteristics['K_L_CO3']*CO3_2neg + self.flow_channel_characteristics['K_L_HCO3']*HCO3_neg + self.flow_channel_characteristics['K_L_OH']*OH_neg + L*r_H2 - self.flow_channel_characteristics['K_L_OH']*c12
        ) / (2*self.flow_channel_characteristics['K_L_CO3'])
        # formula for this B2 is different than for the previous B2
        B_2_1 = L*2*k0*eff_2*Hnr*self.input_parameters.p0 / (2*self.flow_channel_characteristics['K_L_CO3']) * self.mod.exp(-c12*(self.salting_out_exponents['h_OH']+self.salting_out_exponents['h_K']))
        c21 = A_1_1+self.mod.log(
            1+(
                B_2_1*(self.salting_out_exponents['h_CO3']+2*self.salting_out_exponents['h_K']) * self.mod.exp(-A_1_1*(self.salting_out_exponents['h_CO3']+2*self.salting_out_exponents['h_K']))
                ) / (
                    1+self.mod.log(self.mod.sqrt(1+B_2_1*(self.salting_out_exponents['h_CO3']+2*self.salting_out_exponents['h_K'])*self.mod.exp(-A_1_1*(self.salting_out_exponents['h_CO3']+2*self.salting_out_exponents['h_K']))))
                    )
            ) / (
            self.salting_out_exponents['h_CO3']+2*self.salting_out_exponents['h_K']
        )
        c21 = self.mod.maximum(c21, self.mod.zeros_like(c21))
        c03 = self.input_parameters.p0*Hnr_c(c12, c21)*eff_2
        potential_vs_rhe = phi_ext - R*T/F*self.mod.log(c12/OH_neg)
        co_current_density = L*c03*k0*2*F/10
        co2 = c03
        co3 = c21
        pH = self.mod.log10(c12/1000/self.initial_carbonate_equilibria['Kw'])
        fe = co_current_density*10/(co_current_density*10+(F*L*r_H2))
        current = co_current_density + (F*L*r_H2)/10 # mA/cm^2
        parasitic = c03*c12*self.input_parameters.eps*L*self.chemical_reaction_rates['k1f']
        electrode = L*k0*c03
        gdl_flux = self.diffusion_coefficients['Gas diffusion layer mass transfer coefficient']*(
            Hnr_c(c12,c21)*self.input_parameters.p0 - c03*self.mod.sqrt(
                k2*L**2/self.diffusion_coefficients['DCO2']
            ) / self.mod.tanh(
                self.mod.sqrt(k2*L**2/self.diffusion_coefficients['DCO2'])
            )
        )
        ve_rhe = (self.electrode_reaction_kinetics['E_0_CO2'] - R*T/F*self.mod.log(c12/OH_neg))/potential_vs_rhe
        ve_rhe = self.mod.minimum(ve_rhe, self.mod.ones_like(ve_rhe))
        hco3 = self.mod.minimum(
            HCO3_neg+Hnr_c(c12,c21)*self.input_parameters.p0,
            co3*10**(3-pH)/self.initial_carbonate_equilibria['K1']
        )
        ee = self.mod.minimum(self.mod.ones_like(fe), fe*ve_rhe)
        ve_she = (self.electrode_reaction_kinetics['E_0_CO2'] - R*T/F*pH) / (
            phi_ext - R*T/F*pH
        )
        ve_she = self.mod.minimum(self.mod.ones_like(ve_she), ve_she)
        ve_agcl = (self.electrode_reaction_kinetics['E_0_CO2'] - R*T/F*pH - 0.2) / (
            phi_ext - R*T/F*pH - 0.2
        )
        ve_agcl = self.mod.minimum(self.mod.ones_like(ve_agcl), ve_agcl)
        eeshe = fe*ve_she
        eeagcl = fe*ve_agcl
        solubility = Hnr_c(c12,c21)/Hnr
        return {
            'phi_ext': phi_ext,
            'k0': k0,
            'eff_0': eff_0,
            'c00': c00,
            'r_H2': r_H2,
            'c10': c10,
            'k1': k1,
            'eff_1': eff_1,
            'c01': c01,
            'c11': c11,
            'A_1': A_1,
            'B_2': B_2,
            'c20': c20,
            'c02': c02,
            'c12': c12,
            'k2': k2,
            'eff_2': eff_2,
            'c03': c03,
            'c21': c21,
            'A_1_1': A_1_1,
            'B_2_1': B_2_1,
            'potential_vs_rhe': potential_vs_rhe,
            'co_current_density': co_current_density,
            'current_density': current,
            'co2': co2,
            'co3': co3,
            'pH': pH,
            'fe': fe,
            'parasitic': parasitic,
            'electrode': electrode,
            'gdl_flux': gdl_flux,
            've_rhe': ve_rhe,
            'hco3': hco3,
            'ee': ee,
            've_she': ve_she,
            've_agcl': ve_agcl,
            'eeshe': eeshe,
            'eeagcl': eeagcl,
            'solubility': solubility
        }

    def solve_current(
        self,
        target_current_density: Union[float, torch.Tensor],
        voltage_bounds: Tuple = (-2, 0),
        return_residual: bool = False,
        grid_size: int = 500
    ):
        V = target_current_density.item() if isinstance(target_current_density, torch.Tensor) else target_current_density
        if self.mod is torch:
            S = self.copy(keep_grad=False)
        else:
            S = self
        phi = np.linspace(*sorted(voltage_bounds, reverse=True), grid_size) # monotonically decreasing
        I = S.solve(phi)['current_density'] # monotonically increasing
        if not I[0] < V < I[-1]:
            raise ValueError(f'Target current density {V} is not within the bounds of the voltage sweep {voltage_bounds}')
        idx = np.searchsorted(I, V) - 1
        if self.mod==torch:
            phi = torch.linspace(phi[idx], phi[idx+1], grid_size)
        else:
            phi = np.linspace(phi[idx], phi[idx+1], grid_size)
        # now solve again and pick the closest one
        res = self.solve(phi)
        if self.mod==torch:
            idx = torch.argmin(torch.abs(res['current_density'] - V))
        else:
            idx = np.argmin(np.abs(res['current_density'] - V))
        out = {}
        for k, v in res.items():
            out[k] = v[idx]
        if return_residual:
            return out, res['current_density'][idx] - V
        return out

    def copy(self, keep_grad: bool = True):
        if keep_grad:
            return System(
                input_parameters=self.input_parameters,
                diffusion_coefficients=self.diffusion_coefficients,
                salting_out_exponents=self.salting_out_exponents,
                electrode_reaction_kinetics=self.electrode_reaction_kinetics,
                chemical_reaction_rates=self.chemical_reaction_rates,
                mod=self.mod
            )
        return System(
            input_parameters=copy_input_parameters(self.input_parameters, keep_grad=False),
            diffusion_coefficients=detach_dict(self.diffusion_coefficients),
            salting_out_exponents=detach_dict(self.salting_out_exponents),
            electrode_reaction_kinetics=detach_dict(self.electrode_reaction_kinetics),
            chemical_reaction_rates=detach_dict(self.chemical_reaction_rates),
            mod=np
        )
