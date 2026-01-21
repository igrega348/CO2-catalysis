from carbondriver import gde_multi
import torch


erc = gde_multi.electrode_reaction_kinetics | {}
erc['i_0_CO'] = torch.nn.parameter.Parameter(torch.tensor(erc['i_0_CO']))
erc['i_0_C2H4'] = torch.nn.parameter.Parameter(torch.tensor(erc['i_0_C2H4']))
erc['i_0_H2b'] = torch.nn.parameter.Parameter(torch.tensor(erc['i_0_H2b']))
erc['alpha_CO'] = torch.nn.parameter.Parameter(torch.tensor(erc['alpha_CO']))
erc['alpha_C2H4'] = torch.nn.parameter.Parameter(torch.tensor(erc['alpha_C2H4']))
erc['alpha_H2b'] = torch.nn.parameter.Parameter(torch.tensor(erc['alpha_H2b']))
ph_model = gde_multi.System(
    diffusion_coefficients=gde_multi.diffusion_coefficients, 
    salting_out_exponents=gde_multi.salting_out_exponents, 
    electrode_reaction_kinetics=erc,
    electrode_reaction_potentials=gde_multi.electrode_reaction_potentials,
    chemical_reaction_rates=gde_multi.chemical_reaction_rates,
        )  

eps = torch.tensor(0.5)
r = torch.tensor(3e-8)
L = torch.tensor(1e-5)
thetas = {
    'CO': torch.tensor(0.3),
    'C2H4': torch.tensor(0.3),
    'H2b': torch.tensor(0.4)
}
gdl_mass_transfer_coefficient = torch.tensor(0.015)

solution = ph_model.solve_current(
    i_target=233,
    eps=eps,
    r=r,
    L=L,
    thetas=thetas,
    gdl_mass_transfer_coeff=gdl_mass_transfer_coefficient,
    grid_size=1000,
    voltage_bounds=(-1.25,0)
)

print(solution)
