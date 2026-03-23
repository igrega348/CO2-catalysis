from carbondriver import gde_multi
import torch


def _make_system(system_phase='solid'):
    erc = gde_multi.electrode_reaction_kinetics | {}
    erc['i_0_CO'] = torch.nn.parameter.Parameter(torch.tensor(erc['i_0_CO']))
    erc['i_0_C2H4'] = torch.nn.parameter.Parameter(torch.tensor(erc['i_0_C2H4']))
    erc['i_0_H2b'] = torch.nn.parameter.Parameter(torch.tensor(erc['i_0_H2b']))
    erc['alpha_CO'] = torch.nn.parameter.Parameter(torch.tensor(erc['alpha_CO']))
    erc['alpha_C2H4'] = torch.nn.parameter.Parameter(torch.tensor(erc['alpha_C2H4']))
    erc['alpha_H2b'] = torch.nn.parameter.Parameter(torch.tensor(erc['alpha_H2b']))
    return gde_multi.System(
        diffusion_coefficients=gde_multi.diffusion_coefficients,
        salting_out_exponents=gde_multi.salting_out_exponents,
        electrode_reaction_kinetics=erc,
        electrode_reaction_potentials=gde_multi.electrode_reaction_potentials,
        chemical_reaction_rates=gde_multi.chemical_reaction_rates,
        system_phase=system_phase,
    )


def _base_inputs():
    eps = torch.tensor(0.5)
    r = torch.tensor(3e-8)
    L = torch.tensor(1e-5)
    thetas = {
        'CO': torch.tensor(0.3),
        'C2H4': torch.tensor(0.3),
        'H2b': torch.tensor(0.4),
    }
    gdl_mass_transfer_coefficient = torch.tensor(0.015)
    return eps, r, L, thetas, gdl_mass_transfer_coefficient


def test_solid_mode_defaults_to_co2_equilibrium_method():
    ph_model = _make_system(system_phase='solid')
    assert ph_model.method == 'CO2 eql'


def test_liquid_mode_forces_dic_method():
    ph_model = _make_system(system_phase='liquid')
    assert ph_model.method == 'DIC'


def test_solve_current_runs_in_solid_mode_with_gdl_flux_key():
    ph_model = _make_system(system_phase='solid')
    eps, r, L, thetas, gdl_mass_transfer_coefficient = _base_inputs()

    solution = ph_model.solve_current(
        i_target=233,
        eps=eps,
        r=r,
        L=L,
        thetas=thetas,
        gdl_mass_transfer_coeff=gdl_mass_transfer_coefficient,
        grid_size=1000,
        voltage_bounds=(-1.25, 0),
    )

    assert 'gdl_flux' in solution
    assert torch.isfinite(solution['current_density']).all()


def test_solve_current_runs_in_liquid_mode_with_zero_gdl_flux():
    ph_model = _make_system(system_phase='liquid')
    eps, r, L, thetas, gdl_mass_transfer_coefficient = _base_inputs()

    solution = ph_model.solve_current(
        i_target=233,
        eps=eps,
        r=r,
        L=L,
        thetas=thetas,
        gdl_mass_transfer_coeff=gdl_mass_transfer_coefficient,
        grid_size=1000,
        voltage_bounds=(-1.25, 0),
    )

    assert 'gdl_flux' in solution # Make sure we still have the gdl_flux key in liquid mode
    assert torch.allclose(solution['gdl_flux'], torch.zeros_like(solution['gdl_flux'])) # Make sure we actually have zero gdl flux in liquid mode
    assert torch.isfinite(solution['current_density']).all() # Make sure we still get a valid current density solution in liquid mode