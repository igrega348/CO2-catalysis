from typing import Tuple, Optional, Dict, Any
import torch
import gpytorch
from carbondriver import gde_multi
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultitaskMultivariateNormal
from botorch.models.ensemble import EnsembleModel

class PhModel(torch.nn.Module):
    '''
    Model for predicting the Faradaic efficiency of CO and C2H4 on a catalyst.

    The model is a neural network that takes in the following inputs:
    - AgCu Ratio
    - Nafion volume (ul)
    - Sustain volume (ul)
    - Zero_eps_thickness
    - Catalyst mass loading

    The model outputs the Faradaic efficiency of CO and C2H4.

    Values of particle radius r and porosity ε are in
    sensible ranges (40 nm < r < 65 nm, 0.5 < ε < 0.8).
    '''
    
    def __init__(
        self,
        zlt_mu: float,
        zlt_sigma: float,
        current_target: float = 200,
        dropout: float = 0.1,
        ldim: int = 64,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(5, ldim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ldim, ldim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ldim, ldim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ldim, 6)
        )

        erc = gde_multi.electrode_reaction_kinetics | {}
        erc['i_0_CO'] = torch.nn.parameter.Parameter(torch.tensor(erc['i_0_CO']))
        erc['i_0_C2H4'] = torch.nn.parameter.Parameter(torch.tensor(erc['i_0_C2H4']))
        erc['i_0_H2b'] = torch.nn.parameter.Parameter(torch.tensor(erc['i_0_H2b']))
        erc['alpha_CO'] = torch.nn.parameter.Parameter(torch.tensor(erc['alpha_CO']))
        erc['alpha_C2H4'] = torch.nn.parameter.Parameter(torch.tensor(erc['alpha_C2H4']))
        erc['alpha_H2b'] = torch.nn.parameter.Parameter(torch.tensor(erc['alpha_H2b']))
        self.ph_model = gde_multi.System(
            diffusion_coefficients=gde_multi.diffusion_coefficients, 
            salting_out_exponents=gde_multi.salting_out_exponents, 
            electrode_reaction_kinetics=erc,
            electrode_reaction_potentials=gde_multi.electrode_reaction_potentials,
            chemical_reaction_rates=gde_multi.chemical_reaction_rates,
        )
        self.softmax = torch.nn.Softmax(dim=1)
        # zero-eps thickness normalization stats
        self.zlt_mu = float(zlt_mu)
        self.zlt_sigma = float(zlt_sigma)
        self.current_target = current_target
        # configuration (normalization status is read from here only)
        self.config = config or {"normalize": False}
        # persistent forward counter to track how many times forward() was called
        self._forward_counter = 0

    def forward(self, x):
        # increment and print persistent forward counter
        self._forward_counter += 1
        # columns of x: AgCu Ratio, Naf vol (ul), Sust vol (ul), Zero_eps_thickness, Catalyst mass loading
        latents = self.net(x)
        r = 40e-9 * torch.exp(latents[..., [0]])
        eps = torch.sigmoid(latents[..., [1]])

        # If inputs are normalized, denormalize Zero_eps_thickness (feature index 3)
        if bool(self.config.get("normalize", False)):
            zlt = (x[..., 3] * self.zlt_sigma + self.zlt_mu).view(-1, 1)
        else:
            zlt = x[..., 3].view(-1, 1)
        # Prevent division by zero in L calculation
        L = zlt / (1 - eps)
        K_dl_factor = torch.exp(latents[..., [2]])
        thetas = self.softmax(2*latents[..., 3:])
        # CO activation must not be zero
        theta0 = thetas[...,[0]]
        theta1 = thetas[...,[1]]
        theta2 = thetas[...,[2]]
        thetas = {
            'CO': theta0,
            'C2H4': theta1,
            'H2b': theta2
        }
        gdl_mass_transfer_coefficient = K_dl_factor * self.ph_model.bruggeman(gde_multi.diffusion_coefficients['CO2'], eps) / r
        solution = self.ph_model.solve_current(
            i_target=self.current_target,
            eps=eps,
            r=r,
            L=L,
            thetas=thetas,
            gdl_mass_transfer_coeff=gdl_mass_transfer_coefficient,
            grid_size=1000,
            voltage_bounds=(-1.25,0)
        )

        out = torch.cat([solution['fe_c2h4'], solution['fe_co']], dim=-1)
        return out


class MLPModel(torch.nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int = 2, dropout: float = 0.1, ldim: int = 64):
        """
        n_outputs: number of output targets (e.g. 1 for single objective, 2 for FE(Eth)/FE(CO), etc.)
        """
        super().__init__()
        self.mlp = torch.nn.Sequential(
            # Input dimension is inferred at first forward pass
            torch.nn.Linear(n_inputs, ldim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ldim, ldim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ldim, ldim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ldim, n_outputs),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(x)

class EnsPredictor(EnsembleModel):
    def __init__(self,model_input):
        super().__init__()
        self._num_outputs = 1
        self.model_input = model_input

    def forward(self, X: torch.Tensor):
        # Expect X of shape (batch, q, d). We handle q=1 by squeezing.
        if X.dim() == 3 and X.shape[1] == 1:
            X_in = X.squeeze(1)  # (batch, d)
        elif X.dim() == 2:
            X_in = X  # already (batch, d)
        else:
            # Attempt to reshape conservatively: collapse all but last dim into batch
            X_in = X.view(-1, X.shape[-1])

        model_output = self.model_input(X_in)
        # Handle different output dimensions
        if model_output.dim() == 3:
            # Output is [ensemble, batch, features] -> [batch, ensemble, features]
            result = model_output.permute((1, 0, 2)).unsqueeze(2) 
        elif model_output.dim() == 4:
            # Output is [ensemble, batch, features, extra] -> [batch, ensemble, features, extra]
            result = model_output.permute((2, 0, 1, 3))
        else:
            # Fallback: return as is
            result = model_output

        return result


class BoTorchGP(GPyTorchModel): #Wrapper for EI acquisition function
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._num_outputs = 1

    def forward(self, x):
        if x.dim() == 3 and x.shape[1] == 1:
            X_in = x.squeeze(1)  # (batch, d)
        elif x.dim() == 2:
            X_in = x  # already (batch, d)
        return self.model(X_in)

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )
        self.num_outputs = 2
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

class MyMean(gpytorch.means.Mean):
    """
    Mean function.
    """
    def __init__(self, model: Optional[torch.nn.Module] = None, freeze_model: bool = False):
        super().__init__()
        if model is not None:
            self.model = model
        else:
            print('No model provided, using default PhModel with placeholder parameters.')
            self.model = PhModel(zlt_mu_stds=(means['Zero_eps_thickness'], stds['Zero_eps_thickness']), current_target=233)
        
        if freeze_model:
            def remove_dropout(m: torch.nn.Module):
                for child in m.children():
                    if isinstance(child, torch.nn.Dropout):
                        child.p = 0
                    else:
                        remove_dropout(child)
            
            for param in self.model.parameters():
                param.requires_grad = False
            remove_dropout(self.model)

    def forward(self, x):
        return self.model(x).squeeze()
    
class MultitaskGPhysModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, model: Optional[torch.nn.Module] = None, freeze_model: bool = False):
        super(MultitaskGPhysModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = MyMean(model=model, freeze_model=freeze_model)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
