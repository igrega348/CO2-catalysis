import gpytorch
from .models import PhModel, MLPModel, MultitaskGPModel, BoTorchGP, MultitaskGPhysModel
from .train import train_model_ens, train_GP_model, train_GP_Ph_model
from .loaders import normalize_df_torch, feature_stats
from .config import default_config
import pandas as pd
import torch
import numpy as np
#torch.autograd.set_detect_anomaly(True)
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.acquisition.objective import ScalarizedPosteriorTransform

from botorch.models.ensemble import EnsembleModel

class GDEOptimizer():
    """
    Class to optimize gas diffusion electrodes experimental parameters based with Bayesian optimization using various models.
    """
    
    def __init__(self, model_name="GP+Ph", aquisition="EI", quantity="FE (Eth)", maximize=True, output_dir="./out", config=default_config, bounds=None, input_labels=None, output_labels=None):
        """
        Initialize the optimizer with the specified model and acquisition function.

        :param model_name: Name of the model to use (e.g., 'GP', 'Ph', 'MLP', 'GP+Ph')
        :param aquisition: Acquisition function to use (e.g., 'EI' for Log Expected Improvement)
        :param quantity: The quantity to optimize (e.g., 'FE (Eth)')
        :param maximize: Whether to maximize or minimize the quantity
        :param output_dir: Directory to save output files
        :param config: Configuration dictionary with parameters for training and normalization
        :param bounds: Bounds for the optimization, should be a tensor of shape (2, num_features)
        
        """
        
        if model_name == 'GP':
            self.model = MultitaskGPModel
        elif model_name == 'Ph':
            self.model = PhModel
        elif model_name == 'MLP':
            self.model = MLPModel
        elif model_name == 'GP+Ph':
            self.model = MultitaskGPhysModel
        else:
            raise ValueError
            
        if aquisition == "EI":
            self.aquisition = aquisition
        else:
            raise ValueError("Only EI is supported for now.")

        self.output_dir = output_dir

        self.config = config

        self.maximize = maximize

        self.quantity = quantity

        self.i = 0

        self.df = pd.DataFrame()

        self._bounds = bounds

        if input_labels is None:
            self.input_labels = ['AgCu Ratio',  'Naf vol (ul)',  'Sust vol (ul)',  'Zero_eps_thickness',  'Catalyst mass loading']
        else:
            # If input_labels are provided, use them
            self.input_labels = input_labels

        if output_labels is None:
            self.output_labels = ['FE (Eth)', 'FE (CO)']
        else:
            # If output_labels are provided, use them
            self.output_labels = output_labels

        # Stats for normalization of feature columns (set in get_predictor when normalize=True)
        self._means = None  # torch.Tensor of shape (d,)
        self._stds = None   # torch.Tensor of shape (d,)

    @property
    def bounds(self):
        """
        Get the bounds for the optimization based on the data if not specified in the declaration.
        """
        if self._bounds is None:
            bds_max = self.df.loc[:, self.input_labels].values.max(axis=0)
            bds_min = self.df.loc[:, self.input_labels].values.min(axis=0)
            raw_bounds = torch.tensor([bds_min, bds_max], dtype=torch.float32)
            # Debug: show computed raw bounds
            #print(f"[bounds] raw min: {raw_bounds[0].tolist()} raw max: {raw_bounds[1].tolist()}")
            return raw_bounds
        else:
            return self._bounds
        
    def load(path):
        '''
        Load pretrained model.
        '''
        raise NotImplementedError

    def get_predictor(self, new_data):
        """
        Train and return the new predictor based on the new data.
        """

        if isinstance(new_data, pd.Series):
            new_data = new_data.to_frame().T

        self.df = pd.concat([self.df, new_data], axis=0)

        # Special handling for GP and GP+Ph models: these use gpytorch training functions
        # (they are not compatible with the ensemble training pipeline used for MLP/Ph).
        if self.model == MultitaskGPModel:
            # Normalize if requested
            if self.config['normalize']:
                X, y, self._means, self._stds, _ = normalize_df_torch(
                    self.df, self.input_labels, self.output_labels
                )
            else:
                X, y = (
                    self.df.loc[:, self.input_labels].values,
                    self.df.loc[:, self.output_labels].values,
                )
                X = torch.tensor(X, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)

            # Train GP and return BoTorch-compatible model
            _, _, gp_model, likelihood = train_GP_model(
                X,
                y,
                num_iter=self.config["num_iter"],
                DNAME=self.output_dir,
                i=self.i,
                plot=self.config["make_plots"],
            )

            # Return BoTorch-compatible wrapper
            return BoTorchGP(gp_model, likelihood)

        elif self.model == MultitaskGPhysModel:
            # GP+Physics: Ph model constructor must be provided to the GP+Ph trainer.
            if self.config['normalize']:
                X, y, self._means, self._stds, _ = normalize_df_torch(
                    self.df, self.input_labels, self.output_labels
                )
                # zero-eps stats for PhModel
                mu = float(self._means['Zero_eps_thickness'])
                sigma = float(self._stds['Zero_eps_thickness'])
            else:
                X, y = (
                    self.df.loc[:, self.input_labels].values,
                    self.df.loc[:, self.output_labels].values,
                )
                X = torch.tensor(X, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)
                mu = float(self.df['Zero_eps_thickness'].mean())
                sigma = float(self.df['Zero_eps_thickness'].std(ddof=0))

            ph_model_constructor = lambda: PhModel(
                zlt_mu=mu,
                zlt_sigma=sigma,
                current_target=233,
                config=self.config,
            )

            # Train GP+Ph and return BoTorch-compatible model
            _, _, gp_model, likelihood = train_GP_Ph_model(
                X,
                y,
                ph_model_constructor,
                num_iter=self.config["num_iter"],
                DNAME=self.output_dir,
                i=self.i,
                plot=self.config["make_plots"],
            )

            # Return BoTorch-compatible wrapper
            return BoTorchGP(gp_model, likelihood)

        # Handle ensemble models (MLP and Ph)
        else:
            if self.config['normalize']:
                # Normalize inputs; outputs remain unnormalized
                X, y, self._means, self._stds, _ = normalize_df_torch(
                    self.df, self.input_labels, self.output_labels
                )

                if self.model == PhModel:
                    # Pass Zero_eps_thickness stats so model can denormalize that feature
                    mu = float(self._means['Zero_eps_thickness'])
                    sigma = float(self._stds['Zero_eps_thickness'])
                    model_factory = lambda: PhModel(
                        zlt_mu=mu,
                        zlt_sigma=sigma,
                        current_target=233,
                        config=self.config,
                    )

                elif self.model == MLPModel:
                    # NEW: MLP output dimension matches number of output labels
                    n_out = len(self.output_labels)
                    model_factory = lambda: MLPModel(n_outputs=n_out)

                else:
                    # MLP is not used here; fall back to given constructor
                    model_factory = self.model

            elif self.config['normalize'] is False and self.model == PhModel:
                # No normalization: compute tensors directly but still provide stats for completeness
                X, y = (
                    self.df.loc[:, self.input_labels].values,
                    self.df.loc[:, self.output_labels].values,
                )
                X = torch.tensor(X, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)
                # Means/stds from raw data for feature-wise info (won't be used if normalize=False)
                mu = float(self.df['Zero_eps_thickness'].mean())
                sigma = float(self.df['Zero_eps_thickness'].std(ddof=0))
                model_factory = lambda: PhModel(
                    zlt_mu=mu,
                    zlt_sigma=sigma,
                    current_target=233,
                    config=self.config,
                )

            else:
                # No normalization for MLP model or other cases
                X, y = (
                    self.df.loc[:, self.input_labels].values,
                    self.df.loc[:, self.output_labels].values,
                )
                X = torch.tensor(X, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)

                if self.model == MLPModel:
                    # NEW: MLP with dynamic number of outputs
                    n_out = len(self.output_labels)
                    model_factory = lambda: MLPModel(n_outputs=n_out)
                else:
                    model_factory = self.model

            _, model = train_model_ens(
                X,
                y,
                model_factory,
                DNAME=self.output_dir,
                i=self.i,
                num_iter=self.config["num_iter"],
                plot=self.config["make_plots"],
            )

            class Predictor(EnsembleModel):
                def __init__(self):
                    super().__init__()
                    self._num_outputs = 1

                def forward(self, X: torch.Tensor):
                    # Expect X of shape (batch, q, d). We handle q=1 by squeezing.
                    if X.dim() == 3 and X.shape[1] == 1:
                        X_in = X.squeeze(1)  # (batch, d)
                    elif X.dim() == 2:
                        X_in = X  # already (batch, d)
                    else:
                        # Attempt to reshape conservatively: collapse all but last dim into batch
                        X_in = X.view(-1, X.shape[-1])

                    model_output = model(X_in)
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

            return Predictor()

    def _get_acquisition_function(self, predictor):
        """
        Get the acquisition function based on the specified acquisition type.
        """
        if self.aquisition == "EI":
            best_f = torch.tensor(self.df[self.quantity].max())
            # Select the correct output index (from the last two columns)
            try:
                target_idx = self.output_labels.index(self.quantity)
            except ValueError:
                raise ValueError(f"Quantity '{self.quantity}' not found in output columns {self.output_labels}")

            # Check if this is a GP model (BoTorchGP wrapper) or ensemble model
            if hasattr(predictor, 'model') and hasattr(predictor.model, 'likelihood'):
                # This is a GP model - use posterior transform
                weights = torch.zeros(2, dtype=torch.float32)
                weights[target_idx] = 1.0
                post_tf = ScalarizedPosteriorTransform(weights=weights)
                #print(f"[_get_acquisition_function] Using LogEI with posterior transform | best_f={best_f.item():.6f} | target_idx={target_idx} | maximize={self.maximize}")
                return LogExpectedImprovement(
                    predictor,
                    best_f=best_f,
                    maximize=self.maximize,
                    posterior_transform=post_tf,
                )
            else:
                # This is an ensemble model - use LogEI without posterior transform
                # The acq_wrapper will handle target selection
                #print(f"[_get_acquisition_function] Using LogEI for ensemble | best_f={best_f.item():.6f} | target_idx={target_idx} | maximize={self.maximize}")
                return LogExpectedImprovement(
                    predictor,
                    best_f=best_f,
                    maximize=self.maximize,
                )
        else:
            raise ValueError("Unsupported acquisition function.")
    
    def step(self, new_data, bounds=None):

        """
        Perform a step in the optimization process using the new data and bounds.
        
        :param new_data: New data to be added to the existing data for training the model
        :param bounds: Optional bounds for the optimization

        :return: The acquisition function value and the next experiment parameters
        """

        #print(f"[step] normalize flag: {self.config.get('normalize', False)}")

        predictor = self.get_predictor(new_data)

        AF = self._get_acquisition_function(predictor)

        # Determine raw bounds (always in original feature scale)
        # Use the property-created tensor by default. If the caller supplied `bounds`,
        # require that it already be a torch.Tensor.
        raw_bounds = self.bounds if bounds is None else bounds
        if bounds is not None and not isinstance(raw_bounds, torch.Tensor):
            raise TypeError(
                "bounds must be a torch.Tensor of shape (2, d). "
                "Convert lists/arrays with torch.as_tensor(..., dtype=torch.float32) before calling step."
            )
        assert raw_bounds.shape[0] == 2, "Bounds should have shape (2, d)"
        #print(f"[step] raw bounds min: {raw_bounds[0].tolist()} max: {raw_bounds[1].tolist()}")

        # Select the output column index for the quantity by name from the last two columns

        try:
            target_idx =self.output_labels.index(self.quantity)
        except ValueError:
            raise ValueError(f"Quantity '{self.quantity}' not found in output columns {self.output_labels}")
        #print(f"[step] optimizing target column index (target_idx): {target_idx} for quantity '{self.quantity}'")

        # If normalized, convert bounds to normalized space using stored stats
        # Require an explicit boolean in the config to avoid surprising truthiness.
        if 'normalize' not in self.config:
            raise ValueError("config key 'normalize' not specified. Please set config['normalize'] to True or False.")

        if self.config['normalize'] is True:
            means, stds = self._means[self.input_labels], self._stds[self.input_labels]  # feature-only stats
            # Prevent division by zero
            stds[stds < 1e-10] = 1.0
            bounds_norm = torch.stack([
                (raw_bounds[0] - means.values) / stds.values,
                (raw_bounds[1] - means.values) / stds.values,
            ], dim=0)
            #print(f"[step] normalized bounds min: {bounds_norm[0].tolist()} max: {bounds_norm[1].tolist()}")
            opt_bounds = bounds_norm.float()
        else:
            opt_bounds = raw_bounds

        def AF_q(x):
            vals = AF(x)
            # vals can be:
            #  - 1D: (batch,) already scalar per point
            #  - 2D: (batch, m) for m outputs
            if vals.dim() == 1:
                return vals
            if vals.dim() == 2:
                if vals.size(1) == 1:
                    return vals.squeeze(1)
                if target_idx >= vals.size(1):
                    raise RuntimeError(
                        f"Acquisition returned {vals.size(1)} outputs but target_idx={target_idx}"
                    )
                return vals[:, target_idx]
            # Fallback: flatten all but batch dim and take first column
            return vals.view(vals.shape[0], -1)[:, 0]

        next_experiment, _ = optimize_acqf(
            acq_function=AF_q,
            bounds=opt_bounds,
            q=1,
            num_restarts=20,
            raw_samples=30,
            options={}, 
        )

        if self.config['normalize']:
            # Denormalize the candidate if optimization was done in normalized space
            x_candidate = next_experiment * stds.values +  means.values

        else:
            x_candidate = next_experiment

        self.i += 1

        # Evaluate AF at the (normalized) next point for returning EI value
        ei_val = AF_q(next_experiment)

        return ei_val, pd.Series(x_candidate.detach().cpu().numpy().flatten(), index=self.input_labels)

    def step_within_data(self, new_data, possible_data):
        predictor = self.get_predictor(new_data)

        if self.config['normalize']:
            X, y, _, _, _ = normalize_df_torch(possible_data, self.input_labels, self.output_labels, means=self._means, stds=self._stds)
            # Persist feature stats for consistency when mixing calls
            #print(f"[step_within_data] normalize=True | X shape: {tuple(X.shape)}, y shape: {tuple(y.shape)}")
        else:
            X, y = possible_data.loc[:, self.input_labels].values, possible_data.loc[:, self.output_labels].values
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

        AF = self._get_acquisition_function(predictor)
        scores = AF(X.unsqueeze(1))
        #print(scores)
        self.i += 1

        # Determine index of target for selection
        # Determine target index by name (from last two columns)
        try:
            target_idx = self.output_labels.index(self.quantity)
        except ValueError:
            raise ValueError(f"Quantity '{self.quantity}' not found in output columns {self.output_labels}")
        # Robust selection from AF outputs
        if isinstance(scores, torch.Tensor) and scores.dim() == 2:
            if scores.shape[1] <= target_idx:
                raise RuntimeError(f"AF scores shape {tuple(scores.shape)} has no column {target_idx}")
            target_scores = scores[:, target_idx]
        else:
            target_scores = scores.squeeze()
        return target_scores.max(), target_scores.argmax()
        
