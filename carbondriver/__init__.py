from .models import PhModel, MLPModel, MultitaskGPModel, MultitaskGPhysModel
from .train import train_model_ens, train_GP_model, train_GP_Ph_model
from .loaders import load_data, normalize_df_torch, feature_stats
from .config import default_config
import pandas as pd
import torch
#torch.autograd.set_detect_anomaly(True)
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.acquisition.objective import ScalarizedPosteriorTransform

from botorch.models.ensemble import EnsembleModel

class GDEOptimizer():
    """
    Class to optimize gas diffusion electrodes experimental parameters based with Bayesian optimization using various models.
    """
    
    def __init__(self, model_name="GP+Ph", aquisition="EI", quantity="FE (Eth)", maximize=True, output_dir="./out", config=default_config, bounds=None):
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
        # Stats for normalization of feature columns (set in get_predictor when normalize=True)
        self._feature_means = None  # torch.Tensor of shape (d,)
        self._feature_stds = None   # torch.Tensor of shape (d,)

    @property
    def bounds(self):
        """
        Get the bounds for the optimization based on the data if not specified in the declaration.
        """
        if self._bounds is None:
            bds_max = self.df.iloc[:, :-2].values.max(axis=0)
            bds_min = self.df.iloc[:, :-2].values.min(axis=0)
            raw_bounds = torch.tensor([bds_min, bds_max], dtype=torch.float32)
            # Debug: show computed raw bounds
            print(f"[bounds] raw min: {raw_bounds[0].tolist()} raw max: {raw_bounds[1].tolist()}")
            return raw_bounds
        else:
            return self._bounds
        
    def load(path):
        '''
        Load pretrained model.
        '''
        raise NotImplementedError

    def get_predictor(self, new_data):
        '''
        Train and return the new predictor based on the new data.
        '''

        if isinstance(new_data, pd.Series):
            new_data = new_data.to_frame().T
        
        self.df = pd.concat([self.df, new_data], axis=0)

        if self.config['normalize']:
            # Normalize inputs; outputs remain unnormalized
            X, y, means, stds, _ = normalize_df_torch(self.df)
            # Store feature-only means/stds for later bound transforms
            self._feature_means, self._feature_stds = feature_stats(self.df, means, stds, as_torch=True)
            # Debug prints
            print(f"[get_predictor] normalize=True | X shape: {tuple(X.shape)}, y shape: {tuple(y.shape)}")
            print(f"[get_predictor] feature means: {self._feature_means.tolist()}")
            print(f"[get_predictor] feature stds: {self._feature_stds.tolist()}")
            if self.model == PhModel:
                # Pass Zero_eps_thickness stats so model can denormalize that feature
                mu = float(means['Zero_eps_thickness'])
                sigma = float(stds['Zero_eps_thickness'])
                model_factory = lambda: PhModel(
                    zlt_mu=mu,
                    zlt_sigma=sigma,
                    current_target=233,
                    config=self.config,
                )
            else:
                model_factory = self.model
        elif self.config['normalize'] is False and self.model == PhModel:
            # No normalization: compute tensors directly but still provide stats for completeness
            X, y = self.df.iloc[:, :-2].values, self.df.iloc[:, -2:].values
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
            X, y = self.df.iloc[:, :-2].values, self.df.iloc[:, -2:].values
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            model_factory = self.model


        _, model = train_model_ens(X, y, model_factory, DNAME=self.output_dir, i=self.i, num_iter=self.config["num_iter"], plot=self.config["make_plots"])

        class Predictor(EnsembleModel):
            def __init__(self):
                super().__init__()
                self._num_outputs = 1
        
            def forward(self, X: torch.Tensor):
                # Expect X of shape (batch, q, d). We handle q=1 by squeezing.
                print(f"[Predictor.forward] received X shape: {tuple(X.shape)}")
                if X.dim() == 3 and X.shape[1] == 1:
                    X_in = X.squeeze(1)  # (batch, d)
                elif X.dim() == 2:
                    X_in = X  # already (batch, d)
                else:
                    # Attempt to reshape conservatively: collapse all but last dim into batch
                    X_in = X.view(-1, X.shape[-1])
                print(f"[Predictor.forward] using X_in shape: {tuple(X_in.shape)}")

                model_output = model(X_in)
                try:
                    print(f"[Predictor.forward] model_output shape: {tuple(model_output.shape)}")
                except Exception:
                    pass

                # Handle different output dimensions
                if model_output.dim() == 3:
                    # Output is [ensemble, batch, features] -> [batch, ensemble, features]
                    result = model_output.permute((1, 0, 2))
                elif model_output.dim() == 4:
                    # Output is [ensemble, batch, features, extra] -> [batch, ensemble, features, extra]
                    result = model_output.permute((2, 0, 1, 3))
                else:
                    # Fallback: return as is
                    result = model_output
                
                #print(f"DEBUG - Final result shape: {result.shape}")
                try:
                    print(f"[Predictor.forward] result shape: {tuple(result.shape)}")
                except Exception:
                    pass
                return result
                
        return Predictor()

    def _get_acquisition_function(self, predictor):
        """
        Get the acquisition function based on the specified acquisition type.
        """
        if self.aquisition == "EI":
            best_f = torch.tensor(self.df[self.quantity].max())
            print(f"[_get_acquisition_function] Using LogEI | best_f={best_f.item():.6f} | maximize={self.maximize}")
            return LogExpectedImprovement(
                predictor,
                best_f=best_f,
                maximize=self.maximize
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

        print(f"[step] normalize flag: {self.config.get('normalize', False)}")

        predictor = self.get_predictor(new_data)

        AF = self._get_acquisition_function(predictor)

        # Determine raw bounds (always in original feature scale)
        raw_bounds = self.bounds if bounds is None else bounds
        if not isinstance(raw_bounds, torch.Tensor):
            import numpy as _np
            if isinstance(raw_bounds, (list, tuple)):
                raw_bounds = torch.from_numpy(_np.asarray(raw_bounds, dtype=_np.float32))
            else:
                raw_bounds = torch.tensor(raw_bounds, dtype=torch.float32)
        assert raw_bounds.shape[0] == 2, "Bounds should have shape (2, d)"
        d = raw_bounds.shape[1]
        print(f"[step] raw bounds min: {raw_bounds[0].tolist()} max: {raw_bounds[1].tolist()}")

        # Select the output column index for the quantity by name from the last two columns
        target_cols = list(self.df.columns[-2:])
        try:
            target_idx = target_cols.index(self.quantity)
        except ValueError:
            raise ValueError(f"Quantity '{self.quantity}' not found in output columns {target_cols}")
        print(f"[step] optimizing target column index (target_idx): {target_idx} for quantity '{self.quantity}'")

        # If normalized, convert bounds to normalized space using stored stats
        use_norm = bool(self.config.get('normalize', False))
        if use_norm:
            if self._feature_means is None or self._feature_stds is None:
                # Fall back: compute from current df
                self._feature_means, self._feature_stds = feature_stats(self.df, as_torch=True)
            means, stds = self._feature_means, self._feature_stds
            # Prevent division by zero
            stds_safe = torch.where(stds == 0, torch.tensor(1.0, dtype=stds.dtype), stds)
            bounds_norm = torch.stack([
                (raw_bounds[0] - means) / stds_safe,
                (raw_bounds[1] - means) / stds_safe,
            ], dim=0)
            print(f"[step] normalized bounds min: {bounds_norm[0].tolist()} max: {bounds_norm[1].tolist()}")
            opt_bounds = bounds_norm
        else:
            opt_bounds = raw_bounds

        # Wrap AF to be robust to different output shapes and pick the correct target
        def acq_wrapper(x: torch.Tensor):
            # x is expected to be (batch, q, d) by botorch
            vals = AF(x)
            try:
                shape_info = tuple(vals.shape)
            except Exception:
                shape_info = 'unknown'
            print(f"[acq_wrapper] AF input shape: {tuple(x.shape)} | AF output shape: {shape_info}")
            if isinstance(vals, torch.Tensor):
                b = x.shape[0]
                # Case: (batch, ensemble, m)
                if vals.dim() == 3 and vals.shape[0] == b:
                    if vals.shape[2] <= target_idx:
                        raise RuntimeError(f"AF returned shape {tuple(vals.shape)} which has no target index {target_idx}")
                    vals = vals[:, :, target_idx].mean(dim=1)  # (batch,)
                    return vals
                # Case: (batch, m)
                if vals.dim() == 2 and vals.shape[0] == b:
                    if vals.shape[1] <= target_idx:
                        raise RuntimeError(f"AF returned shape {tuple(vals.shape)} which has no column {target_idx}")
                    vals = vals[:, target_idx]
                    return vals
                # Fallback: per-sample evaluation with reduction to scalar
                out_list = []
                for i in range(b):
                    vi = AF(x[i:i+1])
                    if isinstance(vi, torch.Tensor):
                        vi = vi.squeeze()
                        if vi.dim() == 0:
                            pass  # already scalar
                        elif vi.dim() == 1:
                            # Could be (m,), pick target
                            if vi.numel() <= target_idx:
                                raise RuntimeError(f"AF per-sample 1D output {tuple(vi.shape)} has no index {target_idx}")
                            vi = vi[target_idx]
                        elif vi.dim() == 2:
                            # Likely (ensemble, m)
                            if vi.shape[1] <= target_idx:
                                raise RuntimeError(f"AF per-sample 2D output {tuple(vi.shape)} has no target index {target_idx}")
                            vi = vi[:, target_idx].mean()
                        else:
                            # Collapse all to scalar
                            vi = vi.view(-1)[0]
                    out_list.append(vi)
                return torch.stack(out_list, dim=0)
            return vals

        next_experiment, _ = optimize_acqf(
            acq_function=acq_wrapper,
            bounds=opt_bounds,
            q=1,
            num_restarts=20,
            raw_samples=30,
            options={}, 
        )

        # Denormalize the candidate if optimization was done in normalized space
        x_candidate = next_experiment
        if use_norm:
            x_candidate = next_experiment * self._feature_stds + self._feature_means
            print(f"[step] denormalized candidate: {x_candidate.detach().cpu().numpy().flatten().tolist()}")
        else:
            print(f"[step] candidate (raw): {x_candidate.detach().cpu().numpy().flatten().tolist()}")

        self.i += 1

        # Evaluate AF at the (normalized) next point for returning EI value
        ei_val = acq_wrapper(next_experiment)

        return ei_val, x_candidate.detach().cpu().numpy().flatten()

    def step_within_data(self, new_data, possible_data):

        predictor = self.get_predictor(new_data)

        if self.config['normalize']:
            X, y, means, stds, _ = normalize_df_torch(possible_data)
            # Persist feature stats for consistency when mixing calls
            self._feature_means, self._feature_stds = feature_stats(possible_data, means, stds, as_torch=True)
            print(f"[step_within_data] normalize=True | X shape: {tuple(X.shape)}, y shape: {tuple(y.shape)}")
        else:
            X, y = possible_data.iloc[:, :-2].values, possible_data.iloc[:, -2:].values
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

        AF = self._get_acquisition_function(predictor)
        
        scores = AF(X.unsqueeze(1))
        try:
            print(f"[step_within_data] AF scores shape: {tuple(scores.shape)}")
        except Exception:
            pass
        
        self.i += 1

        # Determine index of target for selection
        # Determine target index by name (from last two columns)
        target_cols = list(self.df.columns[-2:])
        try:
            target_idx = target_cols.index(self.quantity)
        except ValueError:
            raise ValueError(f"Quantity '{self.quantity}' not found in output columns {target_cols}")
        # Robust selection from AF outputs
        if isinstance(scores, torch.Tensor) and scores.dim() == 2:
            if scores.shape[1] <= target_idx:
                raise RuntimeError(f"AF scores shape {tuple(scores.shape)} has no column {target_idx}")
            target_scores = scores[:, target_idx]
        else:
            target_scores = scores.squeeze()
        return target_scores.max(), target_scores.argmax()
        
