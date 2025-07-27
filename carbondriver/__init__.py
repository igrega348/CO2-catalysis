from .models import PhModel, MLPModel, MultitaskGPModel, MultitaskGPhysModel
from .train import train_model_ens, train_GP_model, train_GP_Ph_model
from .loaders import load_data, normalize_df_torch
from .config import default_config
import pandas as pd
import torch

from botorch.acquisition.analytic import ExpectedImprovement
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
        :param aquisition: Acquisition function to use (e.g., 'EI' for Expected Improvement)
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

    @property
    def bounds(self):
        """
        Get the bounds for the optimization based on the data if not specified in the declaration.
        """
        if self._bounds is None:
            bds_max = self.df.iloc[:, :-2].values.max(axis=0)
            bds_min = self.df.iloc[:, :-2].values.min(axis=0)
            return torch.tensor([bds_min, bds_max], dtype=torch.float32)
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
            X, y, means, stds, _ = normalize_df_torch(self.df)
        elif self.config['normalize'] is False and self.model==PhModel:
            X, y = self.df.iloc[:, :-2].values, self.df.iloc[:, -2:].values
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            means = torch.tensor(self.df.iloc[:, :-2].mean(axis=0).values, dtype=torch.float32)
            stds = torch.tensor(self.df.iloc[:, :-2].std(axis=0).values, dtype=torch.float32)

            col_idx = self.df.columns.get_loc('Zero_eps_thickness')
            model_factory = lambda: PhModel(zlt_mu_stds=(means[col_idx], stds[col_idx]), current_target=233)
        else:
            X, y = self.df.iloc[:, :-2].values, self.df.iloc[:, -2:].values
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

        _, model = train_model_ens(X, y, model_factory, DNAME=self.output_dir, i=self.i, num_iter=self.config["num_iter"], plot=self.config["make_plots"])

        class Predictor(EnsembleModel):
            def __init__(self):
                super().__init__()
                self._num_outputs = 1
        
            def forward(self, X=torch.zeros((1, X.shape[1]), dtype=torch.float32)):
                return model(X.permute((1,0,2))).permute((2,0,1,3))
                
        return Predictor()

    def _get_acquisition_function(self, predictor):
        """
        Get the acquisition function based on the specified acquisition type.
        """
        if self.aquisition == "EI":
            return ExpectedImprovement(
                predictor,
                best_f=torch.tensor(self.df[self.quantity].max()),
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

        if self.config['normalize']:
            raise NotImplementedError("Normalization is not implemented for this step method.")
        
        predictor = self.get_predictor(new_data)

        AF = self._get_acquisition_function(predictor)

        if bounds is None:
            bounds = self.bounds

        col_i = self.df.columns.get_loc(self.quantity) - 4  # TODO: We should get rid of numerical column calls

        AF_q = lambda x:AF(x)[:,col_i]
        
        next_experiment, _ = optimize_acqf(
            acq_function=AF_q,
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=30,
            options={}, 
        )

        self.i += 1

        return AF(next_experiment), next_experiment.detach().cpu().numpy().flatten()

    def step_within_data(self, new_data, possible_data):

        predictor = self.get_predictor(new_data)

        if self.config['normalize']:
            X, y, _, _, _ = normalize_df_torch(possible_data)
        else:
            X, y = possible_data.iloc[:, :-2].values, possible_data.iloc[:, -2:].values
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

        AF = self._get_acquisition_function(predictor)
        
        scores = AF(X.unsqueeze(1))
        
        self.i += 1

        col_i = self.df.columns.get_loc(self.quantity) - 4  # TODO: We should get rid of numerical column calls
        
        return scores[:, col_i].max(), scores[:, col_i].argmax()
        
