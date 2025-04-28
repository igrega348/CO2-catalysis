from .models import PhModel, MLPModel, MultitaskGPModel, MultitaskGPhysModel
from .train import train_model_ens, train_GP_model, train_GP_Ph_model
from .loaders import load_data, normalize_df_torch
from .config import default_config
import pandas as pd
import torch

def get_ei(mu, sigma, fstar, minimize=False):
    if minimize: 
        mu = -mu
        fstar = -fstar
    diff = mu - fstar
    u = diff / sigma
    unit_normal = torch.distributions.Normal(0, 1)
    ei = ( diff * unit_normal.cdf(u) + 
          sigma * unit_normal.log_prob(u).exp()
    )
    ei[sigma <= 0.] = 0.
    return ei

class GDEOptimizer():
    def __init__(self, model_name="GP+Ph", aquisition="EI", quantity="FE (Eth)", maximize=True, output_dir="./out", config=default_config):
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
            self.aquisition = get_ei
        else:
            raise ValueError("Only EI is supported for now.")

        self.output_dir = output_dir

        self.config = config

        self.maximize = maximize

        self.quantity = quantity

        self.i = 0

        self.df = pd.DataFrame()
        
        
    def load(path):
        '''
        Load pretrained model.
        '''
        raise NotImplementedError

    def get_predictor(self, new_data):
        '''
        Train and return next experiment.
        '''
        self.df = pd.concat([self.df, new_data])

        if self.config['normalize']:
            X, y, means, stds, _ = normalize_df_torch(self.df)
        else:
            X, y = self.df.iloc[:, :-2].values, self.df.iloc[:, -2:].values
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

        _, predict = train_model_ens(X, y, self.model, DNAME=self.output_dir, i=self.i, num_iter=self.config["num_iter"], plot=self.config["make_plots"])

        return predict
    
    def step(self, new_data):

        predictor = self.get_predictor(new_data)

        self.i += 1

        raise NotImplementedError

    def step_within_data(self, new_data, possible_data):

        predictor = self.get_predictor(new_data)

        if self.config['normalize']:
            X, y, _, _, _ = normalize_df_torch(possible_data)
        else:
            X, y = possible_data.iloc[:, :-2].values, possible_data.iloc[:, -2:].values
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

        mu, std = predictor(X)

        col_i = self.df.columns.get_loc(self.quantity) - 4 # TODO: We should get rid of numerical column calls
        
        ei = get_ei(mu[:,col_i], std[:,col_i], torch.tensor(self.df[self.quantity].max()), minimize=False)
        
        self.i += 1
        
        return ei.max(), ei.argmax()
        
