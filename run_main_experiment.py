from pathlib import Path
from typing import Tuple, Optional, Literal
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.func import functional_call
from torch import vmap
import copy
from torch.func import stack_module_state
import gpytorch
import matplotlib.pyplot as plt
from math import ceil
from rich.progress import track
import pickle as pkl
from analytical_models.gde_multi import System
from analytical_models import gde_multi
from analytical_models.loaders import load_data, normalize_df_torch
import sys

class PhModel(torch.nn.Module):
    def __init__(
        self, 
        zlt_mu_stds: Tuple,
        current_target: float = 200,
        dropout: float = 0.1, 
        ldim: int = 64,
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
        self.ph_model = System(
            diffusion_coefficients=gde_multi.diffusion_coefficients, 
            salting_out_exponents=gde_multi.salting_out_exponents, 
            electrode_reaction_kinetics=erc,
            electrode_reaction_potentials=gde_multi.electrode_reaction_potentials,
            chemical_reaction_rates=gde_multi.chemical_reaction_rates,
        )
        self.softmax = torch.nn.Softmax(dim=1)
        self.zlt_mu_stds = zlt_mu_stds
        self.current_target = current_target

    def forward(self, x):
        # columns of x: AgCu Ratio, Naf vol (ul), Sust vol (ul), Zero_eps_thickness, Catalyst mass loading
        latents = self.net(x)
        r = 40e-9 * torch.exp(latents[..., [0]])
        eps = torch.sigmoid(latents[..., [1]])
        zlt = (x[..., 3]*self.zlt_mu_stds[1] + self.zlt_mu_stds[0]).view(-1,1)
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
    def __init__(self, dropout: float = 0.1, ldim: int = 64):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(5, ldim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ldim, ldim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ldim, ldim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ldim, 2),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x)


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )

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


def choose_base_inds_numpy(y: np.ndarray, num_choose: int, how: Literal['max','min'] = 'max', strategy: Literal['uniform','skewed'] = 'uniform', seed: Optional[int] = None):
    ind = np.argsort(y)
    N = y.shape[0]
    i = np.arange(N)
    if strategy=='skewed':
        if how=='max':
            p = (i - i.max())**2
        elif how=='min':
            p = i**2
    elif strategy=='uniform':
        p = np.ones_like(i)
    else: 
        raise ValueError
    p = p/p.sum()
    rng = np.random.default_rng(seed)
    return rng.choice(ind, size=num_choose, replace=False, p=p)


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


def get_cov(batch):
    batch = batch.reshape(*batch.shape[:-2], -1)
    return torch.cov(batch.transpose(-1,-2)) + 1e-6*torch.eye(batch.shape[1])

def get_nll(predictions, targets):
    samples = predictions.sample(sample_shape=torch.Size([1000]))
    return get_nll_samples(samples, targets)

def get_nll_samples(samples, targets, covariance_scaler: Optional[torch.Tensor] = None):
    mean = samples.mean(dim=0)
    covariance = get_cov(samples)
    if covariance_scaler is not None:
        inds = torch.arange(covariance.shape[0])
        covariance[inds, inds] *= covariance_scaler
    gmodel = gpytorch.distributions.MultitaskMultivariateNormal(mean=mean, covariance_matrix=covariance)
    return -(gmodel.log_prob(targets) / gmodel.event_shape.numel())


def train_model_ens(X_train, y_train, model_constructor, num_iter: int, DNAME, i, progress=False, plot=False):
    
    DNAME = Path(DNAME)
    DNAME.mkdir(exist_ok=True)
    if plot:
        fig, ax = plt.subplots(ncols=3, figsize=(10,3))
        ax[1].axline((0,0), slope=1, c='k', ls='--')
        ax[2].axline((0,0), slope=1, c='k', ls='--')

    # set up model and optimizer
    num_models = 50
    model = [model_constructor() for _ in range(num_models)]
    params, buffers = stack_module_state(model)
    base_model = copy.deepcopy(model[0])
    base_model = base_model.to('meta')
    def fmodel(params, buffers, x):
        return functional_call(base_model, (params, buffers), (x,))

    optimizer = torch.optim.Adam(params.values(), lr=0.001)
    variance_scaler = torch.tensor(1.0, requires_grad=True)
    dummy_variance_scaler = torch.tensor(1.0, requires_grad=True)
    variance_optimizer = torch.optim.Adam([variance_scaler], lr=1)

    # batch the train data
    num_data_per_model = ceil(X_train.shape[0]*0.5)
    #inds = torch.stack([torch.arange(num_data_per_model) for _ in range(num_models)], dim=0)
    
    inds = torch.stack([torch.randperm(X_train.shape[0])[:num_data_per_model] for _ in range(num_models)], dim=0)
    X_train_b = torch.stack([X_train[inds[i], :] for i in range(num_models)], dim=0)
    y_train_b = torch.stack([y_train[inds[i], :] for i in range(num_models)], dim=0)

    base_model.train()
    
    stats = {x:np.nan for x in ['loss','val_loss']} | {'step':np.arange(num_iter)}
    stats = pd.DataFrame(stats).set_index('step')

    iterator = range(num_iter)
    if progress:
        iterator = track(iterator)
    for it in iterator:
        optimizer.zero_grad()

        output = vmap(fmodel, in_dims=(0, 0, 0), randomness='different')(params, buffers, X_train_b)
        loss = torch.mean((output - y_train_b)**2)
        loss.backward()
        optimizer.step()
        stats.loc[it, 'loss'] = loss.item()

        # test            
        if it%ceil(num_iter/100)==0:

            base_model.eval()

            with torch.no_grad():
                fe_train = vmap(fmodel, in_dims=(0, 0, None), randomness='different')(params, buffers, X_train)    
                mean_train = fe_train.mean(dim=0)
                std_train = fe_train.std(dim=0)*variance_scaler.sqrt()
            variance_optimizer.zero_grad()
            nll_train = get_nll_samples(fe_train, y_train, covariance_scaler=variance_scaler)

            nll_train.backward()
            variance_optimizer.step()
            stats.loc[it, 'nll'] = nll_train.item()
            stats.loc[it, 'variance_scaler'] = variance_scaler.item()

            base_model.train()

        f = lambda x: round(x.item() if isinstance(x, torch.Tensor) else x, 5)
    if plot:
        stats['nll'].dropna().plot(y='nll', c='C0', ls='--', lw=0.7, alpha=0.5, ax=ax[0])


    if plot:
        # plot parity plots with confidence intervals
        ax[1].errorbar(y_train[:, 0].numpy(), mean_train[:,0].numpy(), yerr=std_train[:, 0].numpy(), fmt='o', alpha=0.5, mfc=f'C0', mec='white')
        ax[2].errorbar(y_train[:, 1].numpy(), mean_train[:,1].numpy(), yerr=std_train[:, 1].numpy(), fmt='o', alpha=0.5, mfc=f'C0', mec='white')
        fig.tight_layout()
        plt.show()
    # save average losses to file
    stats.to_csv(DNAME/f'stats_{i:02d}.csv')
    
    def predict(X_test):
        fe_test = vmap(fmodel, in_dims=(0, 0, None), randomness='different')(params, buffers, X_test)
        mean_test = fe_test.mean(dim=0)
        std_test = fe_test.std(dim=0)*variance_scaler.sqrt()
        return mean_test, std_test

    return stats, predict


def train_GP_model(X_train, y_train, num_iter: int, DNAME, i, progress=False, plot=False):
    DNAME = Path(DNAME)
    DNAME.mkdir(exist_ok=True, parents=True)
    if plot:
        fig, ax = plt.subplots(ncols=3, figsize=(10,3))
        ax[1].axline((0,0), slope=1, c='k', ls='--')
        ax[2].axline((0,0), slope=1, c='k', ls='--')

    # set up model and optimizer
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultitaskGPModel(X_train, y_train, likelihood)

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()

    stats = {x:np.nan for x in ['loss','val_loss']} | {'step':np.arange(num_iter)}
    stats = pd.DataFrame(stats).set_index('step')

    iterator = range(num_iter)
    if progress:
        iterator = track(iterator)
    for it in iterator:
        model.train()
        likelihood.train()
            
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        stats.loc[it, 'loss'] = loss.item()

        # test            
        if it%5==0:

            model.eval()
            likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                predictions = likelihood(model(X_train+1e-6*(torch.rand(X_train.shape)-0.5)))
                mean_train = predictions.mean
                std_train = predictions.stddev
                stats.loc[it, 'nll'] = get_nll(predictions, y_train).item()



    if plot:
        # loss curves
        stats['loss'].dropna().plot(y='loss', c='C0', ls='--', lw=0.7, alpha=0.5, ax=ax[0])

        # plot parity plots with confidence intervals
        ax[1].errorbar(y_train[:,0].numpy(), mean_train[:,0].numpy(), yerr=std_train[:,0].numpy(), fmt='o', alpha=0.5, mfc=f'C{i}', mec='white')
        ax[2].errorbar(y_train[:,1].numpy(), mean_train[:,1].numpy(), yerr=std_train[:,1].numpy(), fmt='o', alpha=0.5, mfc=f'C{i}', mec='white')
        fig.tight_layout()
        plt.show()
        
    # save average losses to file
    stats.to_csv(DNAME/f'stats_{i:02d}.csv')

    def predict(X_test):
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = model(X_test)
            predictions = likelihood(output)
            mean_test = predictions.mean
            std_test = predictions.stddev
        return mean_test, std_test

    return stats, predict


def train_Ph_model(X_train, y_train, model_constructor, num_iter):
    model = model_constructor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    stats = {x:np.nan for x in ['loss','val_loss']} | {'step':np.arange(num_iter)}
    stats = pd.DataFrame(stats).set_index('step')

    model.train()
    for it in range(num_iter):
        optimizer.zero_grad()
        output = model(X_train)
        loss = torch.nn.functional.mse_loss(output, y_train)
        loss.backward()
        optimizer.step()
        stats.loc[it, 'loss'] = loss.item()

    return stats, model

def train_GP(X_train, y_train, mean_model, num_iter):
    # set up model and optimizer
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultitaskGPhysModel(X_train, y_train, likelihood, model=mean_model, freeze_model=True)

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    stats = {x:np.nan for x in ['loss','val_loss']} | {'step':np.arange(num_iter)}
    stats = pd.DataFrame(stats).set_index('step')


    for it in range(num_iter):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        stats.loc[it, 'loss'] = loss.item()

    return stats, model
    
def train_GP_Ph_model(X_train, y_train, model_constructor, num_iter: int, DNAME, i, progress=False, plot=False, ph_frac: float = 0.5):
    DNAME = Path(DNAME)
    DNAME.mkdir(exist_ok=True, parents=True)
    

    # split into subset
    N = X_train.shape[0]
    inds = torch.randperm(N)
    inds_ph = inds[:int(ph_frac*N)]
    inds_gp = inds[int(ph_frac*N):]

    X_ph, y_ph = X_train[inds_ph], y_train[inds_ph]
    X_gp, y_gp = X_train[inds_gp], y_train[inds_gp]


    stats, model = train_Ph_model(X_ph, y_ph, model_constructor, num_iter)
    stats[['loss','val_loss']] = np.log(stats[['loss','val_loss']])

    stats2, model = train_GP(X_gp, y_gp, model, num_iter)
    stats2.index += stats.index.max()
    stats = pd.concat([stats, stats2], axis=0)
    stats.to_csv(DNAME/f'stats_{i:02d}.csv')
    
    def predict(X_test):
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = model(X_test)
            predictions = model.likelihood(output)
            mean_test = predictions.mean
            std_test = predictions.stddev
        return mean_test, std_test
        
    return stats, predict


def process_runs_mean(dname):
    all_df = []
    for p in dname.iterdir():
        if not p.is_dir(): continue
        df = pd.read_csv(p/'df.csv', index_col=0)
        chosen_triplets = pd.read_csv(p/'chosen_triplets.csv', index_col=0)
        df_triplets_mean = df.groupby('triplet').mean()
        chosen_triplets.loc[:, 'cummax FE'] = df_triplets_mean.loc[chosen_triplets['chosen_triplets'], 'FE (Eth)'].cummax().values

        i0 = 2
        chosen_triplets['step'] = chosen_triplets.index - i0
        chosen_triplets['dname'] = p.stem
        all_df.append(chosen_triplets)
    all_df = pd.concat(all_df, axis=0)
    return all_df


if __name__ == '__main__':
    torch.manual_seed(0)
    # Load data
    X, y, means, stds, df = load_data()
    #df['triplet'] = df.index//3

    df_triplet_means = df.groupby('triplet').mean()
    df_triplet_max = df.groupby('triplet').max()

    NUM_RUNS = 100
    col_n = 'FE (Eth)'
    col_i = 0

    REF = sys.argv[1]  # Set to "min" or "max"

    print(f"Running experiment with reference set to {REF}.")

    # Dictionary to store exceptions
    exceptions_dict = {}

    # ========== MLP ensemble ==========
    print("Running MLP ensemble...")
    for d in range(NUM_RUNS):
        exceptions_dict[('MLP', d)] = []
        DNAME = Path(f'./MLP_{REF}/MLP_{d}')
        DNAME.mkdir(exist_ok=True, parents=True)
        df.to_csv(DNAME/'df.csv')
        chosen_triplets = choose_base_inds_numpy(df_triplet_means[col_n].values, 3, seed=d)
        bests = df_triplet_means.loc[chosen_triplets][col_n].cummax().tolist()
        print("Starting triplets:", chosen_triplets)
        print("Starting bests:", bests)

        i = 0
        i_to_max = None
        expected_improvements = [None]*len(chosen_triplets)
        while df_triplet_means[col_n].idxmax() not in chosen_triplets:
            withheld_triplets = df_triplet_means[~df_triplet_means.index.isin(chosen_triplets)]

            chosen_df = df[df['triplet'].isin(chosen_triplets)]
            X, y, means, stds, _ = normalize_df_torch(chosen_df)

            if i_to_max is None and abs(y[:,col_i].max().item()-df[col_n].max())<1e-5:
                i_to_max = i
                print('\r', d, i, ' Max val:', y[:,col_i].max().item(), 'Target:', df[col_n].max(), 'i_to_max:', i_to_max, ' '*20, end='')

            try:
                stats, predict = train_model_ens(X, y, MLPModel, DNAME=DNAME, i=i, num_iter=101, plot=False)
            except Exception as e:
                print(f"Exception during MLP training at run {d}, step {i}: {type(e).__name__}: {str(e)}")
                exceptions_dict[('MLP', d)].append({'step': i, 'exception': type(e).__name__, 'message': str(e)})
                candidate_triplets = withheld_triplets.index.values
                if len(candidate_triplets) == 0:
                    break
                fallback = torch.randint(max(len(candidate_triplets), 1), (1,), dtype=torch.int64).item()
                chosen_triplets = np.append(chosen_triplets, candidate_triplets[fallback])
                expected_improvements.append(np.nan)
                i += 1
                continue

            X_test, y_test, _, _, test_df = normalize_df_torch(withheld_triplets, means, stds)
            y_train_pred, _ = predict(X)
            mu, std = predict(X_test)
            # need values averaged over triplets
            # res = pd.DataFrame({'y': y[:, col_i], 'triplet': chosen_df['triplet']}).groupby('triplet').mean()
            res = pd.DataFrame({'y': y[:, col_i], 'triplet': chosen_df['triplet']})
            if REF == "max":
                reference = torch.tensor(res["y"].max())
            else:
                reference = torch.tensor(res["y"].min())
            print("Reference FE:", reference.item())
            ei = get_ei(mu[:,col_i], std[:,col_i], reference, minimize=False)
            print('Target Scores:', ei)
            
            maxind = ei.argmax().item()
            expected_improvements.append(ei.max().item())
            maxtrip = test_df.index[maxind]
            chosen_triplets = np.append(chosen_triplets, maxtrip)
            print("Chosen triplets so far: ", chosen_triplets)
            bests.append(df_triplet_means.loc[chosen_triplets][col_n].max().item())
            print("Best FE so far: ", bests)

            i += 1
        
        print(f"Found max after {i} iterations.")
        pd.DataFrame({'chosen_triplets': chosen_triplets, 'expected_improvements':expected_improvements}).to_csv(DNAME/'chosen_triplets.csv')


    # ========== Ph ensemble ==========
    print("Running Ph ensemble...")
    ds = range(NUM_RUNS)
    for d in ds:
        exceptions_dict[('Ph', d)] = []
        DNAME = Path(f'./Ph_{REF}/Ph_{d}')
        DNAME.mkdir(exist_ok=True, parents=True)
        df.to_csv(DNAME/'df.csv')
        chosen_triplets = choose_base_inds_numpy(df_triplet_means[col_n].values, 3, strategy='uniform', seed=d)
        bests = df_triplet_means.loc[chosen_triplets][col_n].cummax().tolist()
        print("Starting triplets:", chosen_triplets)
        print("Starting bests:", bests)
        i = 0
        i_to_max = None
        expected_improvements = [None]*len(chosen_triplets)
        while df_triplet_means[col_n].idxmax() not in chosen_triplets:

            print(f"Run: {d}, Iteration: {i}")

            withheld_triplets = df_triplet_means[~df_triplet_means.index.isin(chosen_triplets)]

            chosen_df = df[df['triplet'].isin(chosen_triplets)]
            X, y, means, stds, _ = normalize_df_torch(chosen_df)
            
            model = lambda: PhModel(zlt_mu_stds=(means['Zero_eps_thickness'], stds['Zero_eps_thickness']), current_target=233, dropout=0)
            try:
                stats, predict = train_model_ens(X, y, model, num_iter=101, DNAME=DNAME, i=i)
            except Exception as e:
                print(f"Exception during Ph training at run {d}, step {i}: {type(e).__name__}: {str(e)}")
                exceptions_dict[('Ph', d)].append({'step': i, 'exception': type(e).__name__, 'message': str(e)})
                candidate_triplets = withheld_triplets.index.values
                if len(candidate_triplets) == 0:
                    break
                fallback = torch.randint(max(len(candidate_triplets), 1), (1,), dtype=torch.int64).item()
                chosen_triplets = np.append(chosen_triplets, candidate_triplets[fallback])
                expected_improvements.append(np.nan)
                i += 1
                continue

            X_test, _, _, _, test_df = normalize_df_torch(withheld_triplets, means, stds)
            mu, std = predict(X_test)
            # res = pd.DataFrame({'y': y[:, col_i], 'triplet': chosen_df['triplet']}).groupby('triplet').mean()
            res = pd.DataFrame({'y': y[:, col_i], 'triplet': chosen_df['triplet']})
            if REF == "max":
                reference = torch.tensor(res["y"].max())
            else:
                reference = torch.tensor(res["y"].min())
            print("Reference FE:", reference.item())
            try:
                ei = get_ei(mu[:,col_i], std[:,col_i], reference, minimize=False)
            except Exception as e:
                print(f"Exception during ei at run {d}, step {i}: {type(e).__name__}: {str(e)}")
                exceptions_dict[('Ph', d)].append({'step': i, 'exception': type(e).__name__, 'message': "EI ERROR" + str(e)})
                candidate_triplets = withheld_triplets.index.values
                if len(candidate_triplets) == 0:
                    break
                fallback = torch.randint(max(len(candidate_triplets), 1), (1,), dtype=torch.int64).item()
                chosen_triplets = np.append(chosen_triplets, candidate_triplets[fallback])
                expected_improvements.append(np.nan)
                i += 1
                continue
            
            print('Target Scores:', ei)

            maxind = ei.argmax().item()
            expected_improvements.append(ei.max().item())
            maxtrip = test_df.index[maxind]
            chosen_triplets = np.append(chosen_triplets, maxtrip)
            print("Chosen triplets so far: ", chosen_triplets)
            bests.append(df_triplet_means.loc[chosen_triplets][col_n].max().item())
            print("Best FE so far: ", bests)

            # # EI Visualizer
            # plt.figure()
            # plt.scatter(withheld_triplets.loc[:,'FE (Eth)'], mu[:,col_i].detach().numpy())
            # plt.scatter(withheld_triplets.loc[:,'FE (Eth)'], reference + ei.detach().numpy(), marker='^')
            # plt.errorbar(withheld_triplets.loc[:,'FE (Eth)'], mu[:,col_i].detach().numpy(), yerr=std[:,col_i].detach().numpy(), fmt="", ls="")
            # plt.plot(withheld_triplets.loc[maxtrip, 'FE (Eth)'], mu[maxind,col_i].detach().numpy(), "x")
            # # Parity line
            # plt.axhline(reference)
            # plt.plot([0,0.3], [0,0.3], 'k--')
            # ax = plt.gca()
            # ax.set_aspect('equal')
            # plt.show()

            i += 1           
        
        print(f"Found max after {i} iterations.")
        pd.DataFrame({'chosen_triplets': chosen_triplets, 'expected_improvements':expected_improvements}).to_csv(DNAME/'chosen_triplets.csv')
        
    # ========== GP ==========
    print("Running GP...")
    ds = range(NUM_RUNS)
    for d in ds:
        exceptions_dict[('GP', d)] = []
        DNAME = Path(f'./GP_{REF}/GP_{d}')
        DNAME.mkdir(exist_ok=True, parents=True)
        df.to_csv(DNAME/'df.csv')
        chosen_triplets = choose_base_inds_numpy(df_triplet_means[col_n].values, 3, strategy='uniform', seed=d)
        bests = df_triplet_means.loc[chosen_triplets][col_n].cummax().tolist()
        print("Starting triplets:", chosen_triplets)
        print("Starting bests:", bests)

        i = 0
        i_to_max = None
        expected_improvements = [None]*len(chosen_triplets)
        while df_triplet_means[col_n].idxmax() not in chosen_triplets:
            withheld_triplets = df_triplet_means[~df_triplet_means.index.isin(chosen_triplets)]

            chosen_df = df[df['triplet'].isin(chosen_triplets)]
            X, y, means, stds, _ = normalize_df_torch(chosen_df)

            if i_to_max is None and abs(y[:,col_i].max().item()-df[col_n].max())<1e-5:
                i_to_max = i
                print('\r', d, i, 'Max val:', y[:,col_i].max().item(), 'Target:', df[col_n].max(), 'i_to_max:', i_to_max, ' '*20, end='')

            try:
                stats, predict = train_GP_model(X, y, num_iter=101, DNAME=DNAME, i=i)
            except Exception as e:
                print(f"Exception during GP training at run {d}, step {i}: {type(e).__name__}: {str(e)}")
                exceptions_dict[('GP', d)].append({'step': i, 'exception': type(e).__name__, 'message': str(e)})
                candidate_triplets = withheld_triplets.index.values
                if len(candidate_triplets) == 0:
                    break
                fallback = torch.randint(max(len(candidate_triplets), 1), (1,), dtype=torch.int64).item()
                chosen_triplets = np.append(chosen_triplets, candidate_triplets[fallback])
                expected_improvements.append(np.nan)
                i += 1
                continue

            X_test, _, _, _, test_df = normalize_df_torch(withheld_triplets, means, stds)
            mu, std = predict(X_test)
            # res = pd.DataFrame({'y': y[:, col_i], 'triplet': chosen_df['triplet']}).groupby('triplet').mean()
            res = pd.DataFrame({'y': y[:, col_i], 'triplet': chosen_df['triplet']})
            if REF == "max":
                reference = torch.tensor(res["y"].max())
            else:
                reference = torch.tensor(res["y"].min())
            print("Reference FE:", reference.item())
            ei = get_ei(mu[:,col_i], std[:,col_i], reference, minimize=False)
            print('Target Scores:', ei)
            
            maxind = ei.argmax().item()
            expected_improvements.append(ei.max().item())
            maxtrip = test_df.index[maxind]
            chosen_triplets = np.append(chosen_triplets, maxtrip)
            print("Chosen triplets so far: ", chosen_triplets)
            bests.append(df_triplet_means.loc[chosen_triplets][col_n].max().item())
            print("Best FE so far: ", bests)

            i += 1
                
        
        print(f"Found max after {i} iterations.")
        pd.DataFrame({'chosen_triplets': chosen_triplets, 'expected_improvements':expected_improvements}).to_csv(DNAME/'chosen_triplets.csv')

    # ========== GP+Ph ==========
    print("Running GP+Ph...")
    ds = range(NUM_RUNS)
    for d in ds:
        exceptions_dict[('GP_Ph', d)] = []
        DNAME = Path(f'./GP_Ph_{REF}/GP_Ph_{d}')
        DNAME.mkdir(exist_ok=True, parents=True)
        df.to_csv(DNAME/'df.csv')
        chosen_triplets = choose_base_inds_numpy(df_triplet_means[col_n].values, 3, strategy='uniform', seed=d)
        bests = df_triplet_means.loc[chosen_triplets][col_n].cummax().tolist()
        print("Starting triplets:", chosen_triplets)
        print("Starting bests:", bests)

        i = 0
        i_to_max = None
        expected_improvements = [None]*len(chosen_triplets)
        while df_triplet_means[col_n].idxmax() not in chosen_triplets:
            withheld_triplets = df_triplet_means[~df_triplet_means.index.isin(chosen_triplets)]

            chosen_df = df[df['triplet'].isin(chosen_triplets)]
            X, y, means, stds, _ = normalize_df_torch(chosen_df)

            if i_to_max is None and abs(y[:,col_i].max().item()-df[col_n].max())<1e-5:
                i_to_max = i
                print('\r', d, i, 'Max val:', y[:,col_i].max().item(), 'Target:', df[col_n].max(), 'i_to_max:', i_to_max, ' '*20)

            model = lambda: PhModel(zlt_mu_stds=(means['Zero_eps_thickness'], stds['Zero_eps_thickness']), current_target=233) 
            try:
                stats, predict = train_GP_Ph_model(X, y, model, num_iter=101, DNAME=DNAME, i=i, plot=False)
            except Exception as e:
                print(f"Exception during GP+Ph training at run {d}, step {i}: {type(e).__name__}: {str(e)}")
                exceptions_dict[('GP_Ph', d)].append({'step': i, 'exception': type(e).__name__, 'message': str(e)})
                candidate_triplets = withheld_triplets.index.values
                if len(candidate_triplets) == 0:
                    break
                fallback = torch.randint(max(len(candidate_triplets), 1), (1,), dtype=torch.int64).item()
                chosen_triplets = np.append(chosen_triplets, candidate_triplets[fallback])
                expected_improvements.append(np.nan)
                i += 1
                continue

            X_test, _, _, _, test_df = normalize_df_torch(withheld_triplets, means, stds)
            mu, std = predict(X_test)
            # res = pd.DataFrame({'y': y[:, col_i], 'triplet': chosen_df['triplet']}).groupby('triplet').mean()
            res = pd.DataFrame({'y': y[:, col_i], 'triplet': chosen_df['triplet']})
            if REF == "max":
                reference = torch.tensor(res["y"].max())
            else:
                reference = torch.tensor(res["y"].min())
            print("Reference FE:", reference.item())
            ei = get_ei(mu[:,col_i], std[:,col_i], reference, minimize=False)
            print('Target Scores:', ei)
            
            maxind = ei.argmax().item()
            expected_improvements.append(ei.max().item())
            maxtrip = test_df.index[maxind]
            chosen_triplets = np.append(chosen_triplets, maxtrip)
            print("Chosen triplets so far: ", chosen_triplets)
            bests.append(df_triplet_means.loc[chosen_triplets][col_n].max().item())
            print("Best FE so far: ", bests)

            i += 1
                
        print(f"Found max after {i} iterations.")
        pd.DataFrame({'chosen_triplets': chosen_triplets, 'expected_improvements':expected_improvements}).to_csv(DNAME/'chosen_triplets.csv')

    # ========== Print exceptions summary ==========
    print("\n" + "="*80)
    print("EXCEPTION SUMMARY")
    print("="*80)
    for (model_name, run_num), exceptions in exceptions_dict.items():
        if exceptions:
            print(f"\n{model_name} - Run {run_num}:")
            for exc in exceptions:
                print(f"  Step {exc['step']}: {exc['exception']} - {exc['message']}")
        else:
            if run_num == 0:  # Print for first run of each model to show no exceptions
                print(f"\n{model_name} - Run {run_num}: No exceptions")

    # ========== Post-processing ==========
    print("\nPost-processing...")
    for dname in [f'MLP_{REF}', f'Ph_{REF}', f'GP_{REF}', f'GP_Ph_{REF}']:
        steps_to_finish = []
        for p in Path(dname).iterdir():
            if not p.is_dir():
                continue
            _df = pd.read_csv(p/'df.csv', index_col=0)
            chosen_triplets = pd.read_csv(p/'chosen_triplets.csv', index_col=0)
            df_triplets_mean = _df.groupby('triplet').mean()
            chosen_triplets.loc[:, 'cummax FE'] = df_triplets_mean.loc[chosen_triplets['chosen_triplets'], 'FE (Eth)'].cummax().values

            i0 = 2
            chosen_triplets['step'] = chosen_triplets.index - i0
            chosen_triplets['dname'] = p.stem
            steps_to_finish.append(chosen_triplets.loc[chosen_triplets['cummax FE']>0.245, 'step'].min())

        if steps_to_finish:
            sf = np.array(steps_to_finish)
            sf[sf < 0] = 0
            mean_steps = np.mean(sf[~np.isnan(sf)])
            std_steps = np.std(sf[~np.isnan(sf)])
            accel_factor = 13 / mean_steps if mean_steps > 0 else np.inf
            print(dname, mean_steps, std_steps, 'Accel F', accel_factor)
        else:
            print(dname, 'no runs found')

    # ========== Plotting ==========
    print("Creating plots...")
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10,8), sharex=True, sharey=True)
    all_df = []
    for i, dname in enumerate([f'MLP_{REF}', f'Ph_{REF}', f'GP_{REF}', f'GP_Ph_{REF}']):
        _df = process_runs_mean(Path(dname))
        _df = _df[_df['step']>=0]
        sns.lineplot(data=_df, x='step', y='cummax FE', hue='dname', legend=False, ax=ax[i//2, i%2])
        ax[i//2, i%2].set_title(dname)
        ax[i//2, i%2].set_ylabel('min(FE)')
        all_df.append(_df)
    all_df = pd.concat(all_df, axis=0)
    fig.tight_layout()
    plt.savefig('./active-learning-subplots.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()

    all_df['Model'] = all_df['dname'].map(lambda x: '_'.join(x.split('_')[:-1]))
    plt.figure(figsize=(3.75,3))
    sns.lineplot(data=all_df, x='step', y='cummax FE', hue='Model', marker='o', ms=5)
    plt.ylabel('max FE (Eth)')
    plt.xlabel('Step')
    plt.savefig('./active-learning.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()

    print("Done!")
