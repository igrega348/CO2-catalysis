from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.func import functional_call
from torch import vmap
import copy
from torch.func import stack_module_state
import gpytorch
import matplotlib.pyplot as plt
from math import ceil
from rich.progress import track

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


    def scaled_model(x):
        y = vmap(fmodel, in_dims=(0, 0, None), randomness='different')(params, buffers, x)
        y_diff = y - y.mean(dim=0, keepdim=True)
        return y.mean(dim=0) + y_diff * variance_scaler.sqrt()

    return stats, scaled_model


# In[ ]:


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


# In[13]:


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
