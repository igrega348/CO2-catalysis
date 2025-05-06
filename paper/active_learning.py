from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from typing import Literal

from carbondriver import GDEOptimizer
from carbondriver.loaders import load_data, normalize_df_torch
from carbondriver.models import *
from carbondriver.train import *

NUM_RUNS = 10
col_n = 'FE (Eth)'
col_i = 0

def init_data(data_path = None):
    X, y, means, stds, df = load_data(data_path)
    df['triplet'] = df.index//3
    df_triplet_means = df.groupby('triplet').mean()
    df_triplet_max = df.groupby('triplet').max()

    return df_triplet_means, df

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

def choose_base_inds_numpy(y: np.ndarray, num_choose: int, how: Literal['max','min'] = 'max', strategy: Literal['uniform','skewed'] = 'uniform'):
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
    p = p / p.sum()
    return np.random.choice(ind, size=num_choose, replace=False, p=p)


def mlp_ensemble(df, df_triplet_means, write_path):
    for d in range(NUM_RUNS):
        DNAME = Path(f'{write_path}/MLP_F/MLP_F{d}')
        try:
            DNAME.mkdir(exist_ok=False, parents=True)
        except FileExistsError:
            shutil.rmtree(DNAME)
            DNAME.mkdir(exist_ok=False, parents=True)
        df.to_csv(DNAME/'df.csv')
        chosen_triplets = choose_base_inds_numpy(df_triplet_means[col_n].values, 3)

        i = 0
        i_to_max = None
        expected_improvements = [None]*len(chosen_triplets)
        while len(chosen_triplets)<df_triplet_means.shape[0]:
            withheld_triplets = df_triplet_means[~df_triplet_means.index.isin(chosen_triplets)]

            chosen_df = df[df['triplet'].isin(chosen_triplets)]
            X, y, means, stds, _ = normalize_df_torch(chosen_df)

            if i_to_max is None and abs(y[:,col_i].max().item()-df[col_n].max())<1e-5:
                i_to_max = i
            print('\r', d, i, ' Max val:', y[:,col_i].max().item(), 'Target:', df[col_n].max(), 'i_to_max:', i_to_max, ' '*20, end='')

            try:
                stats, predict = train_model_ens(X, y, MLPModel, DNAME=DNAME, i=i, num_iter=400, plot=False)
            except torch._C._LinAlgError:
                print('')
                break

            X_test, y_test, _, _, test_df = normalize_df_torch(withheld_triplets, means, stds)
            y_train_pred, _ = predict(X)
            mu, std = predict(X_test)
            # need values averaged over triplets
            res = pd.DataFrame({'y': y[:, col_i], 'triplet': chosen_df['triplet']}).groupby('triplet').mean()
            ei = get_ei(mu[:,col_i], std[:,col_i], torch.tensor(res.max().iloc[0]), minimize=False)
            maxind = ei.argmax().item()
            expected_improvements.append(ei.max().item())
            maxtrip = test_df.index[maxind]
            chosen_triplets = np.append(chosen_triplets, maxtrip)

            i += 1
        
        print('')
        pd.DataFrame({'chosen_triplets': chosen_triplets, 'expected_improvements':expected_improvements}).to_csv(DNAME/'chosen_triplets.csv')


def ph_ensemble(df, df_triplet_means, write_path):
    ds = range(NUM_RUNS)
    for d in ds:
        DNAME = Path(f'{write_path}/Ph_F/Ph_F{d}')
        try:
            DNAME.mkdir(exist_ok=False, parents=True)
        except FileExistsError:
            shutil.rmtree(DNAME)
            DNAME.mkdir(exist_ok=False, parents=True)
        df.to_csv(DNAME/'df.csv')
        chosen_triplets = choose_base_inds_numpy(df_triplet_means[col_n].values, 3, strategy='uniform')

        i = 0
        i_to_max = None
        expected_improvements = [None]*len(chosen_triplets)
        while len(chosen_triplets)<df_triplet_means.shape[0]:
            withheld_triplets = df_triplet_means[~df_triplet_means.index.isin(chosen_triplets)]

            chosen_df = df[df['triplet'].isin(chosen_triplets)]
            X, y, means, stds, _ = normalize_df_torch(chosen_df)

            if i_to_max is None and abs(y[:,col_i].max().item()-df[col_n].max())<1e-5:
                i_to_max = i
            print('\r', d, i, 'Max val:', y[:,col_i].max().item(), 'Target:', df[col_n].max(), 'i_to_max:', i_to_max, ' '*20, end='')

            model = lambda: PhModel(zlt_mu_stds=(means['Zero_eps_thickness'], stds['Zero_eps_thickness']), current_target=233)
            try:
                stats, predict = train_model_ens(X, y, model, num_iter=101, DNAME=DNAME, i=i)
            except:
                print('')
                break

            X_test, _, _, _, test_df = normalize_df_torch(withheld_triplets, means, stds)
            mu, std = predict(X_test)
            res = pd.DataFrame({'y': y[:, col_i], 'triplet': chosen_df['triplet']}).groupby('triplet').mean()
            ei = get_ei(mu[:,col_i], std[:,col_i], torch.tensor(res.min().iloc[0]), minimize=False)
            maxind = ei.argmax().item()
            expected_improvements.append(ei.max().item())
            maxtrip = test_df.index[maxind]
            chosen_triplets = np.append(chosen_triplets, maxtrip)

            i += 1
        
        print('')
        pd.DataFrame({'chosen_triplets': chosen_triplets, 'expected_improvements':expected_improvements}).to_csv(DNAME/'chosen_triplets.csv')


def gp_active_learning(df, df_triplet_means, write_path):

    ds = range(NUM_RUNS)
    for d in ds:
        DNAME = Path(f'{write_path}/GP_F/GP_F{d}')
        try:
            DNAME.mkdir(exist_ok=False, parents=True)
        except FileExistsError:
            shutil.rmtree(DNAME)
            DNAME.mkdir(exist_ok=False, parents=True)
        df.to_csv(DNAME/'df.csv')
        chosen_triplets = choose_base_inds_numpy(df_triplet_means[col_n].values, 3, strategy='uniform')

        i = 0
        i_to_max = None
        expected_improvements = [None]*len(chosen_triplets)
        while len(chosen_triplets)<df_triplet_means.shape[0]:
            withheld_triplets = df_triplet_means[~df_triplet_means.index.isin(chosen_triplets)]

            chosen_df = df[df['triplet'].isin(chosen_triplets)]
            X, y, means, stds, _ = normalize_df_torch(chosen_df)

            if i_to_max is None and abs(y[:,col_i].max().item()-df[col_n].max())<1e-5:
                i_to_max = i
            print('\r', d, i, 'Max val:', y[:,col_i].max().item(), 'Target:', df[col_n].max(), 'i_to_max:', i_to_max, ' '*20, end='')

            try:
                stats, predict = train_GP_model(X, y, num_iter=101, DNAME=DNAME, i=i)
            except:
                print('')
                break

            X_test, _, _, _, test_df = normalize_df_torch(withheld_triplets, means, stds)
            mu, std = predict(X_test)
            res = pd.DataFrame({'y': y[:, col_i], 'triplet': chosen_df['triplet']}).groupby('triplet').mean()
            ei = get_ei(mu[:,col_i], std[:,col_i], torch.tensor(res.min().iloc[0]), minimize=False)
            maxind = ei.argmax().item()
            expected_improvements.append(ei.max().item())
            maxtrip = test_df.index[maxind]
            chosen_triplets = np.append(chosen_triplets, maxtrip)

            i += 1
        
        print('')
        pd.DataFrame({'chosen_triplets': chosen_triplets, 'expected_improvements':expected_improvements}).to_csv(DNAME/'chosen_triplets.csv')


def gp_ph_active(df, df_triplet_means, write_path):

    ds = range(NUM_RUNS)
    for d in ds:
        DNAME = Path(f'{write_path}/GP_Ph_F/GP_Ph_F{d}')
        DNAME.mkdir(exist_ok=True, parents=True)
        df.to_csv(DNAME/'df.csv')
        chosen_triplets = choose_base_inds_numpy(df_triplet_means[col_n].values, 3, strategy='uniform')

        i = 0
        i_to_max = None
        expected_improvements = [None]*len(chosen_triplets)
        while len(chosen_triplets)<df_triplet_means.shape[0]:
            withheld_triplets = df_triplet_means[~df_triplet_means.index.isin(chosen_triplets)]

            chosen_df = df[df['triplet'].isin(chosen_triplets)]
            X, y, means, stds, _ = normalize_df_torch(chosen_df)

            if i_to_max is None and abs(y[:,col_i].max().item()-df[col_n].max())<1e-5:
                i_to_max = i
                print('\r', d, i, 'Max val:', y[:,col_i].max().item(), 'Target:', df[col_n].max(), 'i_to_max:', i_to_max, ' '*20)
                break
            print('\r', d, i, 'Max val:', y[:,col_i].max().item(), 'Target:', df[col_n].max(), 'i_to_max:', i_to_max, ' '*20, end='')

            model = lambda: PhModel(zlt_mu_stds=(means['Zero_eps_thickness'], stds['Zero_eps_thickness']), current_target=233) 
            try:
                stats, predict = train_GP_Ph_model(X, y, model, num_iter=101, DNAME=DNAME, i=i, plot=False)
            except:
                print('')
                break

            X_test, _, _, _, test_df = normalize_df_torch(withheld_triplets, means, stds)
            mu, std = predict(X_test)
            res = pd.DataFrame({'y': y[:, col_i], 'triplet': chosen_df['triplet']}).groupby('triplet').mean()
            ei = get_ei(mu[:,col_i], std[:,col_i], torch.tensor(res.min()[0]), minimize=False)
            maxind = ei.argmax().item()
            expected_improvements.append(ei.max().item())
            maxtrip = test_df.index[maxind]
            chosen_triplets = np.append(chosen_triplets, maxtrip)

            i += 1
        print('')
        pd.DataFrame({'chosen_triplets': chosen_triplets, 'expected_improvements':expected_improvements}).to_csv(DNAME/'chosen_triplets.csv')