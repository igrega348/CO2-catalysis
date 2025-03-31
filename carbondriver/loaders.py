from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json


# Load results from folder
def load_results_from_folder(folder: Path, plot: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_res = pd.DataFrame()
    for f in Path(folder).glob('losses*.json'):
        d = json.loads(f.read_text())
        _df = pd.DataFrame(d).set_index('epochs')
        if plot:
            plt.plot(_df.index, _df['train'], c='0.5', lw=0.5)
            plt.plot(_df.index, _df['val'], c='k', linestyle='--', lw=0.5)
        i = int(f.stem.split('_')[-1])
        df_res = df_res.merge(_df, how='outer', left_index=True, right_index=True, suffixes=('', f'_{i}'))  
    df_res

    # separate train and val
    cols = df_res.columns
    train_cols = [c for c in cols if 'train' in c]
    val_cols = [c for c in cols if 'val' in c]
    df_res_train = df_res[train_cols]
    df_res_val = df_res[val_cols]
    return df_res_train, df_res_val

def load_data(file: Optional[Path] = None):
    if file is None:
        file = Path('./Characterization_data.xlsx')
    df = pd.read_excel(file, skiprows=[1], index_col=0)
    df = df[['AgCu Ratio', 'Naf vol (ul)', 'Sust vol (ul)', 'Catalyst mass loading', 'FE (Eth)', 'FE (CO)']]
    df = df.sort_values(by=['AgCu Ratio', 'Naf vol (ul)'])
    df = df.dropna()
    df['FE (CO)'] = df['FE (CO)'] / 100
    df['FE (Eth)'] = df['FE (Eth)'] / 100

    dens_Ag = 10490 # kg/m^3
    dens_Cu = 8935 # kg/m^3
    dens_avg = (1-df['AgCu Ratio'])*dens_Cu + df['AgCu Ratio']*dens_Ag
    mass = df['Catalyst mass loading'] * 1e-6 # kg
    area = 1.85**2 # cm^2
    A = area * 1e-4 # m^2
    thickness = (mass / dens_avg)/A # m
    df.insert(3, column='Zero_eps_thickness', value=thickness)

    # reshuffle triplets
    df['triplet'] = np.arange(len(df)) // 3
#     gen = np.random.default_rng(1)
    gen = np.random.default_rng(2)
    order = gen.permutation(30)
    new_df = pd.DataFrame()
    for i in order:
        new_df = pd.concat([new_df, df[df['triplet'] == i]])
    new_df.reset_index(drop=True, inplace=True)
    df = new_df
    df = df.drop(columns=['triplet'])
    # normalize
    means = df.mean()
    stds = df.std(ddof=0)
    df_n = (df - means) / stds
    X = df_n.iloc[:, :-2].values # inputs normalized
    y = df.iloc[:, -2:].values # outputs (not normalized)
    print(X.shape, y.shape)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y, means, stds, df

def normalize_df_torch(df: pd.DataFrame, means: Optional[pd.DataFrame] = None, stds: Optional[pd.DataFrame] = None):
    if 'triplet' in df.columns:
        df = df.drop(columns=['triplet'])
    # normalize
    if means is None:
        assert stds is None
        means = df.mean()
        stds = df.std(ddof=0)
    df_n = (df - means) / stds
    X = df_n.iloc[:, :-2].values # inputs normalized
    y = df.iloc[:, -2:].values # outputs (not normalized)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y, means, stds, df