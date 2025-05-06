from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

def compare_runs(run_dirs, write_path="."):

    for dname in run_dirs:
        steps_to_finish = []
        for p in Path(dname).iterdir():
            if not p.is_dir(): continue
            _df = pd.read_csv(p/'df.csv', index_col=0)
            chosen_triplets = pd.read_csv(p/'chosen_triplets.csv', index_col=0)
            df_triplets_mean = _df.groupby('triplet').mean()
            chosen_triplets.loc[:, 'cummax FE'] = df_triplets_mean.loc[chosen_triplets['chosen_triplets'], 'FE (Eth)'].cummax().values

            i0 = 2
            chosen_triplets['step'] = chosen_triplets.index - i0
            chosen_triplets['dname'] = p.stem
            steps_to_finish.append(chosen_triplets.loc[chosen_triplets['cummax FE']>0.245, 'step'].min())
        sf = np.array(steps_to_finish)
        sf[sf<0] = 0
        mean = np.mean(sf[~np.isnan(sf)])
        std = np.std(sf[~np.isnan(sf)])
        af = 13 / mean
        print(dname, mean, std, af)

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10,8), sharex=True, sharey=True)
    all_df = []
    for i, d in enumerate(run_dirs):
        dname = d.split("/")[-1]
        _df = process_runs_mean(Path(d))
        _df = _df[_df['step']>=0]
        sns.lineplot(data=_df, x='step', y='cummax FE', hue='dname', legend=False, ax=ax[i//2, i%2])
        ax[i//2, i%2].set_title(dname)
        ax[i//2, i%2].set_ylabel('min(FE)')
        all_df.append(_df)
    all_df = pd.concat(all_df, axis=0)
    fig.tight_layout()
    plt.savefig(f'{write_path}/compare.pdf')

    all_df['Model'] = all_df['dname'].map(lambda x: '_'.join(x.split('_')[:-1]))
    plt.figure(figsize=(3.75,3))
    sns.lineplot(data=all_df, x='step', y='cummax FE', hue='Model', marker='o', ms=5)
    plt.ylabel('max FE (Eth)')
    plt.xlabel('Step')
    plt.savefig(f'{write_path}/active-learning.pdf', bbox_inches='tight', pad_inches=0.1)

