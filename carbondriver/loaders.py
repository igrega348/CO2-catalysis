from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json


# Load results from folder
def load_results_from_folder(
    folder: Path, plot: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training losses from JSON files in folder.

    :param folder: path to folder containing losses*.json files
    :param plot: whether to plot training curves
    :returns: tuple of (train_df, val_df) DataFrames
    """
    df_res = pd.DataFrame()
    for f in Path(folder).glob("losses*.json"):
        d = json.loads(f.read_text())
        _df = pd.DataFrame(d).set_index("epochs")
        if plot:
            plt.plot(_df.index, _df["train"], c="0.5", lw=0.5)
            plt.plot(_df.index, _df["val"], c="k", linestyle="--", lw=0.5)
        i = int(f.stem.split("_")[-1])
        df_res = df_res.merge(
            _df, how="outer", left_index=True, right_index=True, suffixes=("", f"_{i}")
        )
    df_res

    # separate train and val
    cols = df_res.columns
    train_cols = [c for c in cols if "train" in c]
    val_cols = [c for c in cols if "val" in c]
    df_res_train = df_res[train_cols]
    df_res_val = df_res[val_cols]
    return df_res_train, df_res_val


def load_gas_data(file: Optional[Path] = None) -> pd.DataFrame:
    """Load experimental data from Excel and compute electrode thickness.

    :param file: path to Excel file (default: ./Characterization_data.xlsx)
    :returns: DataFrame with features and experimental Faradaic efficiencies
    """
    if file is None:
        file = Path("./Characterization_data.xlsx")
    df = pd.read_excel(file, skiprows=[1], index_col=0)
    df = df[
        [
            "AgCu Ratio",
            "Naf vol (ul)",
            "Sust vol (ul)",
            "Catalyst mass loading",
            "FE (Eth)",
            "FE (CO)",
        ]
    ]
    df = df.sort_values(by=["AgCu Ratio", "Naf vol (ul)"])
    df = df.dropna()
    df["FE (CO)"] = df["FE (CO)"] / 100
    df["FE (Eth)"] = df["FE (Eth)"] / 100

    dens_Ag = 10490  # kg/m^3
    dens_Cu = 8935  # kg/m^3
    dens_avg = (1 - df["AgCu Ratio"]) * dens_Cu + df["AgCu Ratio"] * dens_Ag
    mass = df["Catalyst mass loading"] * 1e-6  # kg
    area = 1.85**2  # cm^2
    A = area * 1e-4  # m^2
    thickness = (mass / dens_avg) / A  # m
    df.insert(3, column="Zero_eps_thickness", value=thickness)

    # reshuffle triplets
    df["triplet"] = np.arange(len(df)) // 3
    #     gen = np.random.default_rng(1)
    gen = np.random.default_rng(2)
    order = gen.permutation(30)
    new_df = pd.DataFrame()
    for i in order:
        new_df = pd.concat([new_df, df[df["triplet"] == i]])
    new_df.reset_index(drop=True, inplace=True)
    # df = new_df
    # normalize
    return df


def feature_stats(
    df: pd.DataFrame,
    all_means: Optional[pd.Series] = None,
    all_stds: Optional[pd.Series] = None,
    as_torch: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[pd.Series, pd.Series]:
    """
    Return per-feature means/stds (excluding the last two target columns).

    :param df: full dataframe with features first and last two columns as targets
    :param all_means: optional precomputed means across all columns
    :param all_stds: optional precomputed stds across all columns
    :param as_torch: return tensors if True, else pandas Series
    :returns: tuple of (means, stds) as torch.Tensor or pd.Series

    Notes
    - If all_means/all_stds are provided, we subset them to feature columns.
    - Otherwise, we compute stats over feature columns only.
    """
    feat_cols = df.columns[:-2]
    if all_means is not None and all_stds is not None:
        m = all_means[feat_cols]
        s = all_stds[feat_cols]
    else:
        m = df[feat_cols].mean()
        s = df[feat_cols].std(ddof=0)
    if as_torch:
        return (
            torch.tensor(m.values, dtype=torch.float32),
            torch.tensor(s.values, dtype=torch.float32),
        )
    else:
        return m, s
