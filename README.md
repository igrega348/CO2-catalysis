# **Car**bon**Driver**

An api and a collection physics-based uncertainty-aware models to drive automated CO2RR laboratories

See our paper: [A physics-based data-driven model for CO2 gas diffusion electrodes to drive automated laboratories](https://arxiv.org/abs/2502.06323v1)

## Install

```
pip install git+https://github.com/igrega348/CO2-catalysis.git@active_learning_api
```

## Example

```
 from carbondriver import GDEOptimizer
 from carbondriver.loaders import load_data
 
 gde = GDEOptimizer("MLP", output_dir="./tmp_test_out")

 _, _, _, _, df = load_data("paper/Characterization_data.xlsx")

 df_train = df.loc[:30]
 df_explore = df.loc[31:]

 ei, next_pick = gde.step_within_data(df_train, df_explore)

 print("The next experiment to try is the following:")
 print(df_explore.iloc[int(next_pick)])

```