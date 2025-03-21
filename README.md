# CO2-catalysis
Repository to accompany paper A physics-based data-driven model for CO2 gas diffusion electrodes to drive automated laboratories (https://arxiv.org/abs/2502.06323v1)

Core files:

```
├── analytical_models
    ├── gde_multi.py # Analytical model for multi-product CO2 electrolysis. Based on paper doi:10.1016/j.electacta.2021.138987
    ├── loaders.py   # Utilities for loading data
├── paper
    ├── Characterization_data.xlsx  # Data used in the work. Comes from automated lab (doi:10.26434/chemrxiv-2024-znh0f)
    ├── *.ipynb  # Jupyter notebooks showcasing some uses of the model. The notebooks reproduce the plots in the paper.
```

There is `pyproject.toml` which can be used to install the core files as python package by running
```
pip install -e .
``` 
Then it can be imported from any location e.g.
```
from analytical_models.gde_multi import System
```
