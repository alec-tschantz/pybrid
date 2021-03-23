# Hybrid inference: Inferring fast and slow

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
![status](https://img.shields.io/badge/status-development-orange)


## Getting started
To install the relevant packages:
```bash
pip install -r requirements.txt
```

## Running
To run one of the scripts
```bash
python -m scripts.hybrid
```

##  `pip`
You can install with `pip`, this is ideal for using the package with `colab`
```
pip install git+https://github.com/alec-tschantz/pybrid.git
```

## `config`
The `pybrid.config` module provides a default config. The scripts assume a structure with the following structure:

```python
default_cfg = {
    "exp": {
        "log_dir": "results/test",
        "seed": 0,
        "num_epochs": 20,
        "test_every": 1,
        "test_hybrid": False,
        "test_pc": False,
        "test_amort": False,
    },
    "data": {"train_size": None, "test_size": None, "label_scale": 0.94, "normalize": True},
    "infer": {
        "mu_dt": 0.01,
        "num_train_iters": 100,
        "num_test_iters": 500,
        "fixed_preds_train": False,
        "fixed_preds_test": False,
        "train_thresh": None,
        "test_thresh": None,
        "init_std": 0.01,
    },
    "model": {
        "nodes": [10, 500, 500, 784],
        "amort_nodes": [784, 500, 500, 10],
        "train_amort": True,
        "use_bias": True,
        "kaiming_init": False,
        "act_fn": "tanh",
    },
    "optim": {
        "name": "Adam",
        "lr": 1e-4,
        "amort_lr": 1e-4,
        "batch_size": 64,
        "batch_scale": True,
        "grad_clip": 5,
        "weight_decay": None,
        "normalize_weights": True,
    },
}
```

Importing the default config from `pybrid.config` provides sensible defaults for most parameters, and can be updated based on experimental needs. For example, we can test a vanilla predictive coding network:

```python
from pybrid.scripts import main
from pybrid.config import default_cfg as cfg

cfg.exp.log_dir = "results/predcoding" 
cfg.model.train_amort = False
cfg.exp.test_pc = True
main(cfg)
```