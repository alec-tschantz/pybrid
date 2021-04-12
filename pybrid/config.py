from pybrid import utils

default_cfg = {
    "exp": {
        "log_dir": "results/test",
        "seed": 0,
        "num_epochs": 20,
        "num_batches": 500,
        "batches_per_epoch": 50,
        "test_every": 1,
        "test_hybrid": False,
        "test_pc": False,
        "test_amort": False,
        "log_batch_every": 300,
        "switch": 2000
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
        "no_backward": False,
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
        "test_batch_size": None,
        "batch_scale": True,
        "grad_clip": 50,
        "weight_decay": None,
        "normalize_weights": True,
    },
}
default_cfg = utils.to_attr_dict(default_cfg)