from pybrid import utils

cfg = {
    "exp": {"log_dir": "results/test", "seed": 0, "num_epochs": 20, "test_every": 1},
    "data": {"train_size": 1000, "test_size": 100, "label_scale": 0.94, "normalize": True},
    "infer": {
        "mu_dt": 0.01,
        "num_train_iters": 100,
        "num_test_iters": 1000,
        "fixed_preds_train": False,
        "fixed_preds_test": False,
        "init_std": 0.01,
        "train_thresh": None,
        "test_thresh": None,
    },
    "model": {
        "nodes": [10, 500, 500, 784],
        "amort_nodes": [784, 500, 500, 10],
        "train_amortised": True,
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
cfg = utils.to_attr_dict(cfg)