from pybrid.split_scripts import main
from pybrid.config import default_cfg as cfg

cfg.exp.log_dir = "results/hybrid_split"
cfg.exp.num_batches = 3000
cfg.exp.batches_per_epoch = 100
cfg.exp.test_hybrid = True
cfg.exp.test_amort = True

cfg.data.train_size = None
cfg.data.test_size = None

cfg.infer.num_train_iters = 100
cfg.infer.num_test_iters = 100

cfg.infer.train_thresh = 0.005
cfg.infer.test_thresh = 0.005

seeds = [0, 1, 2]
for seed in seeds:
    cfg.exp.seed = seed
    main(cfg)