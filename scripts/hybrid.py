from pybrid.scripts import main
from pybrid.config import default_cfg as cfg

cfg.exp.log_dir = "results/local_hybrid"
cfg.exp.test_hybrid = True
cfg.exp.test_amort = True

cfg.data.train_size = 2000
cfg.data.test_size = 1000

cfg.infer.num_train_iters = 20
cfg.infer.num_test_iters = 100

cfg.infer.train_thresh = 0.001
cfg.infer.test_thresh = 0.001

main(cfg)