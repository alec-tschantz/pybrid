import os
import pprint
import subprocess
import logging
import random
import json
import numpy as np
import torch

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def get_device():
    return _device


def setup_experiment(cfg):
    cfg = to_attr_dict(cfg)
    setup_logging()
    cfg.exp.log_dir, cfg.exp.img_dir = setup_logdirs(cfg.exp.log_dir, cfg.exp.seed)
    seed(cfg.exp.seed)
    logging.info(f"Starting experiment @ {cfg.exp.log_dir} [{get_device()}]")
    pprint.pprint(cfg)
    save_json(cfg, cfg.exp.log_dir + "/config.json")
    return cfg


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


def setup_logdirs(dir_name, seed):
    seed_dir_name = dir_name + "/" + str(seed)
    img_seed_dir_name = dir_name + "/" + str(seed) + "/imgs"
    os.makedirs(seed_dir_name, exist_ok=True)
    os.makedirs(img_seed_dir_name, exist_ok=True)
    return seed_dir_name, img_seed_dir_name


def to_attr_dict(_dict):
    attr_dict = AttrDict()
    for k, v in _dict.items():
        if isinstance(v, dict):
            v = to_attr_dict(v)
        attr_dict[k] = v
    return attr_dict


def setup_logdir(dir_name, seed):
    seed_dir_name = dir_name + "/" + str(seed)
    os.makedirs(seed_dir_name, exist_ok=True)
    return seed_dir_name


def seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_tensor(tensor):
    return tensor.to(get_device()).float()


def flatten_array(array):
    return torch.flatten(torch.cat(array, dim=1))


def save_json(obj, path):
    with open(path, "w") as file:
        json.dump(obj, file)


def load_json(path):
    with open(path) as file:
        return json.load(file)


def run_mnist_dl(data_dir):
    cmd = f"""curl https://www.di.ens.fr/~lelarge/MNIST.tar.gz -o MNIST.tar.gz
             tar -zxvf MNIST.tar.gz
             mv MNIST {data_dir}/MNIST
             rm MNIST.tar.gz
        """
    logging.info(f"Downloading MNIST into {data_dir}/MNIST...")
    subprocess.run(cmd, shell=True)


def get_act_fn(act_fn):
    if act_fn == "linear":
        return Linear()
    elif act_fn == "relu":
        return ReLU()
    elif act_fn == "tanh":
        return Tanh()
    else:
        raise ValueError(f"invalid act fn {act_fn}")


class Activation(object):
    def forward(self, inp):
        raise NotImplementedError

    def deriv(self, inp):
        raise NotImplementedError

    def __call__(self, inp):
        return self.forward(inp)


class Linear(Activation):
    def forward(self, inp):
        return inp

    def deriv(self, inp):
        return set_tensor(torch.ones((1,)))


class ReLU(Activation):
    def forward(self, inp):
        return torch.relu(inp)

    def deriv(self, inp):
        out = self(inp)
        out[out > 0] = 1.0
        return out


class Tanh(Activation):
    def forward(self, inp):
        return torch.tanh(inp)

    def deriv(self, inp):
        return 1.0 - torch.tanh(inp) ** 2.0
