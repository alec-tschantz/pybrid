import numpy as np
import matplotlib.pyplot as plt

from pybrid import utils

PALETTE = (
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#e41a1c",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#888888",
    "#a6cee3",
    "#b2df8a",
    "#cab2d6",
    "#fb9a99",
    "#fdbf6f",
)


def get_mean_std(arr):
    arr = np.array(arr)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    upper = mean + std
    lower = mean - std
    return mean, std, upper, lower


def plot_mean_std(ax, mean, upper, lower, color, label=""):
    x_len = mean.shape[0]
    ax.plot(range(x_len), mean, label=label, color=color)
    ax.fill_between(range(x_len), lower, upper, color=color, alpha=0.4)


def set_axes(ax, num_epochs, x_label=None, y_label=None, x_lim=None, y_lim=None):
    if x_lim is not None:
        ax.set_xlim(*x_lim)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if x_label is not None:
        ax.set_ylabel(y_label)
    [spine.set_linewidth(1.3) for spine in ax.spines.values()]
    plt.xticks(np.arange(0, num_epochs + 1, 5), np.arange(0, num_epochs + 1, 5))


def plot_hybrid_metrics():
    path = "results/hybrid"
    seeds = [0, 1, 2, 3]
    hybrid_accs, pc_accs, amort_accs = [], [], []
    for seed in seeds:
        seed_path = path + "/" + str(seed)
        metrics = utils.load_json(seed_path + "/metrics.json")
        hybrid_accs.append(metrics["hybrid_acc"])
        pc_accs.append(metrics["pc_acc"])
        amort_accs.append(metrics["amort_acc"])

    hyb_mean, _, hyb_upper, hyb_lower = get_mean_std(hybrid_accs)
    pc_mean, _, pc_upper, pc_lower = get_mean_std(pc_accs)
    amort_mean, _, amort_upper, amort_lower = get_mean_std(amort_accs)

    _, ax = plt.subplots(figsize=(6, 4))
    plot_mean_std(ax, hyb_mean, hyb_upper, hyb_lower, PALETTE[3], "Hybrid PC")
    plot_mean_std(ax, pc_mean, pc_upper, pc_lower, PALETTE[1], "Predictive Coding")
    plot_mean_std(ax, amort_mean, amort_upper, amort_lower, PALETTE[2], "Amortised")
    set_axes(ax, hyb_mean.shape[0], x_label="Epoch", y_label="Accuracy", y_lim=(0, 1.0))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_hybrid_metrics()