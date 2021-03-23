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


def plot_threshold_metrics(hyb_path_thresh, pc_path_thresh):
    seeds = [0, 1, 2]
    hybrid_accs, pc_accs, amort_accs, num_test_iters, num_test_iters_pc = [], [], [], [], []
    for seed in seeds:
        hyb_seed_path = hyb_path_thresh + "/" + str(seed)
        pc_seed_path = pc_path_thresh + "/" + str(seed)
        hyb_metrics = utils.load_json(hyb_seed_path + "/metrics.json")
        pc_metrics = utils.load_json(pc_seed_path + "/metrics.json")
        hybrid_accs.append(hyb_metrics["hybrid_acc"])
        pc_accs.append(pc_metrics["pc_acc"])
        amort_accs.append(hyb_metrics["amort_acc"])
        num_test_iters.append(hyb_metrics["num_test_iters"])
        num_test_iters_pc.append(pc_metrics["num_test_iters_pc"])

    hyb_mean, _, hyb_upper, hyb_lower = get_mean_std(hybrid_accs)
    pc_mean, _, pc_upper, pc_lower = get_mean_std(pc_accs)
    amort_mean, _, amort_upper, amort_lower = get_mean_std(amort_accs)
    test_iters_mean, _, test_iters_upper, test_iters_lower = get_mean_std(num_test_iters)
    test_iters_mean_pc, _, test_iters_upper_pc, test_iters_lower_pc = get_mean_std(
        num_test_iters_pc
    )

    _, ax = plt.subplots(figsize=(6, 4))
    plot_mean_std(ax, test_iters_mean, test_iters_upper, test_iters_lower, PALETTE[6], "Iterations")
    set_axes(ax, hyb_mean.shape[0], x_label="Epoch", y_label="Iterations")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/iterations.png", dpi=300)
    plt.show()

    _, ax = plt.subplots(figsize=(6, 4))
    plot_mean_std(
        ax, test_iters_mean_pc, test_iters_upper_pc, test_iters_lower_pc, PALETTE[9], "Iterations"
    )
    set_axes(ax, hyb_mean.shape[0], x_label="Epoch", y_label="Iterations")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/iterations_pc.png", dpi=300)
    plt.show()

    _, ax = plt.subplots(figsize=(6, 4))
    plot_mean_std(ax, hyb_mean, hyb_upper, hyb_lower, PALETTE[3], "Hybrid PC")
    plot_mean_std(ax, pc_mean, pc_upper, pc_lower, PALETTE[1], "Predictive Coding")
    plot_mean_std(ax, amort_mean, amort_upper, amort_lower, PALETTE[2], "Amortised")
    set_axes(ax, hyb_mean.shape[0], x_label="Epoch", y_label="Iterations")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/perform_thresh.png", dpi=300)
    plt.show()


def plot_hybrid_metrics(hyb_path, pc_path):
    seeds = [0, 1, 2]
    hybrid_accs, pc_accs, amort_accs, td_errs, bu_errs = [], [], [], [], []
    for seed in seeds:
        hyb_seed_path = hyb_path + "/" + str(seed)
        pc_seed_path = pc_path + "/" + str(seed)
        hyb_metrics = utils.load_json(hyb_seed_path + "/metrics.json")
        pc_metrics = utils.load_json(pc_seed_path + "/metrics.json")
        hybrid_accs.append(hyb_metrics["hybrid_acc"])
        pc_accs.append(pc_metrics["pc_acc"])
        amort_accs.append(hyb_metrics["amort_acc"])
        td_errs.append(hyb_metrics["pc_errs"])
        bu_errs.append(hyb_metrics["amort_errs"])

    hyb_mean, _, hyb_upper, hyb_lower = get_mean_std(hybrid_accs)
    pc_mean, _, pc_upper, pc_lower = get_mean_std(pc_accs)
    amort_mean, _, amort_upper, amort_lower = get_mean_std(amort_accs)

    td_mean, _, td_upper, td_lower = get_mean_std(td_errs)
    bu_mean, _, bu_upper, bu_lower = get_mean_std(bu_errs)

    _, ax = plt.subplots(figsize=(6, 4))
    plot_mean_std(ax, hyb_mean, hyb_upper, hyb_lower, PALETTE[3], "Hybrid PC")
    plot_mean_std(ax, pc_mean, pc_upper, pc_lower, PALETTE[1], "Predictive Coding")
    plot_mean_std(ax, amort_mean, amort_upper, amort_lower, PALETTE[2], "Amortised")
    set_axes(ax, hyb_mean.shape[0], x_label="Epoch", y_label="Accuracy", y_lim=(0, 1.0))
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/performance.png", dpi=300)
    plt.show()

    _, ax = plt.subplots(figsize=(6, 4))
    plot_mean_std(ax, td_mean, td_upper, td_lower, PALETTE[3], "Top-down errors")
    plot_mean_std(ax, bu_mean, bu_upper, bu_lower, PALETTE[1], "Bottom-up errors")
    set_axes(ax, hyb_mean.shape[0], x_label="Epoch", y_label="Prediction error")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/errors.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    hyb_path = "results/hybrid5"
    pc_path = "results/predcoding5"
    plot_hybrid_metrics(hyb_path, pc_path)

    hyb_path_thresh = "results/hybrid6"
    pc_path_thresh = "results/predcoding6"
    plot_threshold_metrics(hyb_path_thresh, pc_path_thresh)