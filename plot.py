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


def set_axes(ax, batch_ids, x_label=None, y_label=None, x_lim=None, y_lim=None):
    if x_lim is not None:
        ax.set_xlim(*x_lim)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if x_label is not None:
        ax.set_ylabel(y_label)
    [spine.set_linewidth(1.3) for spine in ax.spines.values()]
    num_batches = len(batch_ids)
    plot_ids = np.arange(0, num_batches, 5)
    # plot_ids = np.arange(0, num_batches + 1, 5)
    batch_ids_mapped = [batch_ids[i] for i in plot_ids]
    plt.xticks(plot_ids, batch_ids_mapped)


def plot_threshold_metrics(hyb_path_thresh, figsize=(6,5)):
    seeds = [0, 1, 2]
    hybrid_accs, amort_accs, num_test_iters = [], [], []
    batch_ids = []
    for seed in seeds:
        hyb_seed_path = hyb_path_thresh + "/" + str(seed)
        hyb_metrics = utils.load_json(hyb_seed_path + "/metrics.json")
        hybrid_accs.append(hyb_metrics["hybrid_acc"])
        amort_accs.append(hyb_metrics["amort_acc"])
        num_test_iters.append(hyb_metrics["num_test_iters"])
        batch_ids.append(hyb_metrics["batch_idx"])

    batch_ids = batch_ids[0]

    hyb_mean, _, hyb_upper, hyb_lower = get_mean_std(hybrid_accs)
    amort_mean, _, amort_upper, amort_lower = get_mean_std(amort_accs)
    test_iters_mean, _, test_iters_upper, test_iters_lower = get_mean_std(num_test_iters)

    _, ax = plt.subplots(figsize=figsize)
    plot_mean_std(ax, test_iters_mean, test_iters_upper, test_iters_lower, PALETTE[6], "Iterations")
    set_axes(ax, batch_ids, x_label="Batch", y_label="Iterations")
    plt.legend()
    plt.title("Number of variational iterations")
    plt.tight_layout()
    plt.savefig("figures/iterations.png", dpi=300)
    plt.show()

    _, ax = plt.subplots(figsize=figsize)
    plot_mean_std(ax, hyb_mean, hyb_upper, hyb_lower, PALETTE[3], "Hybrid PC")
    plot_mean_std(ax, amort_mean, amort_upper, amort_lower, PALETTE[2], "Amortised")
    set_axes(ax, batch_ids, x_label="Batch", y_label="Accuracy", y_lim=(0, 1.0))
    plt.legend()
    plt.title("Classification accuracy")
    plt.tight_layout()
    plt.savefig("figures/performance_thresh.png", dpi=300)
    plt.show()


def plot_metrics(hyb_path, pc_path, uid, figsize=(6, 4)):
    seeds = [0, 1, 2]
    hybrid_accs, pc_accs, amort_accs = [], [], []
    hybrid_losses, pc_losses = [], []
    hybrid_errs, pc_errs = [], []
    batch_ids = []
    for seed in seeds:
        hyb_seed_path = hyb_path + "/" + str(seed)
        pc_seed_path = pc_path + "/" + str(seed)

        hyb_metrics = utils.load_json(hyb_seed_path + "/metrics.json")
        pc_metrics = utils.load_json(pc_seed_path + "/metrics.json")

        batch_ids.append(hyb_metrics["batch_idx"])
        hybrid_accs.append(hyb_metrics["hybrid_acc"])
        pc_accs.append(pc_metrics["pc_acc"])
        amort_accs.append(hyb_metrics["amort_acc"])
        hybrid_errs.append(hyb_metrics["pc_errs"])
        pc_errs.append(pc_metrics["pc_errs"])
        hybrid_losses.append(hyb_metrics["pc_losses"])
        pc_losses.append(pc_metrics["pc_losses"])

    batch_ids = batch_ids[0]

    hyb_mean, _, hyb_upper, hyb_lower = get_mean_std(hybrid_accs)
    pc_mean, _, pc_upper, pc_lower = get_mean_std(pc_accs)
    amort_mean, _, amort_upper, amort_lower = get_mean_std(amort_accs)

    hybrid_errs = np.array(hybrid_errs) / 64.0 
    pc_errs = np.array(pc_errs) / 64.0 
    hyb_err_mean, _, hyb_err_upper, hyb_err_lower = get_mean_std(hybrid_errs)
    pc_err_mean, _, pc_err_upper, pc_err_lower = get_mean_std(pc_errs)

    hybrid_losses = np.array(hybrid_losses) / 64.0 
    pc_losses = np.array(pc_losses) / 64.0 
    hyb_loss_mean, _, hyb_loss_upper, hyb_loss_lower = get_mean_std(hybrid_losses)
    pc_loss_mean, _, pc_loss_upper, pc_loss_lower = get_mean_std(pc_losses)


    _, ax = plt.subplots(figsize=figsize)
    plot_mean_std(ax, hyb_mean, hyb_upper, hyb_lower, PALETTE[3], "Hybrid PC")
    plot_mean_std(ax, pc_mean, pc_upper, pc_lower, PALETTE[1], "Predictive Coding")
    plot_mean_std(ax, amort_mean, amort_upper, amort_lower, PALETTE[2], "Amortised")
    set_axes(ax, batch_ids, x_label="Batches", y_label="Accuracy", y_lim=(0, 1.0))
    plt.legend()
    plt.title(f"Classification accuracy ({uid} iterations)")
    plt.tight_layout()
    plt.savefig(f"figures/{uid}_performance.png", dpi=300)
    plt.show()

    _, ax = plt.subplots(figsize=figsize)
    plot_mean_std(ax, hyb_err_mean, hyb_err_upper, hyb_err_lower, PALETTE[3], "Hybrid errors")
    plot_mean_std(ax, pc_err_mean, pc_err_upper, pc_err_lower, PALETTE[1], "PC errors")
    set_axes(ax, batch_ids, x_label="Batches", y_label="Prediction errors", y_lim=(0, 250))
    plt.legend()
    plt.title(f"Errors ({uid} iterations)")
    plt.tight_layout()
    plt.savefig(f"figures/{uid}_errors.png", dpi=300)
    plt.show()

    _, ax = plt.subplots(figsize=figsize)
    plot_mean_std(ax, hyb_loss_mean, hyb_loss_upper, hyb_loss_lower, PALETTE[3], "Hybrid losses")
    plot_mean_std(ax, pc_loss_mean, pc_loss_upper, pc_loss_lower, PALETTE[1], "PC losses")
    set_axes(ax, batch_ids, x_label="Batches", y_label="Data Loss", y_lim=(0, 120))
    plt.legend()
    plt.title(f"Generation losses ({uid} iterations)")
    plt.tight_layout()
    plt.savefig(f"figures/{uid}_losses.png", dpi=300)
    plt.show()


def plot_datasize(hyb_path, uid, figsize=(6, 4)):
    seeds = [0, 1, 2]
    hybrid_accs, amort_accs = [], []
    hybrid_losses = []
    hybrid_errs = []
    batch_ids = []
    for seed in seeds:
        hyb_seed_path = hyb_path + "/" + str(seed)

        hyb_metrics = utils.load_json(hyb_seed_path + "/metrics.json")
        batch_ids.append(hyb_metrics["batch_idx"])
        hybrid_accs.append(hyb_metrics["hybrid_acc"])
        amort_accs.append(hyb_metrics["amort_acc"])
        hybrid_errs.append(hyb_metrics["pc_errs"])
        hybrid_losses.append(hyb_metrics["pc_losses"])

    batch_ids = batch_ids[0]

    hyb_mean, _, hyb_upper, hyb_lower = get_mean_std(hybrid_accs)
    amort_mean, _, amort_upper, amort_lower = get_mean_std(amort_accs)

    hybrid_errs = np.array(hybrid_errs) / 64.0 
    hyb_err_mean, _, hyb_err_upper, hyb_err_lower = get_mean_std(hybrid_errs)

    hybrid_losses = np.array(hybrid_losses) / 64.0 
    hyb_loss_mean, _, hyb_loss_upper, hyb_loss_lower = get_mean_std(hybrid_losses)

    _, ax = plt.subplots(figsize=figsize)
    plot_mean_std(ax, hyb_mean, hyb_upper, hyb_lower, PALETTE[3], "Hybrid PC")
    plot_mean_std(ax, amort_mean, amort_upper, amort_lower, PALETTE[2], "Amortised")

    set_axes(ax, batch_ids, x_label="Batches", y_label="Accuracy", y_lim=(0, 1.0))
    plt.legend()
    plt.title(f"Classification accuracy ({uid} examples)")
    plt.tight_layout()
    plt.savefig(f"figures/batch_{uid}_performance.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # plot_metrics("results/hybrid_5", "results/predcoding_5", "5", figsize=(6,5))
    # plot_metrics("results/hybrid_10", "results/predcoding_10", "10", figsize=(6,5))
    # plot_metrics("results/hybrid_25", "results/predcoding_25", "25", figsize=(6,5))
    # plot_metrics("results/hybrid_50", "results/predcoding_50", "50", figsize=(6,5))
    # plot_metrics("results/hybrid_100", "results/predcoding_100", "100", figsize=(6,5))
    plot_datasize("results/hybrid_datasize_100", "100", figsize=(6,5))
    plot_datasize("results/hybrid_datasize_500", "500", figsize=(6,5))
    plot_datasize("results/hybrid_datasize_1000", "1000", figsize=(6,5))
    plot_datasize("results/hybrid_datasize_5000", "5000", figsize=(6,5))
    # plot_threshold_metrics("results/hybrid_thresh", figsize=(6,5))

