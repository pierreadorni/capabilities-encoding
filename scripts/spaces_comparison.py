from capemb.data import load_data, mask_gt, to_relative
from capemb.utils import inputs_summary
from scripts.train import mean_confidence_interval
from capemb.optimization import optimize_gd, distance_computors, full_distance_computors
import jax.numpy as jnp
import numpy as np
from jax import jit
from capemb.plots import plot_dists, plot_umap
from matplotlib import pyplot as plt
import umap
from dataclasses import dataclass
import pandas as pd
import seaborn as sns


sns.set_context("talk")

@dataclass
class Args:
    dims: int
    init: str
    lr: float
    n_iter: int
    val: int
    cv_folds: int
    normalize: bool
    dist: str
    no_umap: bool
    freeze_encoder: bool


args = Args(5, "random", 0.05, 500, 10, 10, True, "l2", True, False)


def plot_dists(all_dists, all_masked_indexes, data_numpy, ax):
    # plot the estimated distance vs the ground truth distance for train and val for all folds on the same plot
    x_train = []
    x_val = []
    y_train = []
    y_val = []
    c_train = []
    c_val = []
    errors_train = []
    errors_val = []
    for dists, masked_indexes in zip(all_dists, all_masked_indexes):
        x_train += data_numpy[~masked_indexes].tolist()
        x_val += data_numpy[masked_indexes].tolist()
        y_train += dists[~masked_indexes].tolist()
        y_val += dists[masked_indexes].tolist()
        errors_train += np.abs(
            dists[~masked_indexes] - data_numpy[~masked_indexes]
        ).tolist()
        errors_val += np.abs(
            dists[masked_indexes] - data_numpy[masked_indexes]
        ).tolist()
        # count the number of non-null values per row
        row_counts = np.sum(~np.isnan(data_numpy), axis=0)
        row_counts = row_counts[None, :]
        c_train += row_counts.repeat(data_numpy.shape[0], axis=0)[
            ~masked_indexes
        ].tolist()
        c_val += row_counts.repeat(data_numpy.shape[0], axis=0)[masked_indexes].tolist()

    # build a df with x, y, count and type
    predictions_df = pd.DataFrame(
        {
            "x": x_train[::10],
            "y": y_train[::10],
            "Number of Fine-tunings": c_train[::10],
            "Split": "Train",
        }
    )
    predictions_df = pd.concat(
        [
            predictions_df,
            pd.DataFrame(
                {
                    "x": x_val,
                    "y": y_val,
                    "Number of Fine-tunings": c_val,
                    "Split": "Val",
                }
            ),
        ],
        ignore_index=True,
    )

    plt.rcParams.update({"font.size": 14})
    plt.rcParams.update({"axes.labelsize": 14})
    plt.rcParams.update({"xtick.labelsize": 14})
    plt.rcParams.update({"ytick.labelsize": 14})
    sns.scatterplot(
        predictions_df,
        x="x",
        y="y",
        # hue="Number of Fine-tunings",
        hue="Split",
        # palette="viridis",
        # style="Split",
        ax=ax,
    )
    sns.despine()
    # plt.colorbar(sm, label="Number of fine-tunings", ax=ax)
    # remove the color scale from the legend
    handles, labels = ax.get_legend_handles_labels()
    legend_elements = [h for h, l in zip(handles, labels) if l == "Train" or l == "Val"]
    legend_labels = [l for l in labels if l == "Train" or l == "Val"]
    ax.legend(legend_elements, legend_labels)
    ax.set(xlabel="Ground-Truth $\Delta$", ylabel="Estimated Distance")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # turn off the axes (box) for the ax
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    
    return (
        predictions_df["Number of Fine-tunings"].min(),
        predictions_df["Number of Fine-tunings"].max(),
    )


def main():
    data = load_data(path="data/data.csv")
    print(inputs_summary(data, args))
    data_numpy = data.to_numpy()
    if args.normalize:
        data_numpy = to_relative(data_numpy)

    metrics = ["l2", "cosine", "poincare"]
    metrics_pretty = ["L2", "Cosine", "Poincaré"]
    dimensions = [5, 15, 5]
    fig, ax = plt.subplots(1, len(metrics), figsize=(6 * len(metrics) + 6, 6))
    global_min, global_max = np.inf, -np.inf

    for i, metric in enumerate(metrics):
        args.dist = metric
        args.dims = dimensions[i]

        all_dists = []
        all_masked_indexes = []

        val_errors = []
        optims_params_history = []
        for _ in range(args.cv_folds):
            masked_data_numpy, masked_indexes = mask_gt(data_numpy, args.val)

            @jit
            def compute_val_error(dists):
                return jnp.mean(jnp.abs(dists - data_numpy)[masked_indexes])

            row_counts = np.sum(~np.isnan(data_numpy), axis=0)
            optim_params = optimize_gd(
                masked_data_numpy, args, compute_val_error, weights=None
            )
            optims_params_history.append(optim_params)
            distance_computor = distance_computors[args.dist]
            dists = distance_computor(optim_params, data.shape[1], args.dims)
            all_dists.append(dists)
            val_errors.append(compute_val_error(dists))
            all_masked_indexes.append(masked_indexes)

        val_errors = jnp.array(val_errors)

        optim_params = optims_params_history[np.argmin(val_errors)]
        optim_coords = (
            optim_params
            if args.dist != "mlp"
            else optim_params[: (data.shape[1] + data.shape[0]) * args.dims]
        )
        optim_coords = optim_coords.reshape(-1, args.dims)

        min_, max_ = plot_dists(all_dists, all_masked_indexes, data_numpy, ax[i])
        ax[i].set_title(metrics_pretty[i])
        global_min = min(global_min, min_)
        global_max = max(global_max, max_)

    norm = plt.Normalize(global_min, global_max)
    # sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    # plt.colorbar(sm, label="Number of fine-tunings", ax=ax)
    plt.savefig("plots/error.pdf", dpi=300, bbox_inches="tight")
    plt.rcParams.update(plt.rcParamsDefault)
    print("end !")


if __name__ == "__main__":
    main()
