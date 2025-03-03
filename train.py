import numpy as np
import jax.numpy as jnp
from jax import jit
import jax
import matplotlib.pyplot as plt
import umap
import argparse
import warnings
import scipy.stats
from adjustText import adjust_text
import seaborn as sns
from utils import inputs_summary
from data import load_data, to_relative, mask_gt
import pandas as pd
from optimization import (
    optimize_gd,
    deserialize_network_params,
    distance_computors,
    full_distance_computors,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_debug_nans", True)

warnings.simplefilter(action="ignore", category=FutureWarning)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


def make_dists_square(dists):
    n = dists.shape[0]
    m = dists.shape[1]
    total = max(n, m)
    square_dists = np.full((total, total), np.nan)
    square_dists[:n, :m] = dists
    square_dists[:m, :n] = dists.T
    return square_dists


def main(args: argparse.Namespace):
    data = load_data(path=args.data)
    print(inputs_summary(data, args))
    data_numpy = data.to_numpy()
    if args.normalize:
        data_numpy = to_relative(data_numpy)

    all_dists = []
    all_masked_indexes = []

    val_errors = []
    optims_params_history = []
    for i in range(args.cv_folds):
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

    # compute val error statistics
    val_errors = jnp.array(val_errors)
    mean, low, high = mean_confidence_interval(val_errors)
    print(f"Mean val error: {mean} ± {mean - low}")
    print(f"Std val error: {jnp.std(val_errors)}")
    print(f"Median val error: {jnp.median(val_errors)}")
    print(
        f"Min val error: {jnp.min(val_errors)}, fold number {np.argmin(val_errors) + 1}"
    )
    optim_params = optims_params_history[np.argmin(val_errors)]
    # unravel the parameters

    np.save("results/optim_params.npy", optim_params)

    optim_coords = (
        optim_params
        if args.dist != "mlp"
        else optim_params[: (data.shape[1] + data.shape[0]) * args.dims]
    )
    optim_coords = optim_coords.reshape(-1, args.dims)

    if not args.no_umap:
        reducer = umap.UMAP(metric="precomputed", n_neighbors=100, min_dist=0)
        distance_matrix = full_distance_computors[args.dist](optim_params, args.dims)
        optim_coords = reducer.fit_transform(distance_matrix)

    if args.dims <= 2 or not args.no_umap:
        # plot the results
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.scatter(
            optim_coords[: data.shape[1], 0],
            optim_coords[: data.shape[1], 1],
            color="blue",
            label="Downstream tasks",
        )
        ax.scatter(
            optim_coords[data.shape[1] :, 0],
            optim_coords[data.shape[1] :, 1],
            color="orange",
            label="Models",
        )
        texts = []
        for i, txt in enumerate(data.columns):
            texts.append(
                ax.text(
                    optim_coords[i, 0],
                    optim_coords[i, 1],
                    txt,
                    ha="center",
                    va="center",
                )
            )
        for i, txt in enumerate(data.index):
            texts.append(
                ax.text(
                    optim_coords[i + data.shape[1], 0],
                    optim_coords[i + data.shape[1], 1],
                    txt,
                    ha="center",
                    va="center",
                )
            )

        for t in texts:
            t.set_transform(ax.transData)

        adjust_text(
            texts,
            x=optim_coords[:, 0],
            y=optim_coords[:, 1],
            # force_points=0.15,
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

        # remove the ticks and box
        plt.xticks([])
        plt.yticks([])
        plt.box(False)

        plt.legend()

        plt.savefig("plots/result.png", bbox_inches="tight", dpi=300)
        print("Saved the result in result.png")

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

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.rcParams.update({"font.size": 14})
    plt.rcParams.update({"axes.labelsize": 14})
    plt.rcParams.update({"xtick.labelsize": 14})
    plt.rcParams.update({"ytick.labelsize": 14})
    norm = plt.Normalize(
        predictions_df["Number of Fine-tunings"].min(),
        predictions_df["Number of Fine-tunings"].max(),
    )
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sns.scatterplot(
        predictions_df,
        x="x",
        y="y",
        hue="Number of Fine-tunings",
        palette="viridis",
        style="Split",
        ax=ax,
    )
    plt.colorbar(sm, label="Nombre de fine-tunings", ax=ax)
    # remove the color scale from the legend
    handles, labels = ax.get_legend_handles_labels()
    legend_elements = [h for h, l in zip(handles, labels) if l == "Train" or l == "Val"]
    legend_labels = [l for l in labels if l == "Train" or l == "Val"]
    ax.legend(legend_elements, legend_labels)
    plt.ylabel("Distance estimée $d$", fontsize=14)
    plt.xlabel("Distance réelle $\Delta$", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.box(False)
    plt.savefig("plots/error.pdf", dpi=300, bbox_inches="tight")
    plt.rcParams.update(plt.rcParamsDefault)

    # plot the train and test errors as a function of the counts per row (or col), that is, c_train and c_val
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].scatter(c_train, errors_train, c="blue", label="Train")
    ax[0].set_xlabel("Number of non-null values per row")
    ax[0].set_ylabel("Error")
    ax[0].set_title("Train error")
    ax[1].scatter(c_val, errors_val, c="orange", label="Val")
    ax[1].set_xlabel("Number of non-null values per row")
    ax[1].set_ylabel("Error")
    ax[1].set_title("Val error")
    plt.savefig("plots/error_per_row.png")
    print("end !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Embed the given dataset in a n-dimensional space using MultiDimensional Scaling.",
    )
    parser.add_argument("data", type=str, help="Path to the dataset to embed.")
    parser.add_argument(
        "--lr", type=float, default=5e-2, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=5000,
        help="Number of iterations for the optimizer.",
    )
    parser.add_argument(
        "--dist",
        type=str,
        default="l2",
        help=f"Distance to use for the optimization. Can be any value from {list(distance_computors.keys())}.",
    )
    parser.add_argument(
        "--dims", type=int, default=10, help="Number of dimensions for the embedding."
    )
    parser.add_argument(
        "--no-umap",
        action="store_true",
        help="Do not use UMAP to reduce the dimensionality of the embedding. Only when the dimension is <= 2.",
    )
    parser.add_argument(
        "--val",
        type=int,
        default=50,
        help="Number of values to mask in the ground truth for validation.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Use normalized distances instead of absolute distances. The normalizes distances are computed by dividing each distance by the maximum distance in its column (task). The resulting number can be interpreted as distance to the best performance on the given task, 0 being the best performance and 1 being the worst.",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5, help="Number of folds for cross-validation."
    )
    parser.add_argument(
        "--init",
        type=str,
        default="random",
        help=f'Initialization for the coordinates. Can be "random" or any valid dist: {list(distance_computors.keys())}.',
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the encoder (latent space coordinates) and only train the decoder (MLP). This is only relevant when using the MLP distance, and is recommended only with a non-random initialization.",
    )
    args = parser.parse_args()

    if not args.normalize and args.dist == "cosine":
        warnings.warn(
            "Using cosine distance with a raw dataset is not recommended. The cosine distance is usually used with normalized data. Please consider using the --normalize flag to normalize the data."
        )

    if args.no_umap and args.dims > 2:
        warnings.warn(
            "UMAP is deactivated but the number of dimensions is greater than 2. No visualization will be computed."
        )

    if args.dist not in list(distance_computors.keys()):
        raise ValueError(
            f"Unknown distance {args.dist}. Available values are {list(distance_computors.keys())}."
        )

    if args.init not in list(distance_computors.keys()) and args.init != "random":
        raise ValueError(
            f"Unknown initialization {args.init}. Available values are {list(distance_computors.keys())} and 'random'."
        )

    if args.freeze_encoder and args.dist != "mlp":
        warnings.warn(
            "Freezing the encoder is only relevant when using the MLP distance. Because {args.dist} is an encoder-only distance, nothing will be learned. You should remove the --freeze-encoder flag."
        )

    main(args)
