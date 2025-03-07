import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
import pandas as pd
import seaborn as sns

def plot_umap(optim_coords, data):
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
    return fig, ax


def plot_dists(all_dists, all_masked_indexes, data_numpy):
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
    plt.colorbar(sm, label="Number of fine-tunings", ax=ax)
    # remove the color scale from the legend
    handles, labels = ax.get_legend_handles_labels()
    legend_elements = [h for h, l in zip(handles, labels) if l == "Train" or l == "Val"]
    legend_labels = [l for l in labels if l == "Train" or l == "Val"]
    ax.legend(legend_elements, legend_labels)
    plt.ylabel("Estimated Distance $d$", fontsize=14)
    plt.xlabel("Ground-Truth $\Delta$", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.box(False)
    return fig, ax