import jax.numpy as jnp
import jax
from jax import jit
from functools import partial
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from capemb.data import load_data, to_relative, mask_gt
from capemb.utils import inputs_summary
from capemb.optimization import optimize_gd, distance_computors
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")


@dataclass
class config:
    lr: float = 5e-2
    n_iter: int = 500
    dist: str = "cosine"
    dims: int = 2000
    no_umap: bool = True
    val: int = 20
    normalize: bool = True
    cv_folds: int = 5
    init: str = "random"
    freeze_encoder: bool = False
    save_file: str = "results/errors_cleaned.pickle"


@partial(jit, static_argnums=(1, 2))
def compute_cosine_dists(coords, n_cols, n_dimensions, scale, translate):
    # coords is a 1D array of shape ((n_cols+n_rows) * n_dimensions,)
    # we want to separate the coordinates of the points called 'rows' and the ones called 'cols'
    # and compute the distances between each pair of points
    vec_coords = jnp.reshape(coords, (-1, n_dimensions))
    cols = vec_coords[:n_cols]
    rows = vec_coords[n_cols:]
    # compute the cosine distance of all pair of points
    dots = rows @ cols.T
    cols_norms = jnp.expand_dims(jnp.linalg.norm(cols, axis=-1), 0)
    rows_norms = jnp.expand_dims(jnp.linalg.norm(rows, axis=-1), 1)
    norms = rows_norms @ cols_norms
    dists = 1 - (dots / norms)
    return jnp.clip(dists - 0.5, 0, 1)


def dimension_grid_search():
    dims_list = list(range(1000, 11000, 1000))
    val_errors = []

    for dims in tqdm(dims_list):
        args = config(dims=dims, cv_folds=15)
        data = load_data(path="data.csv")
        data_numpy = data.to_numpy()
        if args.normalize:
            data_numpy = to_relative(data_numpy)

        fold_errors = []
        for _ in range(args.cv_folds):
            masked_data_numpy, masked_indexes = mask_gt(data_numpy, args.val)

            @jit
            def compute_val_error(dists, scale=1, translate=0):
                return jnp.mean(
                    jnp.abs(jnp.clip(dists / scale - translate, 0, 1) - data_numpy)[
                        masked_indexes
                    ]
                )

            optim_params = optimize_gd(
                masked_data_numpy, args, compute_val_error, verbose=False
            )
            dists = compute_cosine_dists(optim_params, data.shape[1], dims, 1, 0.5)
            fold_errors.append(compute_val_error(dists, scale=0.5, translate=0.5))

        val_errors.append(jnp.mean(jnp.array(fold_errors)))

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(dims_list, val_errors)
    ax1.set_xlabel("Dimensions")
    ax1.set_ylabel("Validation Error")
    ax1.set_title("Linear Scale")

    ax2.semilogx(dims_list, val_errors)
    ax2.set_xlabel("Dimensions (log scale)")
    ax2.set_ylabel("Validation Error")
    ax2.set_title("Log Scale")

    plt.tight_layout()
    plt.show()


def main():
    data = load_data(path="data/data_cleaned.csv")
    args = config()
    print(inputs_summary(data, args))

    data_numpy = data.to_numpy()
    if args.normalize:
        data_numpy = to_relative(data_numpy)

    all_dists = []
    all_masked_indexes = []
    all_optim_params = []

    # scale = 0.5
    # translate = 0.22
    scale = 1
    translate = 0

    val_errors = []
    for i in range(args.cv_folds):
        masked_data_numpy, masked_indexes = mask_gt(data_numpy, args.val)

        @jit
        def compute_val_error(dists, scale=scale, translate=translate):
            return jnp.mean(
                jnp.abs(jnp.clip(dists * scale - translate, 0, 1) - data_numpy)[
                    masked_indexes
                ]
            )

        optim_params = optimize_gd(masked_data_numpy, args, compute_val_error)
        all_optim_params.append(optim_params)
        dists = compute_cosine_dists(
            optim_params, data.shape[1], args.dims, scale, translate
        )
        all_dists.append(dists)
        val_errors.append(compute_val_error(dists))
        all_masked_indexes.append(masked_indexes)

    fig = plt.figure(figsize=(10, 10))
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
        y_val += jnp.clip((dists[masked_indexes] * scale - translate), 0, 1).tolist()
        errors_train += jnp.abs(
            dists[~masked_indexes] - data_numpy[~masked_indexes]
        ).tolist()
        errors_val += jnp.abs(
            dists[masked_indexes] - data_numpy[masked_indexes]
        ).tolist()
        # count the number of non-null values per row
        row_counts = jnp.sum(~jnp.isnan(data_numpy), axis=0)
        row_counts = row_counts[None, :]
        c_train += row_counts.repeat(data_numpy.shape[0], axis=0)[
            ~masked_indexes
        ].tolist()
        c_val += row_counts.repeat(data_numpy.shape[0], axis=0)[masked_indexes].tolist()

    train_plot = plt.scatter(x_train, y_train, c=c_train, cmap="viridis", label="Train")
    val_plot = plt.scatter(
        x_val, y_val, c=c_val, cmap="viridis", label="Val", marker="x"
    )
    # plot the trend line (polyfit) of the val errors
    trend_line = plt.plot(
        jnp.linspace(0, 1, 100),
        jnp.polyval(
            jnp.polyfit(jnp.array(x_val), jnp.array(y_val), 1),
            jnp.linspace(0, 1, 100),
        ),
        color="green",
        label="Val error trend",
    )
    # plot the identity line in red
    plt.plot([0, 1], [0, 1], color="red")
    plt.colorbar()
    plt.ylabel("Estimated distance")
    plt.xlabel("Ground truth distance")
    plt.legend()

    # plot the val acc in the bottom right corner
    val_error_text = plt.text(
        0.5,
        0.2,
        f"Val error: {jnp.mean(jnp.array(val_errors).flatten()):.2f}",
        fontsize=12,
        transform=plt.gcf().transFigure,
    )

    ax_scale = plt.axes([0.25, 0.01, 0.65, 0.03])
    ax_translate = plt.axes([0.25, 0.06, 0.65, 0.03])
    s_scale = Slider(ax_scale, "Scale", 0.0, 3.0, valinit=scale)
    s_translate = Slider(ax_translate, "Translate", 0.0, 1.0, valinit=translate)

    def update():
        new_all_dists = []
        for i in range(args.cv_folds):
            dists = compute_cosine_dists(
                all_optim_params[i],
                data.shape[1],
                args.dims,
                1,
                0.5,
            )
            new_all_dists.append(dists)
        x_train = []
        x_val = []
        y_train = []
        y_val = []
        c_train = []
        c_val = []
        errors_train = []
        errors_val = []
        for dists, masked_indexes in zip(new_all_dists, all_masked_indexes):
            x_train += data_numpy[~masked_indexes].tolist()
            x_val += data_numpy[masked_indexes].tolist()
            y_train += dists[~masked_indexes].tolist()
            y_val += jnp.clip(
                ((dists[masked_indexes]) * s_scale.val - s_translate.val), 0, 1
            ).tolist()
            errors_train += jnp.abs(
                dists[~masked_indexes] - data_numpy[~masked_indexes]
            ).tolist()
            errors_val += jnp.abs(
                jnp.clip(
                    ((dists[masked_indexes]) * s_scale.val - s_translate.val), 0, 1
                )
                - data_numpy[masked_indexes]
            ).tolist()
            # count the number of non-null values per row
            row_counts = jnp.sum(~jnp.isnan(data_numpy), axis=0)
            row_counts = row_counts[None, :]
            c_train += row_counts.repeat(data_numpy.shape[0], axis=0)[
                ~masked_indexes
            ].tolist()
            c_val += row_counts.repeat(data_numpy.shape[0], axis=0)[
                masked_indexes
            ].tolist()

        train_plot.set_offsets(jnp.column_stack((x_train, y_train)))
        train_plot.set_array(jnp.array(c_train))
        val_plot.set_offsets(jnp.column_stack((x_val, y_val)))
        val_plot.set_array(jnp.array(c_val))
        trend_line[0].set_ydata(
            jnp.polyval(
                jnp.polyfit(jnp.array(x_val), jnp.array(y_val), 1),
                jnp.linspace(0, 1, 100),
            )
        )

        # update the val error text
        val_error_text.set_text(
            f"Val error: {jnp.mean(jnp.array(errors_val).flatten()):.2f}"
        )

        fig.canvas.draw_idle()

    s_scale.on_changed(lambda val: update())
    s_translate.on_changed(lambda val: update())

    plt.show()


if __name__ == "__main__":
    main()
