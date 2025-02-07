from dataclasses import dataclass
from data import load_data, to_relative, mask_gt
from utils import inputs_summary
import jax
from jax import jit
import jax.numpy as jnp
from optimization import optimize_gd, distance_computors
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


@dataclass
class config:
    lr: float = 5e-2
    n_iter: int = 500
    dist: str = "cosine"
    dims: int = 10
    no_umap: bool = True
    val: int = 20
    normalize: bool = True
    cv_folds: int = 1
    init: str = "random"
    freeze_encoder: bool = False
    save_file: str = "results/errors_cleaned.pickle"


def number_of_errors_per_index(errors) -> jnp.ndarray:
    max_x = max([key[0] for key in errors.keys()])
    max_y = max([key[1] for key in errors.keys()])
    result = jnp.zeros((max_x + 1, max_y + 1))
    for key, error in errors.items():
        result = result.at[key].set(len(error))
    return result


def mean_error_per_index(errors) -> jnp.ndarray:
    max_x = max([key[0] for key in errors.keys()])
    max_y = max([key[1] for key in errors.keys()])
    result = jnp.zeros((max_x + 1, max_y + 1))
    for key, error in errors.items():
        result = result.at[key].set(jnp.mean(jnp.array(error)))
    return result

def main():
    data = load_data(path='data_cleaned.csv')
    args = config()
    print(inputs_summary(data, args))

    data_numpy = data.to_numpy()
    if args.normalize:
        data_numpy = to_relative(data_numpy)

    try: 
        while True:
            masked_data_numpy, masked_indexes = mask_gt(data_numpy, args.val)
            all_errors = {}
            if os.path.exists(args.save_file):
                with open(args.save_file, "rb") as file:
                    all_errors = pickle.load(file)

            @jit
            def compute_val_error(dists):
                return jnp.mean(jnp.abs(dists - data_numpy)[masked_indexes])

            optim_params = optimize_gd(masked_data_numpy, args, compute_val_error)
            distance_computor = distance_computors[args.dist]
            dists = distance_computor(optim_params, data.shape[1], args.dims)

            errors = jnp.abs(dists - data_numpy)
            for index in jnp.argwhere(masked_indexes):
                index = tuple(index.tolist())
                if index in all_errors:
                    all_errors[index].append(float(errors[index]))
                else:
                    all_errors[index] = [float(errors[index])]

            with open(args.save_file, "wb") as file:
                pickle.dump(all_errors, file)

            # plot heatmap
            errors_per_index = mean_error_per_index(all_errors)
            plt.figure(figsize=(30, 15))
            fig = sns.heatmap(
                errors_per_index, xticklabels=data.columns, yticklabels=data.index
            )
            fig.get_figure().savefig("plots/heatmap.png")
            plt.close()
    except KeyboardInterrupt:
        print("Interrupted")
if __name__ == "__main__":
    main()
