import numpy as np
import jax.numpy as jnp
from jax import jit
import jax
import matplotlib.pyplot as plt
import umap
import argparse
import warnings

from utils import inputs_summary
from data import load_data, to_relative, mask_gt
from optimization import optimize_gd, compute_cosine_dists, compute_dists

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)

warnings.simplefilter(action='ignore', category=FutureWarning)



def main(args: argparse.Namespace):
    data = load_data()
    print(inputs_summary(data, args))
    data_numpy = data.to_numpy()
    if args.normalize:
        data_numpy = to_relative(data_numpy)

    all_dists = []
    all_masked_indexes = []

    for i in range(args.cv_folds):
        masked_data_numpy, masked_indexes = mask_gt(data_numpy, args.val)

        @jit
        def compute_val_error(dists):
            return jnp.mean(jnp.abs(dists - data_numpy)[masked_indexes])
        
        optim_coords = optimize_gd(masked_data_numpy, args, compute_val_error)
        optim_coords = optim_coords.reshape(-1, args.dims)
        distance_computor = compute_dists if args.dist == 'l2' else compute_cosine_dists
        dists = distance_computor(optim_coords, data.shape[1], args.dims)
        all_dists.append(dists)
        all_masked_indexes.append(masked_indexes)

    np.save('optim_coords.npy', optim_coords)

    if not args.no_umap:
        reducer = umap.UMAP(metric='cosine', n_neighbors=100, min_dist=0)
        optim_coords = reducer.fit_transform(optim_coords)

    if args.dims <= 2 or not args.no_umap:
        # plot the results
        plt.figure(figsize=(30, 30))
        plt.scatter(optim_coords[:data.shape[1], 0], optim_coords[:data.shape[1], 1], color='blue')
        plt.scatter(optim_coords[data.shape[1]:, 0], optim_coords[data.shape[1]:, 1], color='orange')
        for i, txt in enumerate(data.columns):
            plt.annotate(txt, (optim_coords[i, 0], optim_coords[i, 1]+0.02))
        for i, txt in enumerate(data.index):
            plt.annotate(txt, (optim_coords[i+data.shape[1], 0], optim_coords[i+data.shape[1], 1]+0.02))
        # for each masked point, draw a line between the two points
        # the color of which is more red if the distance is higher and more green if the distance is lower

        #for j,i in np.argwhere(masked_indexes):
        #    plt.plot([optim_coords[i, 0], optim_coords[j+data.shape[1], 0]], [optim_coords[i, 1], optim_coords[j+data.shape[1], 1]], color=plt.cm.RdYlGn(1 - data_numpy[j, i]))
                # plt.text((optim_coords[i, 0] + optim_coords[j+data.shape[1], 0]) / 2, (optim_coords[i, 1] + optim_coords[j+data.shape[1], 1]) / 2, f'{np.linalg.norm(optim_coords[i] - optim_coords[j+data.shape[1]]):.2f}', fontsize=7)
        plt.savefig('result.png',bbox_inches='tight')
        print("Saved the result in result.png")

    # plot the estimated distance vs the ground truth distance for train and val for all folds on the same plot
    plt.figure(figsize=(10, 10))
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
        # compute the error for each point
        errors_train += np.abs(dists[~masked_indexes] - data_numpy[~masked_indexes]).tolist()
        errors_val += np.abs(dists[masked_indexes] - data_numpy[masked_indexes]).tolist()
        # count the number of non-null values per row
        row_counts = np.sum(~np.isnan(data_numpy), axis=0)
        row_counts = row_counts[None, :]
        c_train += row_counts.repeat(data_numpy.shape[0], axis=0)[~masked_indexes].tolist()
        c_val += row_counts.repeat(data_numpy.shape[0], axis=0)[masked_indexes].tolist()

    plt.scatter(x_train, y_train, c=c_train, cmap='viridis', label='Train')
    plt.scatter(x_val, y_val, c=c_val, cmap='viridis', label='Val', marker='x')
    # plot the identity line in red
    plt.plot([0, 1], [0,1], color='red')
    plt.colorbar()
    plt.ylabel('Estimated distance')
    plt.xlabel('Ground truth distance')
    # plt.xlim(0, 40)
    # plt.ylim(0, 40)
    plt.legend()
    plt.savefig('error.png')

    # plot the train and test errors as a function of the counts per row (or col), that is, c_train and c_val
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].scatter(c_train, errors_train, c='blue', label='Train')
    ax[0].set_xlabel('Number of non-null values per row')
    ax[0].set_ylabel('Error')
    ax[0].set_title('Train error')
    ax[1].scatter(c_val, errors_val, c='orange', label='Val')
    ax[1].set_xlabel('Number of non-null values per row')
    ax[1].set_ylabel('Error')
    ax[1].set_title('Val error')
    plt.savefig('error_per_row.png')
    print("end !")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Embed the given dataset in a n-dimensional space using MultiDimensional Scaling."
    )
    parser.add_argument(
        "data",
        type=str,
        help="Path to the dataset to embed."
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=5e-2,
        help='Learning rate for the optimizer.'
    )
    parser.add_argument(
        '--n-iter',
        type=int,
        default=5000,
        help='Number of iterations for the optimizer.'
    )
    parser.add_argument(
        '--dist',
        type=str,
        default='l2',
        help='Distance to use for the optimization. Can be "l2" or "cosine".'
    )
    parser.add_argument(
        '--dims',
        type=int,
        default=10,
        help='Number of dimensions for the embedding.'
    )
    parser.add_argument(
        '--no-umap',
        action='store_true',
        help='Do not use UMAP to reduce the dimensionality of the embedding. Only when the dimension is <= 2.'
    )
    parser.add_argument(
        '--val',
        type=int,
        default=50,
        help='Number of values to mask in the ground truth for validation.'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Use normalized distances instead of absolute distances. The normalizes distances are computed by dividing each distance by the maximum distance in its column (task). The resulting number can be interpreted as distance to the best performance on the given task, 0 being the best performance and 1 being the worst.'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of folds for cross-validation.'
    )
    args = parser.parse_args()

    if not args.normalize and args.dist == 'cosine':
        warnings.warn("Using cosine distance with a raw dataset is not recommended. The cosine distance is usually used with normalized data. Please consider using the --normalize flag to normalize the data.")

    if args.no_umap and args.dims > 2:
        warnings.warn("UMAP is deactivated but the number of dimensions is greater than 2. No visualization will be computed.")

    if args.dist not in ['l2', 'cosine']:
        raise ValueError(f"Unknown distance {args.dist}. Available values are ['l2', 'cosine'].")

    main(args)