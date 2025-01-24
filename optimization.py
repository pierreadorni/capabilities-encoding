import jax.numpy as jnp
from jax import grad, jit
from functools import partial
import numpy as np
import optax
from tqdm import tqdm

def optimize_gd(data, args, compute_val_error):
    observed_distances = data
    observed_distances[np.isnan(observed_distances)] = np.inf
    gt = jnp.array(observed_distances)
    n_cols = data.shape[1]
    n_rows = data.shape[0]
    n_points = n_cols + n_rows
    coords = jnp.array(np.random.normal(0, 1, n_points * args.dims)) # init random coords
    loss, grad_loss = make_loss(args, n_cols, gt)
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(coords)
    distance_computor = compute_dists if args.dist == 'l2' else compute_cosine_dists
    t = tqdm(range(args.n_iter))
    for i in t:
        updates, opt_state = optimizer.update(grad_loss(coords), opt_state)
        coords = optax.apply_updates(coords, updates)
        if i % 10 == 0:
            dists = distance_computor(coords, n_cols, args.dims)
            # compute the mean error between the computed dists and the gt
            # gt contains some infinities, so we ignore them by setting them to dists
            mean_error = jnp.abs(dists - jnp.where(jnp.isinf(gt), dists, gt)).sum()/ (~jnp.isinf(gt)).sum()
            val_error = compute_val_error(dists)

            t.set_description(f'loss={loss(coords):.2f} train error={mean_error:.2f} val error={val_error:.2f}')
    return coords


def make_loss(args, n_cols, gt):
    @jit
    def l2_loss(coords):
        """
        this loss hypothesizes the ground truth accuracies to be the L2 distances between the datasets and models points in the latent space.
        """
        # coords is a 1D array of shape ((n_cols+n_rows) * n_dimensions,)
        # we want to separate the coordinates of the points called 'rows' and the ones called 'cols'
        # and compute the distances between each pair of points
        vec_coords = jnp.reshape(coords, (-1, args.dims))
        cols = vec_coords[:n_cols]
        rows = vec_coords[n_cols:]
        # repeat the rows and cols to be able to compute the distance between each pair of points
        cols = jnp.repeat(cols[:, None, :], rows.shape[0], axis=1)
        rows = jnp.repeat(rows[None, :, :], n_cols, axis=0)
        # compute the distance (L2), we clip the near-zero distances to 0.01 to avoid differentiation issues of the sqrt at 0
        dists = jnp.sqrt(jnp.sum(jnp.square(rows - cols), axis=-1).clip(0.01)).T
        # compare the obtained distances with the ground truth using mse
        return (((jnp.where(jnp.isinf(gt), dists, gt) - dists)**2).sum()/(~jnp.isinf(gt)).sum()).squeeze()
    
    @jit
    def cosine_loss(coords):
        """
        this loss hypothesizes the ground truth accuracies to be the cosine distances between the datasets and models points in the latent space.
        """
        # coords is a 1D array of shape ((n_cols+n_rows) * n_dimensions,)
        # we want to separate the coordinates of the points called 'rows' and the ones called 'cols'
        # and compute the distances between each pair of points
        vec_coords = jnp.reshape(coords, (-1, args.dims))
        cols = vec_coords[:n_cols]
        rows = vec_coords[n_cols:]
        # compute the cosine distance of all pair of points
        dots = rows @ cols.T
        cols_norms = jnp.expand_dims(jnp.linalg.norm(cols, axis=-1), 0)
        rows_norms = jnp.expand_dims(jnp.linalg.norm(rows, axis=-1), 1)
        norms = rows_norms @ cols_norms
        dists = (1 - (dots / norms))/2
        # compare the obtained distances with the ground truth using mse
        return (((jnp.where(jnp.isinf(gt), dists, gt) - dists)**2).sum()/(~jnp.isinf(gt)).sum()).squeeze()
    
    if args.dist == 'l2':
        return l2_loss, grad(l2_loss)
    elif args.dist == 'cosine':
        return cosine_loss, grad(cosine_loss)


@partial(jit, static_argnums=(1, 2))
def compute_dists(coords, n_cols, n_dimensions):
    # coords is a 1D array of shape ((n_cols+n_rows) * n_dimensions,)
    # we want to separate the coordinates of the points called 'rows' and the ones called 'cols'
    # and compute the distances between each pair of points
    vec_coords = jnp.reshape(coords, (-1, n_dimensions))
    cols = vec_coords[:n_cols]
    rows = vec_coords[n_cols:]
    # repeat the rows and cols to be able to compute the distance between each pair of points
    cols = jnp.repeat(cols[:, None, :], rows.shape[0], axis=1)
    rows = jnp.repeat(rows[None, :, :], n_cols, axis=0)
    # compute the L2
    dists = jnp.linalg.norm(rows - cols, axis=-1).T
    return dists

@partial(jit, static_argnums=(1, 2))
def compute_cosine_dists(coords, n_cols, n_dimensions):
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
    return dists/2