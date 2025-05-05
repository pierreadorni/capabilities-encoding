import jax.numpy as jnp
from jax import grad, jit, vmap, random
from jax.nn import sigmoid, gelu
from functools import partial
import numpy as np
import optax
from tqdm import tqdm
import copy

MLP_LAYER_SIZE = 32
MLP_LAYER_COUNT = 2


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
    dists = jnp.linalg.norm(cols - rows, axis=-1)
    dists = jnp.clip(dists, 0, 1)
    return dists.T


@partial(jit, static_argnums=(1))
def compute_full_l2_dists(coords, n_dimensions):
    # same as compute_dists but for all pairs of points (outputs a square matrix)
    vec_coords = jnp.reshape(coords, (-1, n_dimensions))
    # compute pairwise distances between all points
    expanded = jnp.expand_dims(vec_coords, 1)
    dists = jnp.linalg.norm(expanded - vec_coords, axis=-1)
    dists = jnp.clip(dists, 0, 1)
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
    return jnp.clip(dists - 0.5, 0, 1)


@partial(jit, static_argnums=(1))
def compute_full_cosine_dists(coords, n_dimensions):
    # same as above but for all pairs of points (outputs a square matrix)
    vec_coords = jnp.reshape(coords, (-1, n_dimensions))
    # compute the cosine distance of all pair of points
    dots = vec_coords @ vec_coords.T
    norms = jnp.expand_dims(jnp.linalg.norm(vec_coords, axis=-1), 1)
    norms = norms @ norms.T
    dists = 1 - (dots / norms)
    return jnp.clip(dists - 0.5, 0, 1)


@partial(jit, static_argnums=(1))
def compute_full_mlp_dists(params, n_dimensions):
    layer_sizes = [n_dimensions * 2] + [MLP_LAYER_SIZE] * MLP_LAYER_COUNT + [1]
    n_params = number_of_mlp_params(layer_sizes)
    coords, mlp_params = params[:-n_params], params[-n_params:]
    vec_coords = jnp.reshape(coords, (-1, n_dimensions))

    # initialize MLP and compute all pairwise predictions
    mlp_params = deserialize_network_params(mlp_params, layer_sizes)
    couples_indexes = jnp.stack(
        jnp.meshgrid(jnp.arange(len(vec_coords)), jnp.arange(len(vec_coords))), -1
    ).reshape(-1, 2)
    couples = vec_coords[couples_indexes]
    predictions = batch_mlp_predict(mlp_params, couples)
    return jnp.reshape(predictions, (len(vec_coords), len(vec_coords)))


@partial(jit, static_argnums=(1, 2))
def compute_mlp_dists(params, n_cols, n_dimensions):
    layer_sizes = [n_dimensions * 2] + [MLP_LAYER_SIZE] * MLP_LAYER_COUNT + [1]
    n_params = number_of_mlp_params(layer_sizes)
    coords, mlp_params = params[:-n_params], params[-n_params:]
    vec_coords = jnp.reshape(coords, (-1, n_dimensions))
    cols = vec_coords[:n_cols]
    rows = vec_coords[n_cols:]

    # initialize the MLP
    mlp_params = deserialize_network_params(mlp_params, layer_sizes)
    # compute the predictions of the MLP
    row_indexes = np.arange(0, rows.shape[0], 1)
    col_indexes = np.arange(0, cols.shape[0], 1)

    couples_indexes = jnp.stack(jnp.meshgrid(row_indexes, col_indexes), -1).reshape(
        -1, 2
    )

    couples = vec_coords[couples_indexes]
    predictions = batch_mlp_predict(mlp_params, couples)
    predictions = jnp.reshape(predictions, (n_cols, rows.shape[0])).T
    return predictions


@partial(jit, static_argnums=(1, 2, 3))
def compute_minkowski_dists(coords, n_cols, n_dimensions, p=0.8):
    # coords is a 1D array of shape ((n_cols+n_rows) * n_dimensions,)
    vec_coords = jnp.reshape(coords, (-1, n_dimensions))
    cols = vec_coords[:n_cols]
    rows = vec_coords[n_cols:]
    # repeat the rows and cols to be able to compute the distance between each pair of points
    cols = jnp.repeat(cols[:, None, :], rows.shape[0], axis=1)
    rows = jnp.repeat(rows[None, :, :], n_cols, axis=0)
    # compute Minkowski distance
    dists = jnp.sum(jnp.abs(cols - rows) ** p, axis=-1) ** (1 / p)
    return dists.T


@partial(jit, static_argnums=(1, 2))
def compute_full_minkowski_dists(coords, n_dimensions, p=0.8):
    # same as compute_minkowski_dists but for all pairs of points (outputs a square matrix)
    vec_coords = jnp.reshape(coords, (-1, n_dimensions))
    # compute pairwise Minkowski distances between all points
    expanded = jnp.expand_dims(vec_coords, 1)
    dists = jnp.sum(jnp.abs(expanded - vec_coords) ** p, axis=-1) ** (1 / p)
    return dists


@partial(jit, static_argnums=(1, 2))
def compute_poincare_dists(coords, n_cols, n_dimensions):
    vec_coords = jnp.reshape(coords, (-1, n_dimensions))
    cols = vec_coords[:n_cols]
    rows = vec_coords[n_cols:]

    # Compute pairwise distances in Poincaré ball model
    cols_norm = jnp.sum(cols**2, axis=-1, keepdims=True)
    rows_norm = jnp.sum(rows**2, axis=-1, keepdims=True)

    # Broadcast for pairwise computations
    cols = jnp.repeat(cols[:, None, :], rows.shape[0], axis=1)
    rows = jnp.repeat(rows[None, :, :], n_cols, axis=0)

    num = 2 * jnp.sum((cols - rows) ** 2, axis=-1)
    denom = (1 - cols_norm) * (1 - rows_norm.T)

    return jnp.arccosh(1 + num / denom).T


@partial(jit, static_argnums=(1))
def compute_full_poincare_dists(coords, n_dimensions):
    vec_coords = jnp.reshape(coords, (-1, n_dimensions))

    # Compute norms
    norms = jnp.sum(vec_coords**2, axis=-1, keepdims=True)

    # Compute pairwise squared distances
    expanded = jnp.expand_dims(vec_coords, 1)
    diff_squared = jnp.sum((expanded - vec_coords) ** 2, axis=-1)

    # Compute denominator terms
    denom = (1 - norms) @ (1 - norms.T)

    return jnp.arccosh(1 + 2 * diff_squared / denom)


distance_computors = {
    "l2": compute_dists,
    "cosine": compute_cosine_dists,
    "mlp": compute_mlp_dists,
    "minkowski": compute_minkowski_dists,
    "poincare": compute_poincare_dists,
}

full_distance_computors = {
    "l2": compute_full_l2_dists,
    "cosine": compute_full_cosine_dists,
    "mlp": compute_full_mlp_dists,
    "minkowski": compute_full_minkowski_dists,
    "poincare": compute_full_poincare_dists,
}


def optimize_gd(data, args, compute_val_error, verbose=True, weights=None):
    observed_distances = data
    observed_distances[np.isnan(observed_distances)] = np.inf
    gt = jnp.array(observed_distances)
    n_cols = data.shape[1]
    n_rows = data.shape[0]
    n_points = n_cols + n_rows
    coords = jnp.array(
        np.random.normal(0, 0.1, n_points * args.dims)
    )  # init random coords
    if args.dist == "poincare":
        # we need to project the random points inside the Poincaré ball (no point x with ||x|| > 1)
        vec_coords = jnp.reshape(coords, (-1, args.dims))
        norms = jnp.linalg.norm(vec_coords, axis=-1)
        norms = jnp.maximum(norms, 1)
        vec_coords = vec_coords / norms[:, None] * 0.001 - 0.0005
        coords = vec_coords.flatten()

    if args.init != "random":
        subargs = copy.deepcopy(args)
        subargs.init = "random"
        subargs.dist = args.init
        subargs.freeze_encoder = False
        coords = optimize_gd(data, subargs, compute_val_error)
        if args.init == "mlp":
            coords = coords[: (n_cols + n_rows) * args.dims]
    params = coords
    if args.dist == "mlp":
        params = jnp.concatenate(
            [
                coords,
                serialize_network_params(
                    init_network_params(
                        [args.dims * 2] + [MLP_LAYER_SIZE] * MLP_LAYER_COUNT + [1],
                        random.PRNGKey(0),
                    )
                ),
            ]
        )
    loss, grad_loss = make_loss(args, n_cols, gt, weights=weights)
    schedule = optax.schedules.warmup_cosine_decay_schedule(
        args.lr / 100, args.lr, 100, args.n_iter
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)
    distance_computor = distance_computors[args.dist]
    if verbose:
        t = tqdm(range(args.n_iter))
    else:
        t = range(args.n_iter)
    for i in t:
        updates, opt_state = optimizer.update(grad_loss(params), opt_state)
        if args.freeze_encoder:
            updates = updates.at[: (n_cols + n_rows) * args.dims].set(0)
        params = optax.apply_updates(params, updates)
        if args.dist == "poincare":
            params = poincare_sphere_projection(params, n_cols, args.dims)
        if i % 10 == 0:
            dists = distance_computor(params, n_cols, args.dims)
            # compute the mean error between the computed dists and the gt
            # gt contains some infinities, so we ignore them by setting them to dists
            mean_error = (
                jnp.abs(dists - jnp.where(jnp.isinf(gt), dists, gt)).sum()
                / (~jnp.isinf(gt)).sum()
            )

            val_error = compute_val_error(dists)

            if verbose:
                t.set_description(
                    f"loss={loss(params):.2f} train error={mean_error:.2f} val error={val_error:.2f}"
                )
    return params


def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [
        random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


def serialize_network_params(params):
    return jnp.concatenate([jnp.concatenate([w.ravel(), b]) for w, b in params])


def deserialize_network_params(params, sizes):
    layers = []
    start = 0
    for i in range(1, len(sizes)):
        end = start + sizes[i] * sizes[i - 1]
        w = jnp.reshape(params[start:end], (sizes[i], sizes[i - 1]))
        start = end
        b = params[start : start + sizes[i]]
        layers.append((w, b))
        start += sizes[i]
    return layers


@jit
def mlp_predict(params, vectors):
    # per-example predictions
    activations = vectors.flatten()

    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = gelu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return sigmoid(logits)


batch_mlp_predict = vmap(mlp_predict, in_axes=(None, 0), out_axes=0)


def poincare_sphere_projection(coords, n_cols, n_dimensions):
    vec_coords = jnp.reshape(coords, (-1, n_dimensions))
    norms = jnp.linalg.norm(vec_coords, axis=-1)
    norms = jnp.maximum(norms, 1)
    vec_coords = vec_coords / norms[:, None] * 0.999
    return vec_coords.flatten()


def make_loss(args, n_cols, gt, weights=None):
    distance_computor = distance_computors[args.dist]

    @jit
    def loss(params):
        dists = distance_computor(params, n_cols, args.dims)
        # compare the obtained distances with the ground truth using mse
        return (
            (
                (jnp.where(jnp.isinf(gt), dists, gt) - dists) ** 2
                * (1 if weights is None else weights)
            ).sum()
            / (~jnp.isinf(gt)).sum()
        ).squeeze()

    if args.dist == "poincare":
        grad_loss = grad(loss)
        poincare_grad_loss = (
            lambda params: grad_loss(params)
            / 4
            * (1 - jnp.linalg.norm(params) ** 2) ** 2
        )
        return loss, poincare_grad_loss
    return loss, grad(loss)


def number_of_mlp_params(layers):
    return sum((m + 1) * n for m, n in zip(layers[:-1], layers[1:]))
