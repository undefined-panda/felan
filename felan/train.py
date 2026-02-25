from typing import Tuple, Iterable, Dict, Any, List

from flax import struct
import functools
from collections import defaultdict

import os
import pickle
import time

import jax
import jax.numpy as jnp
from flax.training import checkpoints
from flax.traverse_util import flatten_dict

def save_model_fn(params, model_name, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Folder created!")

    print(f'Saving model: {model_name}')
    with open(f"{folder_path}/{model_name}" + ".pkl", "wb") as f:
        pickle.dump(params, f)

def load_model_fn(model_name, folder_path):
    with open(f"{folder_path}/{model_name}" + ".pkl", "rb") as f:
        return pickle.load(f)

@struct.dataclass(frozen=True)
class TrainConfig:
    num_epochs: int
    loss_power: bool
    norm_tau: jnp.ndarray
    batch_size: int
    log_period: int
    hyper: Dict[str, Any] = struct.field(hash=False)
    repo_dir: str
    tb_folder: str
    save_checkpoint_model: bool


def create_dataset(raw_list: List[jnp.ndarray]):
    return tuple(jnp.array(var, dtype=jnp.float32) for var in raw_list)

def data_loader(
    dataset: Tuple[jnp.ndarray, ...],
    batch_size: int,
    shuffle: bool = True,
    rng: jax.random.PRNGKey = jax.random.PRNGKey(0)
) -> Iterable[Tuple[jnp.ndarray, ...]]:
    n_samples = dataset[0].shape[0]
    
    # Create shuffled or sequential indices
    if shuffle:
        perm = jax.random.permutation(rng, n_samples)
    else:
        perm = jnp.arange(n_samples)

    # Generate batches
    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        idx = perm[start:end]
        yield tuple(d[idx] for d in dataset)

def count_parameters(params):
    return sum(jnp.size(p) for p in jax.tree_util.tree_leaves(params))
    
# Loss 
def loss_fn(state, params, batch_data, config: TrainConfig):
    q, qd, qdd, tau = batch_data
    tau_hat, dEdt_hat, extras = state.apply_fn(params, q, qd, qdd)

    # Compute the loss of the Euler-Lagrange Differential Equation:
    err_inv = jnp.sum((tau_hat - tau) ** 2 / config.norm_tau, axis=1)
    l_mean_inv_dyn = jnp.mean(err_inv)
    l_var_inv_dyn = jnp.var(err_inv)
    l_mean_squared_inv_dyn = jnp.mean(err_inv ** 2)

    # Compute the loss of the Power Conservation:
    dEdt = jnp.sum(qd * tau, axis=1)
    err_dEdt = (dEdt_hat - dEdt) ** 2
    l_mean_dEdt = jnp.mean(err_dEdt)
    l_mean_squared_dEdt = jnp.mean(err_dEdt ** 2)
    l_var_dEdt = jnp.var(err_dEdt)

    loss = l_mean_inv_dyn + (l_mean_dEdt if config.loss_power else 0.0)

    metrics = {
        'l_mean_inv_dyn': l_mean_inv_dyn,
        'l_var_inv_dyn': l_var_inv_dyn,
        'l_mean_squared_inv_dyn': l_mean_squared_inv_dyn,
        'l_mean_dEdt': l_mean_dEdt,
        'l_var_dEdt': l_var_dEdt,
        'l_mean_squared_dEdt': l_mean_squared_dEdt,
    }


    if config.hyper['penalize_extra']:
        # Compute the loss of mass:
        diff_mass = extras['diff_mass']
        err_mass = diff_mass
        l_mean_mass = jnp.mean(err_mass)
        l_var_mass = jnp.var(err_mass)

        loss += config.hyper['penalize_diff_mass'] * l_mean_mass

        metrics.update({
            'l_mean_mass': l_mean_mass,
            'l_var_mass': l_var_mass,
        })

        # Compute the loss of rot eigenvalue
        diff_rot_eigenvalue = extras['diff_rot_eigenvalue']
        err_rot_eigenvalue = diff_rot_eigenvalue ** 2

        l_mean_rot_eigenvalue = jnp.mean(err_rot_eigenvalue)
        l_var_rot_eigenvalue = jnp.var(err_rot_eigenvalue)

        loss += config.hyper['penalize_rot_eigenvalue'] * l_mean_rot_eigenvalue

        metrics.update({
            'l_mean_rot_eigenvalue': l_mean_rot_eigenvalue,
            'l_var_rot_eigenvalue': l_var_rot_eigenvalue,
        })

    # add mean and variance for everything in extras
    for key, value in extras.items():
        # reduce over batch dimension (assume axis=0 is batch)
        metrics[f"{key}_mean"] = jnp.mean(value)
        metrics[f"{key}_var"] = jnp.var(value)
        metrics[f"{key}_max"] = jnp.max(value)

    return loss, metrics

def train_step(state, batch, config: TrainConfig):
    # Gradient function
    grad_fn = jax.value_and_grad(loss_fn,
                                 argnums=1,
                                 has_aux=True
                                )
    # Determine gradients for current model, parameters and batch
    (loss, metrics), grads = grad_fn(state, state.params, batch, config)
    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)

    grad_norm = jnp.sqrt(sum([jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)]))
    metrics.update({'grad_norm': grad_norm})

    # Parameter Gradient Norm
    flat_grads = flatten_dict(grads)
    for path, g in flat_grads.items():
        name = "/".join(path)
        param_norm = jnp.linalg.norm(g)
        metrics[f"grad_norm/{name}"] = param_norm

    return state, loss, metrics

def train_model(state, dataset, rng, config: TrainConfig, tb_writer = None):
    # Start Training Loop:
    t0_start = time.perf_counter()
    t_avg_epoch = 0.0

    train_step_fn = functools.partial(train_step, config=config)
    jit_train_step = jax.jit(train_step_fn)

    # Training loop
    for epoch in range(config.num_epochs):
        t0_epoch = time.perf_counter()
        n_batches, epoch_loss = 0.0, 0.0
        epoch_mean_inv_dyn, epoch_var_inv_dyn = 0.0, 0.0
        epoch_mean_dEdt, epoch_var_dEdt = 0.0, 0.0
        epoch_mean_diff_mass, epoch_var_diff_mass = 0.0, 0.0
        epoch_mean_rot_eigenvalue, epoch_var_rot_eigenvalue = 0.0, 0.0
        epoch_mean_squared_inv_dyn, epoch_mean_squared_dEdt = 0.0, 0.0

        rng, data_rng = jax.random.split(rng, num=2)
        data_batches = data_loader(dataset, batch_size=config.batch_size, shuffle=True, rng=data_rng)

        epoch_sums = defaultdict(float)

        for batch_id, batch in enumerate(data_batches):
            state, loss, metrics = jit_train_step(state, batch)

            n_batches += 1
            epoch_loss += loss
            epoch_mean_inv_dyn += metrics['l_mean_inv_dyn']
            epoch_var_inv_dyn += metrics['l_var_inv_dyn']
            epoch_mean_squared_inv_dyn += metrics['l_mean_squared_inv_dyn']
            epoch_mean_dEdt += metrics['l_mean_dEdt']
            epoch_var_dEdt += metrics['l_var_dEdt']
            epoch_mean_squared_dEdt += metrics['l_mean_squared_dEdt']

            if 'l_mean_mass' in metrics:
                epoch_mean_diff_mass += metrics['l_mean_mass']
                epoch_var_diff_mass += metrics['l_var_mass']

            if 'l_mean_rot_eigenvalue' in metrics:
                epoch_mean_rot_eigenvalue += metrics['l_mean_rot_eigenvalue']
                epoch_var_rot_eigenvalue += metrics['l_var_rot_eigenvalue']

            # accumulate all metrics automatically
            for k, v in metrics.items():
                epoch_sums[k] += v

        # normalize at the end
        epoch_metrics = {}
        for k, v in epoch_sums.items():
            # Normalize all values
            avg_value = v / n_batches
            # If key ends with '_max', take the max instead
            if k.endswith('_max'):
                epoch_metrics[k] = jnp.max(v)  # assumes v is array-like
            else:
                epoch_metrics[k] = avg_value

        t_epoch = time.perf_counter() - t0_epoch
        t_avg_epoch = t_avg_epoch + t_epoch
        epoch_loss /= n_batches
        epoch_mean_inv_dyn /= n_batches
        epoch_var_inv_dyn /= n_batches
        epoch_mean_dEdt /= n_batches
        epoch_var_dEdt /= n_batches
        epoch_mean_diff_mass /= n_batches
        epoch_var_diff_mass /= n_batches
        epoch_mean_rot_eigenvalue /= n_batches
        epoch_var_rot_eigenvalue /= n_batches
        epoch_mean_squared_inv_dyn /= n_batches
        epoch_mean_squared_dEdt /= n_batches

        epoch_inv_dyn_std = jnp.sqrt(epoch_mean_squared_inv_dyn - epoch_mean_inv_dyn ** 2)
        epoch_dEdt_std = jnp.sqrt(epoch_mean_squared_dEdt - epoch_mean_dEdt ** 2)

        if epoch % config.log_period == 0 or epoch == (config.num_epochs - 1):
            print("Epoch {0:05d}: ".format(epoch), end=" ")
            print("Time = {0:05.1f}s".format(time.perf_counter() - t0_start), end=", ")
            print("Time/Epoch = {0:05.4f}s".format(t_avg_epoch / config.log_period), end=", ")
            print("Loss = {0:.3e}".format(epoch_loss), end=", ")
            print("Inv Dyn = {0:.3e} \u00B1 {1:.3e} Var {2:.3e}".format(epoch_mean_inv_dyn, 1.96 * epoch_inv_dyn_std, 1.96 * jnp.sqrt(epoch_var_inv_dyn)), end=", ")
            print("Power Con = {0:.3e} \u00B1 {1:.3e} Var {2:.3e}".format(epoch_mean_dEdt, 1.96 * epoch_dEdt_std, 1.96 * jnp.sqrt(epoch_var_dEdt)))

            # Log Tensorboard Data
            if tb_writer is not None:
                tb_writer.add_scalar("loss", epoch_loss, epoch)
                tb_writer.add_scalar("loss_inv_dyn/mean", epoch_mean_inv_dyn, epoch)
                tb_writer.add_scalar("loss_inv_dyn/var_mean",  1.96 * jnp.sqrt(epoch_var_inv_dyn), epoch)
                tb_writer.add_scalar("loss_inv_dyn/var",  1.96 * epoch_inv_dyn_std, epoch)
                tb_writer.add_scalar("loss_power_con/mean", epoch_mean_dEdt, epoch)
                tb_writer.add_scalar("loss_power_con/var_mean", 1.96 * jnp.sqrt(epoch_var_dEdt), epoch)
                tb_writer.add_scalar("loss_power_con/var", 1.96 * epoch_dEdt_std, epoch)
                tb_writer.add_scalar("loss_mass/mean", epoch_mean_diff_mass, epoch)
                tb_writer.add_scalar("loss_mass/var", 1.96 * jnp.sqrt(epoch_var_diff_mass), epoch)
                tb_writer.add_scalar("loss_rot_eigenvalue/mean", epoch_mean_rot_eigenvalue, epoch)
                tb_writer.add_scalar("loss_rot_eigenvalue/var", 1.96 * jnp.sqrt(epoch_var_rot_eigenvalue), epoch)

                for k, v in epoch_metrics.items():
                    tb_writer.add_scalar(f"train/{k}", v, epoch)

                if config.save_checkpoint_model:
                    checkpoints.save_checkpoint(
                        ckpt_dir = f"{config.repo_dir}/{config.tb_folder}",
                        target = {"params": state.params, "hyper": config.hyper},
                        step = epoch,
                        prefix = "model_",
                        overwrite = False,
                    )
            t_avg_epoch = 0.0

        metrics = {
            'loss/total': epoch_loss,
            'loss/inv_dyn/mean': epoch_mean_inv_dyn,
            'loss/inv_dyn/var_mean': jnp.sqrt(epoch_var_inv_dyn),
            'loss/inv_dyn/var': epoch_inv_dyn_std,
            'loss/power_con/mean': epoch_mean_dEdt,
            'loss/power_con/var_mean': jnp.sqrt(epoch_var_dEdt),
            'loss/power_con/var': epoch_dEdt_std,
            'loss/mass/mean': epoch_mean_diff_mass,
            'loss/mass/var': jnp.sqrt(epoch_var_diff_mass),
            'loss/rot_eigenvalue/mean': epoch_mean_rot_eigenvalue,
            'loss/rot_eigenvalue/var': jnp.sqrt(epoch_var_rot_eigenvalue),
        }

    return state, metrics