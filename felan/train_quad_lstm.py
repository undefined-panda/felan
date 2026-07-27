"""Trainings- und Eval-Pipeline für das LSTM-Black-Box-Modell.

Angelehnt an train_quad.py, aber mit den Anpassungen die ein sequenz-basiertes
Modell braucht:
  - Fenster-Sampler, der Run-Grenzen respektiert
  - Target = Residual-Torque des NÄCHSTEN Zeitschritts
  - Kein M/C/G-Decomposition-Eval (Black-Box), stattdessen reiner Torque-Vergleich

Aufruf (analog zu train_quad.py):
    python -m felan.train_quad_lstm --robot go2

Bugs die ich in deinem Ursprungs-Script gefixt habe:
  - z_dim / Output = nq (12 bei go2), nicht 6
  - Target y = tau_hat[start+T, :] (alle Joints), nicht [:6]
  - rng → key konsistent
  - Loss mit norm_tau normalisiert (fair über Gelenke)
  - valid_starts respektieren Run-Grenzen
"""

import os
import time
import pickle
import argparse
from functools import partial

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
from jax import random
import numpy as np

import optax
from flax.training import train_state
from tensorboardX import SummaryWriter

import matplotlib as mp
import matplotlib.pyplot as plt

if os.getenv("DISPLAY"):
    try:
        mp.use("Qt5Agg")
    except ImportError:
        pass

repo_dir = os.path.dirname(os.path.abspath(__file__))

from felan.data_scripts.jax_utils import init_env
from felan.train import save_model_fn, load_model_fn, count_parameters
from felan.models.lstm_black_box import LSTMBlackBox, get_config_from_dict as get_lstm_config


# ---------------------------------------------------------------------------
# Prepare data
# ---------------------------------------------------------------------------

def compute_residual_from_pkl(dataset_full_path, test_labels):
    """Read raw .pkl file and compute residual torque.
    """
    with open(dataset_full_path, 'rb') as f:
        data = pickle.load(f)

    # Robust nach Labels suchen
    labels_key = None
    for k in ('labels', 'run_labels', 'names', 'run_names'):
        if k in data:
            labels_key = k
            break
    if labels_key is None:
        # Fallback: labels aus Reihenfolge konstruieren
        n_runs = np.asarray(data['q']).shape[0]
        run_labels = [f'env_0_run_{i}' for i in range(n_runs)]
    else:
        run_labels = list(data[labels_key])

    q_all       = np.asarray(data['q'])                
    qd_all      = np.asarray(data['qd'])               
    tau_m       = np.asarray(data['tau_m'])          
    tau_dyn     = np.asarray(data['tau_dyn'])        
    tau_ext_tot = np.asarray(data['tau_ext_total'])  

    q_joints   = q_all[..., 7:]    
    qd_joints  = qd_all[..., 6:]     
    tau_res    = (tau_m - tau_dyn - tau_ext_tot)[..., 6:]

    # Train/Test Split based on test_labels
    test_mask  = np.array([lbl in test_labels for lbl in run_labels])
    train_mask = ~test_mask

    train_q,   test_q   = q_joints[train_mask],  q_joints[test_mask]
    train_qd,  test_qd  = qd_joints[train_mask], qd_joints[test_mask]
    train_tau, test_tau = tau_res[train_mask],   tau_res[test_mask]
    train_labels = [l for l, m in zip(run_labels, train_mask) if m]
    test_labels_out = [l for l, m in zip(run_labels, test_mask) if m]

    return (train_labels, train_q, train_qd, train_tau,
            test_labels_out, test_q, test_qd, test_tau)

def compute_residual_from_npz(dataset_full_path, train_size=0.75):
    """Read raw .npz file and compute residual torque.
    """
    data = np.load(dataset_full_path)
    
    num_datasets = data["time"].shape[0]
    test_run_indices = np.random.choice(np.arange(num_datasets), int(num_datasets * (1-train_size)), replace=False)

    joint_pos = data["joint_pos"]
    joint_vel = data["joint_vel"]
    residual_torque = (data["diff_tau_m_nom"] + data["diff_tau_c_nom"] + data["diff_tau_g_nom"])[..., :6]

    n_runs = joint_pos.shape[0]
    run_labels = [f'run_{i}' for i in range(n_runs)]

    # Train/Test-Split anhand Run-Indizes
    test_mask  = np.array([i in test_run_indices for i in range(n_runs)])
    train_mask = ~test_mask

    train_q,   test_q   = joint_pos[train_mask], joint_pos[test_mask]
    train_qd,  test_qd  = joint_vel[train_mask], joint_vel[test_mask]
    train_tau, test_tau = residual_torque[train_mask],   residual_torque[test_mask]

    train_labels = [run_labels[i] for i in range(n_runs) if train_mask[i]]
    test_labels  = [run_labels[i] for i in range(n_runs) if test_mask[i]]

    return (train_labels, train_q, train_qd, train_tau,
            test_labels,  test_q,  test_qd,  test_tau)


def concatenate_runs(q_runs, qd_runs, tau_runs):
    """(n_runs, T, D) -> (N, D) + divider-Array mit Run-Grenzen."""
    n_runs, T, _ = q_runs.shape
    divider = np.arange(n_runs + 1) * T          # [0, T, 2T, ...]
    q_flat   = q_runs.reshape(-1, q_runs.shape[-1])
    qd_flat  = qd_runs.reshape(-1, qd_runs.shape[-1])
    tau_flat = tau_runs.reshape(-1, tau_runs.shape[-1])
    return q_flat, qd_flat, tau_flat, divider


def build_valid_starts(divider, time_window):
    """Alle Startindizes s, für die [s, s+time_window] komplett innerhalb eines Runs liegt.

    time_window Zeitschritte gehen als Input rein, tau[s+time_window] ist das Target.
    Also brauchen wir s + time_window < run_end.
    """
    valid = []
    for i in range(len(divider) - 1):
        run_start, run_end = int(divider[i]), int(divider[i + 1])
        # s+time_window muss < run_end sein (echte Ungleichung, weil tau[s+T] gebraucht wird)
        for s in range(run_start, run_end - time_window):
            valid.append(s)
    return jnp.asarray(valid, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Sequenz-Sampler
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=["time_window", "batch_size"])
def sample_lstm_batch(key, q_flat, qd_flat, tau_flat, valid_starts,
                      time_window: int, batch_size: int):
    """Zieht batch_size Fenster von Länge time_window; y = tau bei time_window+1-tem Schritt."""
    k1, _ = random.split(key)
    idx = random.randint(k1, (batch_size,), 0, valid_starts.shape[0])
    starts = valid_starts[idx]

    def one_sample(start):
        q_win   = jax.lax.dynamic_slice_in_dim(q_flat,   start, time_window, axis=0)
        qd_win  = jax.lax.dynamic_slice_in_dim(qd_flat,  start, time_window, axis=0)
        tau_win = jax.lax.dynamic_slice_in_dim(tau_flat, start, time_window, axis=0)
        x = jnp.concatenate([q_win, qd_win, tau_win], axis=-1)   # (T, 3*nq)
        y = tau_flat[start + time_window]                        # (nq,)
        return x, y

    X, y = jax.vmap(one_sample)(starts)
    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=["apply_fn"])
def train_step(state, x_batch, y_batch, norm_tau, apply_fn):
    def loss_fn(params):
        tau_pred, _, _ = apply_fn(params, x_batch)
        print(tau_pred.shape, y_batch.shape)
        return jnp.mean(jnp.sum((tau_pred - y_batch) ** 2 / norm_tau, axis=-1))
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train_lstm(state, q_flat, qd_flat, tau_flat, valid_starts,
               norm_tau, key, time_window, batch_size, num_epochs,
               log_period=50, tb_writer=None):
    losses = []
    apply_fn = state.apply_fn
    t0 = time.perf_counter()
    for epoch in range(num_epochs):
        key, sk = random.split(key)
        x_batch, y_batch = sample_lstm_batch(sk, q_flat, qd_flat, tau_flat,
                                             valid_starts, time_window, batch_size)
        state, loss = train_step(state, x_batch, y_batch, norm_tau, apply_fn)
        losses.append(float(loss))
        if epoch % log_period == 0 and epoch > 0:
            recent = float(np.mean(losses[-log_period:]))
            print("Epoch {0:05d}: ".format(epoch), end=" ")
            print("Time = {0:05.1f}s".format(time.perf_counter() - t0), end=", ")
            print(f"Mean loss (last {log_period}) = {recent:.6f}")
            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", recent, epoch)
    print(f"Training total time: {time.perf_counter() - t0:.1f}s")
    return state, losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_lstm(state, q_flat, qd_flat, tau_flat, divider, time_window, norm_tau):
    """Rollt für jeden Test-Run alle gültigen Fenster durch und sammelt Predictions.

    Rückgabe:
        y_true, y_pred:  (N_eval, nq)   -- gepackt
        eval_divider:    Grenzen der Runs im gepackten Array (für Plot)
    """
    y_true_all, y_pred_all = [], []
    eval_divider = [0]

    for i in range(len(divider) - 1):
        run_start = int(divider[i])
        run_end   = int(divider[i + 1])
        n_win = run_end - run_start - time_window
        if n_win <= 0:
            eval_divider.append(eval_divider[-1])
            continue

        # Für alle Startpositionen dieses Runs die Fenster bauen (vektorisiert)
        starts = jnp.arange(n_win, dtype=jnp.int32) + run_start

        def one(start):
            q_win   = jax.lax.dynamic_slice_in_dim(q_flat,   start, time_window, axis=0)
            qd_win  = jax.lax.dynamic_slice_in_dim(qd_flat,  start, time_window, axis=0)
            tau_win = jax.lax.dynamic_slice_in_dim(tau_flat, start, time_window, axis=0)
            x = jnp.concatenate([q_win, qd_win, tau_win], axis=-1)
            y = tau_flat[start + time_window]
            return x, y

        # In Chunks durchgehen, um Speicher zu schonen
        chunk = 512
        pred_chunks = []
        true_chunks = []
        for c0 in range(0, n_win, chunk):
            c1 = min(c0 + chunk, n_win)
            X, Y = jax.vmap(one)(starts[c0:c1])
            tau_pred, _, _ = state.apply_fn(state.params, X)
            pred_chunks.append(np.asarray(tau_pred))
            true_chunks.append(np.asarray(Y))

        y_pred_all.append(np.concatenate(pred_chunks, axis=0))
        y_true_all.append(np.concatenate(true_chunks, axis=0))
        eval_divider.append(eval_divider[-1] + n_win)

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    # Metrik
    err = np.sum((y_pred - y_true) ** 2 / np.asarray(norm_tau), axis=1)
    print(f"\nTest MSE (norm.):  mean={err.mean():.3e}  std={err.std():.3e}")
    print(f"Test MSE (raw):    mean={np.mean((y_pred - y_true)**2):.3e}")

    return y_true, y_pred, np.asarray(eval_divider)


def plot_lstm_torques(y_true, y_pred, eval_divider, test_labels,
                      model_folder, model_name, repo_dir,
                      render=True, combined=False):
    """Compare torque prediction of LSTM with ground truth

    Args:
        combined: False -> plot for each joint
                  True  -> all joints in one plot
    """
    n_dof = y_true.shape[-1]
    ticks = (eval_divider[:-1] + eval_divider[1:]) / 2
    fig_dir = os.path.join(repo_dir, "figures", model_folder, model_name)
    os.makedirs(fig_dir, exist_ok=True)

    def _style_axis(ax, j, show_legend=False):
        ax.set_title(f"Joint {j}")
        ax.set_ylabel("Residual Torque [Nm]")
        ax.plot(y_true[:, j], color="k", label="Ground truth")
        ax.plot(y_pred[:, j], color="r", alpha=0.8, label="LSTM prediction")
        ax.set_xticks(ticks)
        ax.set_xticklabels(test_labels, rotation=0, fontsize=8)
        for d in eval_divider:
            ax.axvline(d, color="gray", linestyle="--", lw=0.5)
        ax.set_xlim(eval_divider[0], eval_divider[-1])
        if show_legend:
            ax.legend(loc="upper right", fontsize=8)

    if combined:
        # Ein Grid: n_cols=2 Spalten, so viele Zeilen wie nötig
        n_cols = 2
        n_rows = (n_dof + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(24.0 / 1.54, 3.0 * n_rows),
            dpi=100, squeeze=False,
        )
        fig.subplots_adjust(left=0.06, bottom=0.05, right=0.98, top=0.96,
                            wspace=0.2, hspace=0.5)
        for j in range(n_dof):
            r, c = divmod(j, n_cols)
            _style_axis(axes[r][c], j, show_legend=(j == 0))
        # überzählige Achsen ausblenden, falls n_dof ungerade
        for j in range(n_dof, n_rows * n_cols):
            r, c = divmod(j, n_cols)
            axes[r][c].axis("off")

        fig.savefig(os.path.join(fig_dir, "lstm_joints_all.png"), format="png")
        fig.savefig(os.path.join(fig_dir, "lstm_joints_all.pdf"), format="pdf")

    else:
        # Wie bisher: pro Gelenkpaar eine eigene Figure
        for i in range(0, n_dof, 2):
            fig = plt.figure(figsize=(24.0 / 1.54, 8.0 / 1.54), dpi=100)
            fig.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=0.93,
                                hspace=0.35)
            for row, j in enumerate([i, i + 1]):
                if j >= n_dof:
                    break
                ax = fig.add_subplot(2, 1, row + 1)
                _style_axis(ax, j, show_legend=(row == 0))

            fig.savefig(os.path.join(fig_dir, f"lstm_joints_{i}_{i + 1}.png"), format="png")
            fig.savefig(os.path.join(fig_dir, f"lstm_joints_{i}_{i + 1}.pdf"), format="pdf")

    if render:
        plt.show()

def plot_train_loss(losses, title="Training Loss", save_path=None):
    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, linestyle="-", linewidth=0.8, color="tab:blue", label="Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", nargs=1, type=int, default=[True])
    parser.add_argument("-i", nargs=1, type=int, default=[0])
    parser.add_argument("-s", nargs=1, type=int, default=[0])
    parser.add_argument("-r", nargs=1, type=int, default=[1])
    parser.add_argument("-l", nargs=1, type=int, default=[0])
    parser.add_argument("-m", nargs=1, type=int, default=[1])
    parser.add_argument("--robot", type=str, default="go2",
                        choices=["go2", "spot_real", "hyqreal2", "spot_arm_real", "aliengo"])
    parser.add_argument("--file_type", type=str, default="pkl",
                        choices=["npz", "pkl"])
    args = parser.parse_args()
    seed, cuda, render, load_model, save_model = init_env(args)

    robot_prefix = args.robot
    file_type = "."+args.file_type
    nn_id = "LSTM"

    # ------------------ Robot / Dataset ------------------
    if 'arm' in robot_prefix:
        n_arms, nq_arm = 1, 7
    else:
        n_arms, nq_arm = 0, 0
    n_legs, nq_leg, nq_torso = 4, 3, 0
    nq = nq_torso + n_legs * nq_leg + n_arms * nq_arm    # 12 für go2

    dataset_map = {
        'go2':          'go2_sim_n_runs_50_data_freq_100hz',
        'spot_real':    'spot_real_freq_100hz',
        'spot_arm_real':'spot_arm_real_freq_100hz',
        'hyqreal2':     'hyqreal2_real_freq_100hz',
        'aliengo':      'quad_mass_dataset_run6'
    }
    dataset_name = dataset_map[robot_prefix]
    dataset_full_path = os.path.join(repo_dir, 'data', 'datasets', dataset_name + file_type)

    test_labels_wanted = {
        'go2':           ['env_0_run_2', 'env_0_run_10', 'env_0_run_11', 'env_0_run_28', 'env_0_run_41'],
        'spot_real':     ['env_0_run_32', 'env_0_run_52'],
        'spot_arm_real': ['env_0_run_60', 'env_0_run_71'],
        'hyqreal2':      ['env_0_run_2', 'env_0_run_5', 'env_0_run_25'],
        'aliengo':       []
    }[robot_prefix]

    # ------------------ Daten laden + Residuum berechnen ------------------
    print("Loading dataset ...")
    if file_type.split(".")[-1] == "pkl":
        (train_labels, train_q, train_qd, train_tau,
        test_labels, test_q, test_qd, test_tau) = compute_residual_from_pkl(
            dataset_full_path, test_labels_wanted)

    elif file_type.split(".")[-1] == "npz":
        (train_labels, train_q, train_qd, train_tau,
            test_labels, test_q, test_qd, test_tau) = compute_residual_from_npz(dataset_full_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    print(f"  # train runs: {len(train_labels)}")
    print(f"  # test runs:  {len(test_labels)}")
    print(f"  run shape:    {train_q.shape[1:]}")

    train_q_flat, train_qd_flat, train_tau_flat, train_divider = concatenate_runs(
        train_q, train_qd, train_tau)
    test_q_flat, test_qd_flat, test_tau_flat, test_divider = concatenate_runs(
        test_q, test_qd, test_tau)

    train_q_flat   = jnp.asarray(train_q_flat)
    train_qd_flat  = jnp.asarray(train_qd_flat)
    train_tau_flat = jnp.asarray(train_tau_flat)
    test_q_flat    = jnp.asarray(test_q_flat)
    test_qd_flat   = jnp.asarray(test_qd_flat)
    test_tau_flat  = jnp.asarray(test_tau_flat)

    # ------------------ Hyperparameter ------------------
    hyper = {
        'lstm_hidden_size': 10,
        'lstm_num_layers':  5,
        'lstm_dropout':     0.0,
        'time_window':      20,
        'batch_size':       256,
        'learning_rate':    5e-4,
        'weight_decay':     1e-5,
        'max_epoch':        5000,
        'norm_tau_eps':     1e-2,
        'n_output':         6,
        # Robot
        'n_legs':   n_legs,
        'n_dof_leg': nq_leg,
        'n_arms':   n_arms,
        'n_dof_arm': nq_arm,
        'n_dof_torso': nq_torso,
    }

    time_window = hyper['time_window']
    batch_size  = hyper['batch_size']

    valid_starts_train = build_valid_starts(train_divider, time_window)
    print(f"  # valid training windows: {int(valid_starts_train.shape[0])}")

    # Normalisierung anhand Trainings-Varianz
    norm_tau = jnp.var(train_tau_flat, axis=0) + hyper['norm_tau_eps']
    print(f"  norm_tau: {norm_tau}")

    # ------------------ Modell aufsetzen ------------------
    rng = jax.random.PRNGKey(seed)
    lstm_cfg = get_lstm_config(hyper)
    model = LSTMBlackBox(n_dof=nq, config=lstm_cfg)   # n_dof nur der Kompatibilität wegen

    dummy = jnp.zeros((4, time_window, 30))
    rng, param_rng = jax.random.split(rng)
    params = model.init(param_rng, dummy)
    print(f"  # model params: {count_parameters(params)}")

    optimizer = optax.adamw(learning_rate=hyper['learning_rate'],
                            weight_decay=hyper['weight_decay'])
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer)

    # ------------------ Training ------------------
    model_folder = f"{robot_prefix}/{nn_id}"
    model_name   = f"epochs_{hyper['max_epoch']}_{dataset_name}_{seed}"
    folder_path  = os.path.join(repo_dir, "trained_models", model_folder)

    if load_model:
        eval_params, _ = load_model_fn(model_name, folder_path)
        state = state.replace(params=eval_params)
    else:
        tb_folder = os.path.join(repo_dir, "tensorboard", model_folder, model_name)
        tb_writer = SummaryWriter(log_dir=tb_folder)
        print("\n### Training LSTM ###")
        rng, train_rng = jax.random.split(rng)
        state, losses = train_lstm(
            state, train_q_flat, train_qd_flat, train_tau_flat,
            valid_starts_train, norm_tau, train_rng,
            time_window, batch_size, hyper['max_epoch'],
            log_period=50, tb_writer=tb_writer,
        )
        tb_writer.close()
        if save_model:
            save_model_fn(state.params, hyper, model_name, folder_path)

    # ------------------ Evaluation ------------------
    print("\n### Evaluating LSTM ###")
    y_true, y_pred, eval_div = evaluate_lstm(
        state, test_q_flat, test_qd_flat, test_tau_flat,
        test_divider, time_window, norm_tau,
    )

    # ------------------ Plot ------------------
    print("\n### Plotting ###")
    plot_train_loss(losses, save_path="train_loss.png")

    plot_lstm_torques(y_true, y_pred, eval_div, test_labels,
                      model_folder, model_name, repo_dir, render=render,
                      combined=True)

    print("\nDone.")
