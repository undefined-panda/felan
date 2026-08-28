"""
Validate a trained CaDeLaC / LogChol-CaDeLaC model's predicted inertia matrix
`M` and bias-force `qfrc_bias` (see DeLaNPotParam.__call__ in
models/log_chol_cadelac_pot_param.py) against the ground truth available in
the dataset:

  1. Mass recovery: extras["M"][:, 0, 0] (the top-left m*I3 block) vs. the
     true residual payload mass (base_mass - nominal_mass) per test run --
     the only inertial quantity with an exact ground truth in this dataset.
  2. Physical sanity: eigenvalues of M (must be > 0), predicted mass range,
     and temporal stability of the prediction within a run (the true payload
     does not change mid-run, so a good context encoder should converge to a
     roughly constant mass estimate).
  3. Per-run M/C/G torque-decomposition MSE -- the same decomposition
     eval_components() computes, just broken out per run/mass instead of
     aggregated, so you can see whether accuracy degrades for specific
     payloads.

Usage:
    python3 check_inertia_predictions.py --model_name <name-without-.pkl> [--robot aliengo] [--nn LogChol-CaDeLaC]
"""
import os
import argparse
import copy

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import numpy as np

import matplotlib as mp
mp.use("Agg")
import matplotlib.pyplot as plt

repo_dir = os.path.dirname(os.path.abspath(__file__))

from felan.train import load_model_fn
from felan.data_scripts.data_loaders import load_custom_dataset, estimate_nominal_base_mass
from felan.models.np_math_utils import get_euler_from_mj_quat
from felan.models.log_chol_cadelac_pot_param import CaDeLaCLogChol, get_config_from_dict as get_cadelac_log_chol_pp_config
from felan.models.cadelac_pot_param import CaDeLaC, get_config_from_dict as get_cadelac_pp_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Saved model filename without .pkl, e.g. epochs_3000_..._seed_0")
    parser.add_argument("--robot", type=str, default="aliengo")
    parser.add_argument("--nn", type=str, default="LogChol-CaDeLaC", choices=["CaDeLaC", "LogChol-CaDeLaC"])
    parser.add_argument("--input_values", type=str, default="joint", help="Must match what the model was trained with")
    parser.add_argument("--nominal_mass", type=float, default=None, help="Base mass without payload; recovered from the residual gravity torque if omitted")
    args = parser.parse_args()

    model_folder = f"{args.robot}/{args.nn}"
    folder_path = repo_dir + f"/trained_models/{model_folder}"
    params, hyper = load_model_fn(args.model_name, folder_path)

    dataset_full_path = repo_dir + '/' + hyper['dataset_path']
    time_window = hyper['time_window']
    hist_stride = hyper.get('history_stride', 1)
    hist_gap = hyper.get('history_gap', 0)
    print(f"Context: time_window={time_window} pts, stride={hist_stride}, gap={hist_gap} "
          f"({hyper.get('history_span_s', time_window / hyper.get('f_log', 1.0)):.3g}s span)")

    raw = np.load(dataset_full_path)
    if args.nominal_mass is not None:
        nominal_mass, mass_fit_slope = args.nominal_mass, float('nan')
    else:
        # NOT min(base_mass): the generator draws the payload as U(0, 5) kg, so even
        # the lightest run carries one (0.168 kg on run7) and every error reported
        # below would inherit that as a constant floor -- which is the same order as
        # the effect being measured. Recovered from the residual gravity torque.
        nominal_mass, mass_fit_slope = estimate_nominal_base_mass(raw)
    print(f"Nominal (payload-free) base mass = {nominal_mass:.3f} kg "
          f"{f'(gravity-torque fit, slope {mass_fit_slope:.4f}, must be ~1.0)' if args.nominal_mass is None else '(user-specified)'}")

    seed = 0  # all current models were trained with --seeds 0
    train_data, test_data, divider, dt_mean, test_base_mass = load_custom_dataset(
        dataset_full_path, hist_length=time_window, sample_offset=0, seed=seed, add_noise=False,
        hist_stride=hist_stride, hist_gap=hist_gap)

    (test_labels, test_qp, test_qv, test_qa, test_tau, test_m, test_c, test_g,
     test_hist_joint_pos, test_hist_joint_vel, test_hist_diff_tau,
     test_hist_base_orient, test_hist_base_vel, test_hist_base_ang_vel,
     test_hist_base_pos_z) = test_data

    if list(test_labels) != list(hyper['test_labels']):
        raise RuntimeError(
            "Reconstructed test split does not match the model's training split "
            f"(got {test_labels}, model was trained on {hyper['test_labels']}). "
            "The model was likely trained with a different --seed than 0."
        )

    # Base orientation: MuJoCo wxyz quaternion -> euler (same conversion as training)
    raw_test_qp = copy.deepcopy(test_qp)
    test_euler = get_euler_from_mj_quat(raw_test_qp[:, 3:7])
    test_qp = np.hstack((raw_test_qp[:, 0:3], test_euler, raw_test_qp[:, 7:]))

    # Feature order MUST mirror the match-block in train_quad_cadelac.py exactly,
    # otherwise the LSTM sees permuted channels and silently produces garbage.
    if args.input_values == "joint":
        test_history = np.concatenate([test_hist_joint_pos, test_hist_joint_vel, test_hist_diff_tau], axis=-1)
    elif args.input_values == "base_pos_z":
        test_history = np.concatenate([test_hist_base_orient, test_hist_base_vel, test_hist_base_ang_vel,
                                       test_hist_base_pos_z, test_hist_diff_tau], axis=-1)
    elif args.input_values == "base":
        test_history = np.concatenate([test_hist_base_orient, test_hist_base_vel, test_hist_base_ang_vel,
                                       test_hist_diff_tau], axis=-1)
    else:
        raise NotImplementedError(f"--input_values {args.input_values} not implemented in this script yet")

    # Guard against a silently wrong --input_values: the LSTM input kernel's row
    # count is the ground truth for the feature dimension the model was trained on.
    lstm_in_dim = int(params["params"]["lstm"]["lstm_layer_0"]["ii"]["kernel"].shape[0])
    if test_history.shape[-1] != lstm_in_dim:
        raise ValueError(
            f"--input_values {args.input_values} builds {test_history.shape[-1]}-dim features, "
            f"but the model was trained on {lstm_in_dim}-dim features "
            f"(joint=30, base_pos_z=17, base=16 -- pass the value the model name carries).")

    test_qp = jnp.asarray(test_qp, dtype=jnp.float32)
    test_qv = jnp.asarray(test_qv, dtype=jnp.float32)
    test_qa = jnp.asarray(test_qa, dtype=jnp.float32)
    test_tau = jnp.asarray(test_tau, dtype=jnp.float32)
    test_m = jnp.asarray(test_m, dtype=jnp.float32)
    test_c = jnp.asarray(test_c, dtype=jnp.float32)
    test_g = jnp.asarray(test_g, dtype=jnp.float32)
    test_history = jnp.asarray(test_history, dtype=jnp.float32)

    if args.nn == "LogChol-CaDeLaC":
        nn_config = get_cadelac_log_chol_pp_config(hyper)
        learned_model = CaDeLaCLogChol(hyper['nv_dof'], nn_config)
    else:
        nn_config = get_cadelac_pp_config(hyper)
        learned_model = CaDeLaC(hyper['nv_dof'], nn_config)

    print("\nRunning forward pass on the full test set (unjitted, may take ~30-60s)...")
    tau_pred, dEdt, extras = learned_model.apply(params, test_qp, test_qv, test_qa, test_history)

    if "M" not in extras:
        raise RuntimeError(f"Model of type --nn {args.nn} does not expose extras['M'] -- use --nn LogChol-CaDeLaC.")
    M = np.asarray(extras["M"])
    pred_mass = M[:, 0, 0]  # top-left block of M is m*I3 by construction (spatial_inertia_from_params)

    print("\n================ Physical sanity checks ================")
    eigvals = np.linalg.eigvalsh(M)
    n_negative_eig = int(np.sum(eigvals <= 0))
    print(f"Non-positive eigenvalues of predicted M: {n_negative_eig} / {eigvals.size} "
          f"({'OK' if n_negative_eig == 0 else 'VIOLATION -- should not happen by construction, check the log-Cholesky code path'})")
    print(f"Predicted residual mass range: [{pred_mass.min():.3f}, {pred_mass.max():.3f}] kg "
          f"(expected: roughly within [0, 5] kg -- the simulated payload range)")

    divider = np.array(divider)
    print("\n================ Per-run mass recovery ================")
    print(f"{'run':<10}{'true payload [kg]':>20}{'pred mean [kg]':>18}{'pred std [kg]':>16}{'error [kg]':>14}")
    fig, axes = plt.subplots(1, len(test_labels), figsize=(4 * len(test_labels), 3), sharey=True)
    if len(test_labels) == 1:
        axes = [axes]
    errors = []
    for i, label in enumerate(test_labels):
        s, e = divider[i], divider[i + 1]
        run_pred_mass = pred_mass[s:e]
        true_payload = test_base_mass[i] - nominal_mass
        err = float(run_pred_mass.mean() - true_payload)
        errors.append(err)
        print(f"{label:<10}{true_payload:>20.3f}{run_pred_mass.mean():>18.3f}{run_pred_mass.std():>16.3f}{err:>14.3f}")

        axes[i].plot(run_pred_mass, color="tab:blue", label="predicted")
        axes[i].axhline(true_payload, color="k", linestyle="--", label="true payload")
        axes[i].set_title(label)
        axes[i].set_xlabel("timestep")
    axes[0].set_ylabel("residual mass [kg]")
    axes[0].legend()
    fig.tight_layout()

    out_dir = repo_dir + f"/figures/{model_folder}/{args.model_name}"
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/mass_recovery_check.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved mass-recovery plot to {out_dir}/mass_recovery_check.png")

    errors = np.array(errors)
    print(f"\nMean |mass error| across test runs: {np.abs(errors).mean():.3f} kg "
          f"(RMSE: {np.sqrt((errors ** 2).mean()):.3f} kg)")

    # ---- Physical bound: the best payload estimate that exists on the window ----
    # mean(diff_tau_z)/g over the same span the model sees. A model that actually
    # does physics lands near this on both splits. Below it on train means the
    # information cannot have come from the window -- it came from run identity,
    # i.e. the model memorised. Far above it on test means it is not using the
    # window at all.
    span = time_window * hist_stride
    raw_base_mass = raw['base_mass'].mean(axis=1)
    raw_diff_tau_z = (raw['diff_tau_m_nom'] + raw['diff_tau_c_nom'] + raw['diff_tau_g_nom'])[..., 2]
    test_run_idx = [int(lbl.split('_')[-1]) for lbl in test_labels]
    train_run_idx = [i for i in range(raw_base_mass.shape[0]) if i not in test_run_idx]

    def window_bound(run_indices):
        errs = []
        for i in run_indices:
            win = np.lib.stride_tricks.sliding_window_view(raw_diff_tau_z[i], span)[:, ::hist_stride]
            est = win.mean(axis=1) / 9.81  # only the points the network actually sees
            errs.append(est - (raw_base_mass[i] - nominal_mass))
        e = np.concatenate(errs)
        return np.abs(e).mean(), np.sqrt((e ** 2).mean())

    bound_train_mae, bound_train_rmse = window_bound(train_run_idx)
    bound_test_mae, bound_test_rmse = window_bound(test_run_idx)
    print(f"\n================ Physical bound over the same {span} samples "
          f"({span / hyper.get('f_log', 200.0) * 1e3:.0f} ms) ================")
    print(f"  mean(diff_tau_z)/g   train MAE {bound_train_mae:.3f} kg (RMSE {bound_train_rmse:.3f})")
    print(f"                       test  MAE {bound_test_mae:.3f} kg (RMSE {bound_test_rmse:.3f})")
    net_test_mae = np.abs(errors).mean()
    if net_test_mae > 2.0 * bound_test_mae:
        print(f"  -> model test error {net_test_mae:.3f} kg is >2x the bound: it is not using the window.")
    elif net_test_mae <= bound_test_mae * 1.3:
        print(f"  -> model test error {net_test_mae:.3f} kg is at the bound: it is doing physics.")
    else:
        print(f"  -> model test error {net_test_mae:.3f} kg vs bound {bound_test_mae:.3f} kg.")
    print("  (compare the model's TRAIN mass error against the train bound: "
          "below it = memorisation, the window cannot supply that accuracy)")

    print("\n================ Per-run M / qfrc_bias torque-decomposition MSE ================")
    print("(same decomposition as eval_components' Inertial/Coriolis/Gravitational MSE, broken out per run)")
    norm_tau = jnp.var(test_tau, axis=0) + 1e-2
    qd_zeros = jnp.zeros_like(test_qv)
    eval_g, _, _ = learned_model.apply(params, test_qp, qd_zeros, qd_zeros, test_history)
    eval_c, _, _ = learned_model.apply(params, test_qp, test_qv, qd_zeros, test_history)
    eval_m, _, _ = learned_model.apply(params, test_qp, qd_zeros, test_qa, test_history)
    eval_c = eval_c - eval_g
    eval_m = eval_m - eval_g

    print(f"{'run':<10}{'M-MSE':>12}{'C-MSE':>12}{'G-MSE':>12}")
    for i, label in enumerate(test_labels):
        s, e = divider[i], divider[i + 1]
        mse_m = float(jnp.mean(jnp.sum((eval_m[s:e] - test_m[s:e]) ** 2 / norm_tau, axis=1)))
        mse_c = float(jnp.mean(jnp.sum((eval_c[s:e] - test_c[s:e]) ** 2 / norm_tau, axis=1)))
        mse_g = float(jnp.mean(jnp.sum((eval_g[s:e] - test_g[s:e]) ** 2 / norm_tau, axis=1)))
        print(f"{label:<10}{mse_m:>12.3f}{mse_c:>12.3f}{mse_g:>12.3f}")


if __name__ == "__main__":
    main()
