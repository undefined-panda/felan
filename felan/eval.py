import os
import time

import matplotlib as mp
import matplotlib.pyplot as plt

import jax.numpy as jnp
from felan.train import create_dataset, data_loader

def eval_components(model, eval_params, eval_dataset, norm_tau = None):
    print("\n################################################")
    print("Model Evaluation:")

    data_batches = data_loader(eval_dataset, batch_size=1000, shuffle=False)
    t0_eval = time.perf_counter()

    # Init Variables
    eval_tau_all, eval_m_all, eval_c_all, eval_g_all = [], [], [], []
    tau_all, tau_m_all, tau_c_all, tau_g_all = [], [], [], []
    eval_dEdt_all, dEdt_all = [], []
    q_all, qd_all, qdd_all = [], [], []

    for batch in data_batches:
        q, qd, qdd, tau, tau_m, tau_c, tau_g = batch

        qd_zeros = jnp.zeros_like(qd)
        if norm_tau is None:
            norm_tau = jnp.ones_like(tau)

        # Evaluate Components
        eval_g, _, _ = model.apply(eval_params, q, qd_zeros, qd_zeros)
        eval_c, _, _ = model.apply(eval_params, q, qd, qd_zeros)
        eval_m, _, _ = model.apply(eval_params, q, qd_zeros, qdd)
        eval_c = eval_c - eval_g
        eval_m = eval_m - eval_g
        eval_tau, eval_dEdt, _ = model.apply(eval_params, q, qd, qdd)
        test_dEdt = jnp.sum(tau * qd, axis=1)

        # Collect outputs
        eval_tau_all.append(eval_tau)
        eval_m_all.append(eval_m)
        eval_c_all.append(eval_c)
        eval_g_all.append(eval_g)
        eval_dEdt_all.append(eval_dEdt)
        tau_all.append(tau)
        tau_m_all.append(tau_m)
        tau_c_all.append(tau_c)
        tau_g_all.append(tau_g)
        dEdt_all.append(test_dEdt)
        q_all.append(q)
        qd_all.append(qd)
        qdd_all.append(qdd)

    # Concatenate full arrays
    q = jnp.concatenate(q_all, axis=0)
    qd = jnp.concatenate(qd_all, axis=0)
    qdd = jnp.concatenate(qdd_all, axis=0)
    eval_tau = jnp.concatenate(eval_tau_all, axis=0)
    eval_m = jnp.concatenate(eval_m_all, axis=0)
    eval_c = jnp.concatenate(eval_c_all, axis=0)
    eval_g = jnp.concatenate(eval_g_all, axis=0)
    eval_dEdt = jnp.concatenate(eval_dEdt_all, axis=0)
    tau_all = jnp.concatenate(tau_all, axis=0)
    tau_m_all = jnp.concatenate(tau_m_all, axis=0)
    tau_c_all = jnp.concatenate(tau_c_all, axis=0)
    tau_g_all = jnp.concatenate(tau_g_all, axis=0)
    dEdt_all = jnp.concatenate(dEdt_all, axis=0)

    t_eval = time.perf_counter() - t0_eval

    # Sum Errors
    n_eval_samples = float(eval_dataset[0].shape[0])
    err_tau = jnp.sum((eval_tau - tau_all) ** 2 / norm_tau, axis=1)
    err_m = jnp.sum((eval_m - tau_m_all) ** 2 / norm_tau, axis=1)
    err_c = jnp.sum((eval_c - tau_c_all) ** 2 / norm_tau, axis=1)
    err_g = jnp.sum((eval_g - tau_g_all) ** 2 / norm_tau, axis=1)
    err_cg = jnp.sum((eval_c + eval_g - tau_c_all - tau_g_all) ** 2 / norm_tau, axis=1)
    err_dEdt = (eval_dEdt - dEdt_all) ** 2

    err_tau_mean = jnp.mean(err_tau)
    err_tau_std  = jnp.std(err_tau)
    err_m_mean   = jnp.mean(err_m)
    err_m_std    = jnp.std(err_m)
    err_c_mean   = jnp.mean(err_c)
    err_c_std    = jnp.std(err_c)
    err_g_mean   = jnp.mean(err_g)
    err_g_std    = jnp.std(err_g)
    err_cg_mean  = jnp.mean(err_cg)
    err_cg_std   = jnp.std(err_cg)
    err_dEdt_mean = jnp.mean(err_dEdt)
    err_dEdt_std  = jnp.std(err_dEdt)


    print("\nPerformance:")
    print("                Torque MSE = {0:.3e} \u00B1 {1:.3e}".format(err_tau_mean, 1.96 * err_tau_std))
    print("              Inertial MSE = {0:.3e} \u00B1 {1:.3e}".format(err_m_mean, 1.96 * err_m_std))
    print("Coriolis & Centrifugal MSE = {0:.3e} \u00B1 {1:.3e}".format(err_c_mean, 1.96 * err_c_std))
    print("         Gravitational MSE = {0:.3e} \u00B1 {1:.3e}".format(err_g_mean, 1.96 * err_g_std))
    print("      Cor & Cen & Grav MSE = {0:.3e} \u00B1 {1:.3e}".format(err_cg_mean, 1.96 * err_cg_std))
    print("    Power Conservation MSE = {0:.3e} \u00B1 {1:.3e}".format(err_dEdt_mean, 1.96 * err_dEdt_std))
    print("      Comp Time per Sample = {0:.3e}s / {1:.1f}Hz".format(t_eval / n_eval_samples, n_eval_samples/t_eval))

    metrics = {
        "eval/tau/mean": err_tau_mean,
        "eval/tau/std": err_tau_std,
        "eval/tau_m/mean": err_m_mean,
        "eval/tau_m/std": err_m_std,
        "eval/tau_c/mean": err_c_mean,
        "eval/tau_c/std": err_c_std,
        "eval/tau_g/mean": err_g_mean,
        "eval/tau_g/std": err_g_std,
        "eval/tau_cg/mean": err_cg_mean,
        "eval/tau_cg/std": err_cg_std,
        "eval/dEdt/mean": err_dEdt_mean,
        "eval/dEdt/std": err_dEdt_std,
    }

    return create_dataset([q, qd, qdd, eval_tau, eval_m, eval_c, eval_g]), metrics

def plot_components(eval_results, eval_dataset, test_labels, divider, model_type_folder, model_name, render = True, force_index = [], repo_dir = ''):
    q, qd, qdd, test_tau, test_m, test_c, test_g = eval_dataset
    _, _, _, eval_tau, eval_m, eval_c, eval_g = eval_results
    n_dof = test_tau.shape[-1]

    print("\n################################################")
    print("Plotting Performance:")

    # Alpha of the graphs:
    plot_alpha = 0.8

    # Plot the performance:
    y_t_low = jnp.clip(1.2 * jnp.min(jnp.vstack((test_tau, eval_tau)), axis=0), -jnp.inf, -0.01)
    y_t_max = jnp.clip(1.5 * jnp.max(jnp.vstack((test_tau, eval_tau)), axis=0), 0.01, jnp.inf)

    y_m_low = jnp.clip(1.2 * jnp.min(jnp.vstack((test_m, eval_m)), axis=0), -jnp.inf, -0.01)
    y_m_max = jnp.clip(1.2 * jnp.max(jnp.vstack((test_m, eval_m)), axis=0), 0.01, jnp.inf)

    y_c_low = jnp.clip(1.2 * jnp.min(jnp.vstack((test_c, eval_c)), axis=0), -jnp.inf, -0.01)
    y_c_max = jnp.clip(1.2 * jnp.max(jnp.vstack((test_c, eval_c)), axis=0), 0.01, jnp.inf)

    y_g_low = jnp.clip(1.2 * jnp.min(jnp.vstack((test_g, eval_g)), axis=0), -jnp.inf, -0.01)
    y_g_max = jnp.clip(1.2 * jnp.max(jnp.vstack((test_g, eval_g)), axis=0), 0.01, jnp.inf)
    if n_dof % 2 == 1:
        y_t_low = jnp.concatenate((y_t_low, -10*jnp.ones(1)))
        y_t_max = jnp.concatenate((y_t_max, 10*jnp.ones(1)))
        y_m_low = jnp.concatenate((y_m_low, -10*jnp.ones(1)))
        y_m_max = jnp.concatenate((y_m_max, 10*jnp.ones(1)))
        y_c_low = jnp.concatenate((y_c_low, -10*jnp.ones(1)))
        y_c_max = jnp.concatenate((y_c_max, 10*jnp.ones(1)))
        y_g_low = jnp.concatenate((y_g_low, -10*jnp.ones(1)))
        y_g_max = jnp.concatenate((y_g_max, 10*jnp.ones(1)))

    plt.rc('text', usetex=True)
    color_i = ["r", "b", "g", "k"]

    ticks = jnp.array(divider)
    ticks = (ticks[:-1] + ticks[1:]) / 2

    for i in range(0, n_dof, 2):

        fig = plt.figure(figsize=(24.0/1.54, 8.0/1.54), dpi=100)
        fig.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=0.95, wspace=0.3, hspace=0.2)
        # fig.canvas.manage.set_window_title('Seed = {0}'.format(seed))

        legend = [mp.patches.Patch(color=color_i[0], label="Model"),
                mp.patches.Patch(color="k", label="Ground Truth")]

        # Plot Torque
        ax0 = fig.add_subplot(2, 4, 1)
        # ax0.set_title(r"$\boldsymbol{\tau}$")
        ax0.set_title('Torque')
        ax0.text(s=f'Joint {i}', x=-0.35, y=.5, fontsize=12, fontweight="bold", rotation=90, horizontalalignment="center", verticalalignment="center", transform=ax0.transAxes)
        if i in force_index:
            ax0.set_ylabel("Force [N]")
        else:
            ax0.set_ylabel("Torque [Nm]")
        ax0.get_yaxis().set_label_coords(-0.2, 0.5)
        ax0.set_ylim(y_t_low[i+0], y_t_max[i+0])
        ax0.set_xticks(ticks)
        ax0.set_xticklabels(test_labels)
        ax0.vlines(divider, y_t_low[i+0], y_t_max[i+0], linestyles='--', lw=0.5, alpha=1.)
        ax0.set_xlim(divider[0], divider[-1])

        ax1 = fig.add_subplot(2, 4, 5)
        ax1.text(s=f'Joint {i+1}', x=-.35, y=0.5, fontsize=12, fontweight="bold", rotation=90,
                horizontalalignment="center", verticalalignment="center", transform=ax1.transAxes)

        ax1.text(s=r"\textbf{(a)}", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
                verticalalignment="center", transform=ax1.transAxes)

        if i in force_index:
            ax1.set_ylabel("Force [N]")
        else:
            ax1.set_ylabel("Torque [Nm]")
        ax1.get_yaxis().set_label_coords(-0.2, 0.5)
        ax1.set_ylim(y_t_low[i+1], y_t_max[i+1])
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(test_labels)
        ax1.vlines(divider, y_t_low[i+1], y_t_max[i+1], linestyles='--', lw=0.5, alpha=1.)
        ax1.set_xlim(divider[0], divider[-1])

        ax0.legend(handles=legend, bbox_to_anchor=(0.0, 1.0), loc='upper left', ncol=1, framealpha=1.)

        # Plot Ground Truth Torque:
        ax0.plot(test_tau[:, i+0], color="k")
        if i+1 < n_dof:
            ax1.plot(test_tau[:, i+1], color="k")

        # Plot Torque:
        ax0.plot(eval_tau[:, i+0], color=color_i[0], alpha=plot_alpha)
        if i+1 < n_dof:
            ax1.plot(eval_tau[:, i+1], color=color_i[0], alpha=plot_alpha)

        # Plot Mass Torque
        ax0 = fig.add_subplot(2, 4, 2)
        ax0.set_title(r"$\displaystyle\mathbf{H}(\mathbf{q}) \ddot{\mathbf{q}}$")
        if i in force_index:
            ax0.set_ylabel("Force [N]")
        else:
            ax0.set_ylabel("Torque [Nm]")
        ax0.set_ylim(y_m_low[i+0], y_m_max[i+0])
        ax0.set_xticks(ticks)
        ax0.set_xticklabels(test_labels)
        ax0.vlines(divider, y_m_low[i+0], y_m_max[i+0], linestyles='--', lw=0.5, alpha=1.)
        ax0.set_xlim(divider[0], divider[-1])

        ax1 = fig.add_subplot(2, 4, 6)
        ax1.text(s=r"\textbf{(b)}", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
                verticalalignment="center", transform=ax1.transAxes)

        if i in force_index:
            ax1.set_ylabel("Force [N]")
        else:
            ax1.set_ylabel("Torque [Nm]")
        ax1.set_ylim(y_m_low[i+1], y_m_max[i+1])
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(test_labels)
        ax1.vlines(divider, y_m_low[i+1], y_m_max[i+1], linestyles='--', lw=0.5, alpha=1.)
        ax1.set_xlim(divider[0], divider[-1])

        # Plot Ground Truth Inertial Torque:
        ax0.plot(test_m[:, i+0], color="k")
        if i+1 < n_dof:
            ax1.plot(test_m[:, i+1], color="k")

        # Plot Inertial Torque:
        ax0.plot(eval_m[:, i+0], color=color_i[0], alpha=plot_alpha)
        if i+1 < n_dof:
            ax1.plot(eval_m[:, i+1], color=color_i[0], alpha=plot_alpha)

        # Plot Coriolis Torque
        ax0 = fig.add_subplot(2, 4, 3)
        ax0.set_title(r"$\displaystyle\mathbf{c}(\mathbf{q}, \dot{\mathbf{q}})$")
        if i in force_index:
            ax0.set_ylabel("Force [N]")
        else:
            ax0.set_ylabel("Torque [Nm]")
        ax0.set_ylim(y_c_low[i+0], y_c_max[i+0])
        ax0.set_xticks(ticks)
        ax0.set_xticklabels(test_labels)
        ax0.vlines(divider, y_c_low[i+0], y_c_max[i+0], linestyles='--', lw=0.5, alpha=1.)
        ax0.set_xlim(divider[0], divider[-1])

        ax1 = fig.add_subplot(2, 4, 7)
        ax1.text(s=r"\textbf{(c)}", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
                verticalalignment="center", transform=ax1.transAxes)

        if i in force_index:
            ax1.set_ylabel("Force [N]")
        else:
            ax1.set_ylabel("Torque [Nm]")
        ax1.set_ylim(y_c_low[i+1], y_c_max[i+1])
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(test_labels)
        ax1.vlines(divider, y_c_low[i+1], y_c_max[i+1], linestyles='--', lw=0.5, alpha=1.)
        ax1.set_xlim(divider[0], divider[-1])

        # Plot Ground Truth Coriolis Torque:
        ax0.plot(test_c[:, i+0], color="k")
        if i+1 < n_dof:
            ax1.plot(test_c[:, i+1], color="k")

        # Plot Coriolis Torque:
        ax0.plot(eval_c[:, i+0], color=color_i[0], alpha=plot_alpha)
        if i+1 < n_dof:
            ax1.plot(eval_c[:, i+1], color=color_i[0], alpha=plot_alpha)

        # Plot Gravity
        ax0 = fig.add_subplot(2, 4, 4)
        ax0.set_title(r"$\displaystyle\mathbf{g}(\mathbf{q})$")
        if i in force_index:
            ax0.set_ylabel("Force [N]")
        else:
            ax0.set_ylabel("Torque [Nm]")
        ax0.set_ylim(y_g_low[i+0], y_g_max[i+0])
        ax0.set_xticks(ticks)
        ax0.set_xticklabels(test_labels)
        ax0.vlines(divider, y_g_low[i+0], y_g_max[i+0], linestyles='--', lw=0.5, alpha=1.)
        ax0.set_xlim(divider[0], divider[-1])

        ax1 = fig.add_subplot(2, 4, 8)
        ax1.text(s=r"\textbf{(d)}", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
                verticalalignment="center", transform=ax1.transAxes)

        if i in force_index:
            ax1.set_ylabel("Force [N]")
        else:
            ax1.set_ylabel("Torque [Nm]")
        ax1.set_ylim(y_g_low[i+1], y_g_max[i+1])
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(test_labels)
        ax1.vlines(divider, y_g_low[i+1], y_g_max[i+1], linestyles='--', lw=0.5, alpha=1.)
        ax1.set_xlim(divider[0], divider[-1])

        # Plot Ground Truth Gravity Torque:
        ax0.plot(test_g[:, i+0], color="k")
        if i+1 < n_dof:
            ax1.plot(test_g[:, i+1], color="k")

        # Plot Gravity Torque:
        ax0.plot(eval_g[:, i+0], color=color_i[0], alpha=plot_alpha)
        if i+1 < n_dof:
            ax1.plot(eval_g[:, i+1], color=color_i[0], alpha=plot_alpha)

        fig_dir = repo_dir + f"/figures/{model_type_folder}/{model_name}"
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
        fig.savefig(f"{fig_dir}/joints_{i}_{i+1}.pdf", format="pdf")
        fig.savefig(f"{fig_dir}/joints_{i}_{i+1}.png", format="png")

    if render:
        plt.show()

    print("\n################################################\n\n\n")


def plot_torques(eval_results, eval_dataset, test_labels, divider, model_type_folder, model_name, render = True, force_index = [], norm_tau = None, repo_dir = ''):
    q, qd, qdd, test_tau, test_m, test_c, test_g = eval_dataset
    _, _, _, eval_tau, eval_m, eval_c, eval_g = eval_results
    n_dof = test_tau.shape[-1]

    if norm_tau is None:
        norm_tau = jnp.ones_like(test_tau)

    n_eval_samples =  float(q.shape[0])
    nom_model_tau = test_g
    err_model_tau = 1. / n_eval_samples * jnp.sum((test_tau - nom_model_tau) ** 2 / norm_tau)
    print("      Nominal Model Torque MSE = {0:.3e}".format(err_model_tau))

    print("\n################################################")
    print("Plotting Performance:")

    # Alpha of the graphs:
    plot_alpha = 0.8

    # Plot the performance:
    y_t_low = jnp.clip(1.2 * jnp.min(jnp.vstack((test_tau, eval_tau)), axis=0), -jnp.inf, -0.01)
    y_t_max = jnp.clip(1.5 * jnp.max(jnp.vstack((test_tau, eval_tau)), axis=0), 0.01, jnp.inf)

    y_m_low = jnp.clip(1.2 * jnp.min(jnp.vstack((test_m, eval_m)), axis=0), -jnp.inf, -0.01)
    y_m_max = jnp.clip(1.2 * jnp.max(jnp.vstack((test_m, eval_m)), axis=0), 0.01, jnp.inf)

    y_c_low = jnp.clip(1.2 * jnp.min(jnp.vstack((test_c, eval_c)), axis=0), -jnp.inf, -0.01)
    y_c_max = jnp.clip(1.2 * jnp.max(jnp.vstack((test_c, eval_c)), axis=0), 0.01, jnp.inf)

    y_g_low = jnp.clip(1.2 * jnp.min(jnp.vstack((test_g, eval_g)), axis=0), -jnp.inf, -0.01)
    y_g_max = jnp.clip(1.2 * jnp.max(jnp.vstack((test_g, eval_g)), axis=0), 0.01, jnp.inf)
    if n_dof % 2 == 1:
        y_t_low = jnp.concatenate((y_t_low, -10*jnp.ones(1)))
        y_t_max = jnp.concatenate((y_t_max, 10*jnp.ones(1)))
        y_m_low = jnp.concatenate((y_m_low, -10*jnp.ones(1)))
        y_m_max = jnp.concatenate((y_m_max, 10*jnp.ones(1)))
        y_c_low = jnp.concatenate((y_c_low, -10*jnp.ones(1)))
        y_c_max = jnp.concatenate((y_c_max, 10*jnp.ones(1)))
        y_g_low = jnp.concatenate((y_g_low, -10*jnp.ones(1)))
        y_g_max = jnp.concatenate((y_g_max, 10*jnp.ones(1)))

    plt.rc('text', usetex=True)
    color_i = ["r", "b", "g", "k"]

    ticks = jnp.array(divider)
    ticks = (ticks[:-1] + ticks[1:]) / 2

    for i in range(0, n_dof, 2):

        fig = plt.figure(figsize=(24.0/1.54, 8.0/1.54), dpi=100)
        fig.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=0.95, wspace=0.3, hspace=0.2)
        # fig.canvas.manage.set_window_title('Seed = {0}'.format(seed))

        legend = [mp.patches.Patch(color=color_i[0], label="Model"),
                mp.patches.Patch(color="k", label="Measured")]

        # Plot Torque
        ax0 = fig.add_subplot(2, 1, 1)
        # ax0.set_title(r"$\boldsymbol{\tau}$")
        ax0.set_title('Torque')
        ax0.text(s=f'Joint {i}', x=-0.35, y=.5, fontsize=12, fontweight="bold", rotation=90, horizontalalignment="center", verticalalignment="center", transform=ax0.transAxes)
        if i in force_index:
            ax0.set_ylabel("Force [N]")
        else:
            ax0.set_ylabel("Torque [Nm]")
        ax0.get_yaxis().set_label_coords(-0.2, 0.5)
        ax0.set_ylim(y_t_low[i+0], y_t_max[i+0])
        ax0.set_xticks(ticks)
        ax0.set_xticklabels(test_labels)
        ax0.vlines(divider, y_t_low[i+0], y_t_max[i+0], linestyles='--', lw=0.5, alpha=1.)
        ax0.set_xlim(divider[0], divider[-1])

        ax1 = fig.add_subplot(2, 1, 2)
        ax1.text(s=f'Joint {i+1}', x=-.35, y=0.5, fontsize=12, fontweight="bold", rotation=90,
                horizontalalignment="center", verticalalignment="center", transform=ax1.transAxes)

        ax1.text(s=r"\textbf{(a)}", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
                verticalalignment="center", transform=ax1.transAxes)

        if i in force_index:
            ax1.set_ylabel("Force [N]")
        else:
            ax1.set_ylabel("Torque [Nm]")
        ax1.get_yaxis().set_label_coords(-0.2, 0.5)
        ax1.set_ylim(y_t_low[i+1], y_t_max[i+1])
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(test_labels)
        ax1.vlines(divider, y_t_low[i+1], y_t_max[i+1], linestyles='--', lw=0.5, alpha=1.)
        ax1.set_xlim(divider[0], divider[-1])

        ax0.legend(handles=legend, bbox_to_anchor=(0.0, 1.0), loc='upper left', ncol=1, framealpha=1.)

        # Plot Measured Torque:
        ax0.plot(test_tau[:, i+0], color="k")
        if i+1 < n_dof:
            ax1.plot(test_tau[:, i+1], color="k")

        # Plot Torque:
        ax0.plot(eval_tau[:, i+0], color=color_i[0], alpha=plot_alpha)
        if i+1 < n_dof:
            ax1.plot(eval_tau[:, i+1], color=color_i[0], alpha=plot_alpha)

        fig_dir = repo_dir + f"/figures/{model_type_folder}/{model_name}"
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
        fig.savefig(f"{fig_dir}/joints_torque_{i}_{i+1}.pdf", format="pdf")
        fig.savefig(f"{fig_dir}/joints_torque_{i}_{i+1}.png", format="png")

    if render:
        plt.show()

    print("\n################################################\n\n\n")