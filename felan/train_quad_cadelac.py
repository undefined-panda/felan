import os
import argparse
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp

import optax
from flax.training import train_state

import copy
import matplotlib as mp
from functools import partial
from jax import random

if os.getenv("DISPLAY"):
    try:
        mp.use("Qt5Agg")
        mp.rc('text', usetex=True)
        mp.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    except ImportError:
        pass
repo_dir = os.path.dirname(os.path.abspath(__file__))

from tensorboardX import SummaryWriter
from felan.data_scripts.jax_utils import init_env
from felan.models.np_math_utils import *
from felan.data_scripts.data_loaders import load_custom_dataset

from felan.train import *
from felan.eval import *
from felan.models.cadelac_pot_param import CaDeLaC, get_config_from_dict as get_cadelac_pp_config
from felan.models.lstm_black_box import LSTMBlackBox, get_config_from_dict as get_lstm_config

def loss_fn_cadelac(state, params, batch_data, config: TrainConfig):
    q, qd, qdd, tau, history = batch_data
    tau_hat, dEdt_hat, extras = state.apply_fn(params, q, qd, qdd, history)

    err_inv = jnp.sum((tau_hat - tau) ** 2 / config.norm_tau, axis=1)
    l_mean_inv_dyn = jnp.mean(err_inv)
    l_var_inv_dyn = jnp.var(err_inv)
    l_mean_squared_inv_dyn = jnp.mean(err_inv ** 2)

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

    if config.hyper.get('penalize_extra', False):
        diff_mass = extras['diff_mass']
        l_mean_mass = jnp.mean(diff_mass)
        l_var_mass = jnp.var(diff_mass)
        loss += config.hyper['penalize_diff_mass'] * l_mean_mass
        metrics.update({'l_mean_mass': l_mean_mass, 'l_var_mass': l_var_mass})

        diff_rot_eigenvalue = extras['diff_rot_eigenvalue']
        err_rot_eigenvalue = diff_rot_eigenvalue ** 2
        l_mean_rot_eigenvalue = jnp.mean(err_rot_eigenvalue)
        l_var_rot_eigenvalue = jnp.var(err_rot_eigenvalue)
        loss += config.hyper['penalize_rot_eigenvalue'] * l_mean_rot_eigenvalue
        metrics.update({
            'l_mean_rot_eigenvalue': l_mean_rot_eigenvalue,
            'l_var_rot_eigenvalue': l_var_rot_eigenvalue,
        })

    for key, value in extras.items():
        metrics[f"{key}_mean"] = jnp.mean(value)
        metrics[f"{key}_var"] = jnp.var(value)
        metrics[f"{key}_max"] = jnp.max(value)

    return loss, metrics

if __name__ == "__main__":

    # Read Command Line Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", nargs=1, type=int, required=False, default=[True, ], help="Training using CUDA.")
    parser.add_argument("-i", nargs=1, type=int, required=False, default=[0, ], help="Set the CUDA id.")
    parser.add_argument("-s", nargs=1, type=int, required=False, default=[0, ], help="Set the random seed")
    parser.add_argument("-r", nargs=1, type=int, required=False, default=[1, ], help="Render the figure")
    parser.add_argument("-l", nargs=1, type=int, required=False, default=[0, ], help="Load the model")
    parser.add_argument("-m", nargs=1, type=int, required=False, default=[1, ], help="Save the model")
    parser.add_argument("--robot", type=str, default="go2", choices=["go2", "spot_real", "hyqreal2", "spot_arm_real", "aliengo"])
    parser.add_argument("--nn", type=str, default="CaDeLaC", choices=["CaDeLaC", "LSTM"])
    parser.add_argument("--inertia-param", type=str, default="PrincipalTriangular", choices=["PrincipalTriangular", "PrincipalUnconstrained", "SpatialCov", "SpatialSpd", "SpatialLogCholesky"], help="Inertia parametrization",)
    parser.add_argument("--file_type", type=str, default="pkl", choices=["npz", "pkl"])
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--delan_size", type=int, nargs="+", default=[16,16])
    parser.add_argument("--input_values", type=str, default="joint", choices=["joint", "base", "base_pos_z"])

    args = parser.parse_args()
    seed, cuda, render, load_model, save_model = init_env(parser.parse_args())

    robot_prefix = args.robot
    nn_id = args.nn
    file_type = "."+args.file_type
    only_lstm = nn_id == "LSTM"
    dyn_parametrization = args.inertia_param
    epochs = args.epochs
    delan_size = args.delan_size
    input_values = args.input_values

    if 'arm' in robot_prefix:
        n_arms = 1
        nq_arm = 7
        xml_path = repo_dir + '/data/robot_models/boston_dynamics_spot/spot_arm_full.xml'
        xml_path = 'data/robot_models/boston_dynamics_spot/spot_arm_full.xml'
    else:
        n_arms = 0
        nq_arm = 0
        xml_path = repo_dir + '/data/robot_models/go2/go2.xml'
        xml_path = 'data/robot_models/go2/go2.xml'

    dataset_use = 1.0
    loss_power = False

    nq = 0 #12 + n_arms * nq_arm
    nq_dof_full = 7 + nq
    nv_dof_full = 6 + nq
    flag_normalize_tau = True
    norm_tau_epsilon = 1e-2
    sample_offset = 10
    tau_field = 'tau_ext_total'
    final_sample_offset = sample_offset
    save_checkpoint_model = False
    log_period = 50
    all_data_pin_to_mj = True if nn_id == 'MjxDNEA' else False

    tri_ineq = True
    skew_sym_ineq = True
    mass_ineq = True

    add_noise_to_load_data = False
    pot_net = False

    # Model DoF - nq = 6 creates a 6x6 inertia matrix
    nq_dof_model = nq + (7 if nn_id == 'MjxDNEA' else 6)
    nv_dof_model = nq + 6

    if sample_offset > 0:
        print(f'#####\n Using sample offset {sample_offset}!!\n#####')

    dataset_map = {
            'go2':          'go2_sim_n_runs_50_data_freq_100hz',
            'spot_real':    'spot_real_freq_100hz',
            'spot_arm_real':'spot_arm_real_freq_100hz',
            'hyqreal2':     'hyqreal2_real_freq_100hz',
            'aliengo':      'quad_mass_dataset_run7'
        }
    
    dataset_name = dataset_map[robot_prefix]
    dataset_path = 'data/datasets/' + dataset_name + file_type
    dataset_full_path = repo_dir + '/' + dataset_path

    test_labels_wanted = {
        'go2':           ['env_0_run_2', 'env_0_run_10', 'env_0_run_11', 'env_0_run_28', 'env_0_run_41'],
        'spot_real':     ['env_0_run_32', 'env_0_run_52'],
        'spot_arm_real': ['env_0_run_60', 'env_0_run_71'],
        'hyqreal2':      ['env_0_run_2', 'env_0_run_5', 'env_0_run_25'],
        'aliengo':       []
    }[robot_prefix]

    model_folder = str(robot_prefix) + '/' + nn_id

    time_window = 15

    train_data, test_data, divider, dt_mean, test_base_mass = load_custom_dataset(
        dataset_full_path,
        hist_length=time_window,
        sample_offset=0,
        seed=seed
    )

    (train_labels, train_qp, train_qv, train_qa, train_tau,
    train_hist_joint_pos, train_hist_joint_vel, train_hist_diff_tau,
    train_hist_base_orient, train_hist_base_vel, train_hist_base_ang_vel,
    train_hist_base_pos_z) = train_data

    (test_labels, test_qp, test_qv, test_qa, test_tau, test_m, test_c, test_g,
    test_hist_joint_pos, test_hist_joint_vel, test_hist_diff_tau,
    test_hist_base_orient, test_hist_base_vel, test_hist_base_ang_vel,
    test_hist_base_pos_z) = test_data

    print(f'nq_dof_model: {nq_dof_model} | nq_dof_full: {nq_dof_full}')
    if nq_dof_model != nq_dof_full:
        raw_train_qp = copy.deepcopy(train_qp)
        raw_test_qp = copy.deepcopy(test_qp)
        
        train_euler = get_euler_from_pin_quat(raw_train_qp[:,3:7])
        test_euler = get_euler_from_pin_quat(raw_test_qp[:,3:7])
        train_qp = jnp.hstack((raw_train_qp[:,0:3], train_euler, raw_train_qp[:,7:]))
        test_qp = jnp.hstack((raw_test_qp[:,0:3], test_euler, raw_test_qp[:,7:]))

    if all_data_pin_to_mj:
        print('Converting quaternion data to mujoco convention !!!!!')
        raw_train_qp = copy.deepcopy(train_qp)
        raw_test_qp = copy.deepcopy(test_qp)

        train_mj_quat = get_mj_quat_from_pin_quat(raw_train_qp[:,3:7])
        train_qp = jnp.hstack((raw_train_qp[:,0:3], train_mj_quat, raw_train_qp[:,7:]))
        test_mj_quat = get_mj_quat_from_pin_quat(raw_test_qp[:,3:7])
        test_qp = jnp.hstack((raw_test_qp[:,0:3], test_mj_quat, raw_test_qp[:,7:]))

    match input_values.split("_")[0]:
        case "joint":            
            # input size: 12+12+6 = 30
            train_history = jnp.asarray(np.concatenate(
                [train_hist_joint_pos, train_hist_joint_vel, train_hist_diff_tau],
                axis=-1
            ))

            test_history = jnp.asarray(np.concatenate(
                [test_hist_joint_pos, test_hist_joint_vel, test_hist_diff_tau],
                axis=-1
            ))

            train_input_list = [train_qp, train_qv, train_qa, train_tau, train_history]
            test_input_list = [test_qp, test_qv, test_qa, test_tau, test_m, test_c, test_g, test_history]

        case "base":
            match input_values:
                case "base_pos_z":
                    # input size: 4+3+3+6 = 16
                    train_history = jnp.asarray(np.concatenate(
                        [train_hist_base_orient, train_hist_base_vel, train_hist_base_ang_vel, train_hist_base_pos_z, train_hist_diff_tau],
                        axis=-1
                    ))

                    test_history = jnp.asarray(np.concatenate(
                        [test_hist_base_orient, test_hist_base_vel, test_hist_base_ang_vel, test_hist_base_pos_z, test_hist_diff_tau],
                        axis=-1
                    ))

                case "base":
                    # input size: 4+3+6 = 13
                    train_history = jnp.asarray(np.concatenate(
                        [train_hist_base_orient, train_hist_base_vel, train_hist_base_ang_vel, train_hist_diff_tau],
                        axis=-1
                    ))

                    test_history = jnp.asarray(np.concatenate(
                        [test_hist_base_orient, test_hist_base_vel, test_hist_base_ang_vel, test_hist_diff_tau],
                        axis=-1
                    ))

                case _:
                    raise ValueError(f"Invalid value for 'input_values': {input_values}")

            train_input_list = [train_qp, train_qv, train_qa, train_tau, train_history]
            test_input_list = [test_qp, test_qv, test_qa, test_tau, test_m, test_c, test_g, test_history]

        case _:
            raise ValueError(f"Invalid value for 'input_values': {input_values}")

    print(f"# Shape of history: {train_history.shape[-1]}")

    print("\n\n################################################")
    print("Runs:")
    print("   Test Runs = {0}".format(test_labels))
    print("  Train Runs = {0}".format(train_labels))
    print("# Training Samples = {0:05d}".format(int(train_qp.shape[0])))
    print("")

    # Training Parameters:
    # Construct Hyperparameters:
    hyper = {
             'diagonal_epsilon': 0.01 if nn_id == 'MjxDNEA' else 0.1,
             'activation': 'Tanh',
             'net_arch_torso': [0, 0],
             'net_arch_arm': [0, 0] if nq_arm == 0 else [16, 16],
             'net_arch_leg': [16, 16],
             'net_arch_base_rot': [16, 16],
             'net_arch_pot': delan_size, # size of pot_net
             'net_arch_mlp': [32, 32],
             'n_minibatch': 1024,
             'learning_rate': 5.e-04,
             'weight_decay': 1.e-5,
             'init_tf': True,
             'act_ld': 'Softplus',
             'softplus_beta': 1.0,
             'net_arch_inertia_full': delan_size, # size of inertia_net
            ## Extras
             'mass_ineq': mass_ineq,
             'skew_sym_ineq': skew_sym_ineq,
             'tri_ineq': tri_ineq,
             'tri_ineq_act_name': 'Softplus',
             'tri_ineq_softplus_shift': 0.0,
             'tri_ineq_softplus_beta': 6.0,
             'diff_mass_softplus_beta': 5.0,
             'diff_mass_shift_loss': 5.0,
             'hr_sigma_epsilon': 1e-1,
             'H_epsilon': 0.0,
             'mass_epsilon': 0.01,
             'rot_epsilon': 0.001,
             'penalize_diff_mass': 1.0,
             'penalize_rot_eigenvalue': 1.0,
             'penalize_extra': True if nn_id == "FeLaN" else False,
             'softplus_beta_base_rot': 10.0 if tri_ineq else 1.0,
             'final_sigma_epsilon': 1e-2,
             'dnea_inertia_epsilon': 1e-4,
             'dnea_mass_epsilon': 1e-2,
             # Robot
             'nq_dof': nq_dof_full,
             'nv_dof': nv_dof_full,
             'n_dof_torso': 0,
             'n_arms': n_arms,
             'n_dof_arm': nq_arm,
             'n_legs': 4,
             'n_dof_leg': 3,
             'init_mass': 50.0,
             'pot_net': pot_net,
             'xml_path': xml_path,
             # MJX
             'dyn_parametrization': dyn_parametrization,
             # Data
             'dataset_path': dataset_path,
             'tau_field': tau_field,
             'train_labels': train_labels,
             'test_labels': test_labels,
             # LSTM
             'lstm_hidden_size': 10,
             'lstm_num_layers': 5,
             'lstm_dropout': 0.0,
             'time_window': time_window,
             'n_output': 6 if only_lstm else 10,
             #
             'max_epoch': epochs
            }

    if flag_normalize_tau:
        norm_tau = jnp.var(train_tau,axis=0) + norm_tau_epsilon
    else:
        norm_tau = jnp.ones(6)
    print(f'Norm Tau:\n{norm_tau}')

    batch_size  = hyper['n_minibatch']

    ## Define Model Name
    if only_lstm:
        model_name = 'epochs_' + str(hyper['max_epoch']) + '_' + dataset_name + '_input_values_' + input_values + '_seed_' + str(seed) + '_LSTM'
    else:
        model_name = 'epochs_' + str(hyper['max_epoch']) + '_' + dataset_name + '_input_values_' + input_values + '_seed_' + str(seed) + '_' + ','.join(str(x) for x in delan_size)

    if nn_id == 'MjxDNEA':
        model_name += '_' + hyper['dyn_parametrization']

    rng = jax.random.PRNGKey(seed)

    print(f"\n# Training samples (post-history) = {int(train_qp.shape[0])}")
    print(f"# Test samples (post-history) = {int(test_qp.shape[0])}")

    train_dataset = create_dataset(train_input_list)
    eval_dataset = create_dataset(test_input_list)

    # Construct model:
    if only_lstm:
        nn_config = get_lstm_config(hyper)
        learned_model = LSTMBlackBox(n_dof=nq, config=nn_config)
    else:
        nn_config = get_cadelac_pp_config(hyper)
        learned_model = CaDeLaC(nv_dof_model, nn_config)

    folder_path = repo_dir + f"/trained_models/{model_folder}"

    if load_model:
        ################# Load Model #################
        eval_params, _ = load_model_fn(model_name, folder_path)

    else:
        ################# Train Model #################

        ## Initialize Model Parameters
        dumb_n_batch = 5
        dumb_q = jnp.zeros((dumb_n_batch, nq_dof_model))
        dumb_qd = jnp.zeros((dumb_n_batch, nv_dof_model))
        dumb_history = jnp.zeros((dumb_n_batch, time_window, train_history.shape[-1]))
        rng, param_rng = jax.random.split(rng, num=2)
        params = learned_model.init(param_rng, dumb_q, dumb_qd, dumb_qd, dumb_history)

        # Check number of parameters
        sum_param = count_parameters(params)
        print(f'Number of model parameters {sum_param}')

        # Training Parameters:
        print("\n################################################")
        if nn_id == 'MjxDNEA':
            print(f"Training {learned_model.__class__.__name__} ({hyper['dyn_parametrization']}):")
        else:
            print(f"Training {learned_model.__class__.__name__}:")

            # Init Tensorboard
        tb_folder = repo_dir + f"/tensorboard/{model_folder}/{model_name}"
        tb_writer = SummaryWriter(log_dir=tb_folder)

        # Init Optimizer
        optimizer = optax.adamw(
            learning_rate=hyper["learning_rate"],
            weight_decay=hyper["weight_decay"]
        )

        opt_state = train_state.TrainState.create(apply_fn = learned_model.apply,
                                                  params = params,
                                                  tx = optimizer)
        
        train_config = TrainConfig(num_epochs = hyper['max_epoch'],
                                   loss_power = loss_power,
                                   norm_tau = norm_tau,
                                   batch_size = hyper['n_minibatch'],
                                   log_period = log_period,
                                   hyper = hyper,
                                   tb_folder = tb_folder,
                                   repo_dir = repo_dir,
                                   save_checkpoint_model = save_checkpoint_model,)

        # # Train Model
        rng, train_rng = jax.random.split(rng, num=2)
        train_model_state, _ = train_model(opt_state, train_dataset, train_rng, train_config, tb_writer)

        # Get Trained parameters for Evaluation
        eval_params = train_model_state.params

        # Save the Model:
        if save_model:
            save_model_fn(train_model_state.params, hyper, model_name, folder_path)


    ################# Evaluate Model #################
    norm_tau_eval = norm_tau
    eval_results, eval_metrics = eval_components(learned_model, eval_params, eval_dataset, norm_tau_eval,)

    ################# Plot Results #################
    plot_dataset = (eval_dataset[0], eval_dataset[1], eval_dataset[2],
                    eval_dataset[3], eval_dataset[4], eval_dataset[5],
                    eval_dataset[6])

    n_test_post = test_qp.shape[0]
    plot_divider = np.linspace(0, n_test_post, len(test_labels) + 1).astype(int)

    for i in range(len(test_labels)):
        test_labels[i] += f"\nMass: ({test_base_mass[i]:.2f})"

    plot_torques(eval_results, plot_dataset, test_labels, plot_divider,
                 model_folder, model_name, render, force_index=[0, 1, 2],
                 norm_tau=norm_tau_eval, repo_dir=repo_dir)
