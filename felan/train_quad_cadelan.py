import os
import argparse
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp

import optax
from flax.training import train_state

import copy
import matplotlib as mp

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
from felan.data_scripts.data_loaders import load_custom_dataset
from felan.models.np_math_utils import *

from felan.train import *
from felan.eval import *

from felan.models.lstm_black_box import LSTMBlackBox, get_config_from_dict as get_lstm_config
from felan.models.cadelan_pot_param import CaDeLaN, get_config_from_dict as get_cadelan_pp_config

if __name__ == "__main__":

    # Read Command Line Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", nargs=1, type=int, required=False, default=[True, ], help="Training using CUDA.")
    parser.add_argument("-i", nargs=1, type=int, required=False, default=[0, ], help="Set the CUDA id.")
    parser.add_argument("-s", nargs=1, type=int, required=False, default=[0, ], help="Set the random seed")
    parser.add_argument("-r", nargs=1, type=int, required=False, default=[1, ], help="Render the figure")
    parser.add_argument("-l", nargs=1, type=int, required=False, default=[0, ], help="Load the model")
    parser.add_argument("-m", nargs=1, type=int, required=False, default=[1, ], help="Save the model")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--nn", type=str, default="LSTM", choices=["LSTM", "CaDeLaN"])
    parser.add_argument("--lstm_num_layers", type=int, default=5, help="Number of stacked LSTM layers in the context encoder")
    parser.add_argument("--lstm_hidden_size", type=int, default=10, help="Hidden size of each LSTM layer")
    parser.add_argument("--delan_size", type=int, nargs="+", default=[16,16])
    parser.add_argument("--history_span", type=float, default=0.5, help="Wall-clock span of the LSTM context window [s]")
    parser.add_argument("--history_stride", type=int, default=4, help="Spacing between context-window samples [samples]")
    parser.add_argument("--history_input", type=str, default="joint", choices=["joint", "base"], help="Input data for LSTM history")
    
    args = parser.parse_args()
    seed, cuda, render, load_model, save_model = init_env(parser.parse_args())

    epochs = args.epochs
    nn_id = args.nn
    lstm_num_layers = args.lstm_num_layers
    lstm_hidden_size = args.lstm_hidden_size
    delan_size = args.delan_size
    history_span = args.history_span
    history_stride = args.history_stride
    history_input = args.history_input

    lstm_alone = nn_id == "LSTM"
    n_arms, nq_arm = 0, 0

    minibatch = 1024
    loss_power = False

    # nq = 12 + n_arms * nq_arm
    nq = 0 # joint part of jacobians is zero, i.e. only depends on base part
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

    # Model DoF
    nq_dof_model = nq + (7 if nn_id == 'MjxDNEA' else 6)
    nv_dof_model = nq + 6

    if sample_offset > 0:
        print(f'#####\n Using sample offset {sample_offset}!!\n#####')

    dataset_name = "quad_mass_dataset_run8"

    model_folder = 'aliengo/' + nn_id
    dataset_path = 'data/datasets/' + dataset_name + '.npz'
    dataset_full_path = repo_dir + '/' + dataset_path

    def dt_mean_probe(dataset_full_path):
        t = np.load(dataset_full_path)["time"]
        dt = np.diff(t, axis=-1)
        assert np.var(dt) < 1e-12, "dataset has a non-constant logging period"
        return float(np.mean(dt))

    f_log = 1.0 / dt_mean_probe(dataset_full_path)
    time_window = max(1, int(round(history_span * f_log / history_stride)))
    print(f"# Context: {history_span}s @ {f_log:.1f}Hz / stride {history_stride} "
          f"-> time_window = {time_window} points")
    
    test_labels = []

    train_data, test_data, divider, dt_mean, test_base_mass = load_custom_dataset(
            filename=dataset_full_path,
            sample_offset=0,
            hist_length=time_window,
            seed=seed,
            hist_stride=history_stride,
            history_input=history_input,
            nq_dof=nq_dof_full,
            nv_dof=nv_dof_full
        )

    train_labels, train_qp, train_qv, train_qa, train_tau, train_history = train_data
    test_labels, test_qp, test_qv, test_qa, test_tau, test_m, test_c, test_g, test_history = test_data

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

    print("\n\n################################################")
    print("Runs:")
    print("   Test Runs = {0}".format(test_labels))
    print("  Train Runs = {0}".format(train_labels))
    print("# Training Samples = {0:05d}".format(int(train_qp.shape[0])))
    print(f"# Shape of history: {train_history.shape[-1]}")
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
             'net_arch_pot': [16, 16],
             'net_arch_mlp': [32, 32],
             'n_minibatch': minibatch,
             'learning_rate': 5.e-04,
             'weight_decay': 1.e-5,
             'init_tf': True,
             'act_ld': 'Softplus',
             'softplus_beta': 1.0,
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
             'nq': nq,
             'nq_dof': nq_dof_full,
             'nv_dof': nv_dof_full,
             'n_dof_torso': 0,
             'n_arms': n_arms,
             'n_dof_arm': nq_arm,
             'n_legs': 4,
             'n_dof_leg': 3,
             'init_mass': 50.0,
             'pot_net': pot_net,
             # Data
             'dataset_path': dataset_path,
             'tau_field': tau_field,
             'train_labels': train_labels,
             'test_labels': test_labels,
             # LSTM
             'lstm_hidden_size': args.lstm_hidden_size,
             'lstm_num_layers': args.lstm_num_layers,
             'time_window': time_window,
             'history_stride': history_stride,
             'history_span': history_span,
             'f_log': f_log,
             'n_output': 6 if lstm_alone else 10, # residual base torque / environment context
             #
             'max_epoch': epochs
            }

    if flag_normalize_tau:
        norm_tau = jnp.var(train_tau,axis=0) + norm_tau_epsilon
    else:
        norm_tau = jnp.ones(nv_dof_full)
    print(f'Norm Tau:\n{norm_tau}')

    ## Define Model Name
    model_name = 'epochs_' + str(hyper['max_epoch']) + '_' + dataset_name + '_input_values_' + history_input + '_seed_' + str(seed)
    model_name += f"_lstm{hyper['lstm_num_layers']}x{hyper['lstm_hidden_size']}"
    model_name += f"_span{hyper['history_span']}-stride{hyper['history_stride']}"

    rng = jax.random.PRNGKey(seed)

    train_input_list = [train_qp, train_qv, train_qa, train_tau, train_history]
    test_input_list = [test_qp, test_qv, test_qa, test_tau, test_m, test_c, test_g, test_history]

    train_dataset = create_dataset(train_input_list)
    eval_dataset = create_dataset(test_input_list)

    # Construct model:
    match nn_id:
        case "LSTM":
            nn_config = get_lstm_config(hyper)
            nn_type = LSTMBlackBox
        case "CaDeLaN":
            nn_config = get_cadelan_pp_config(hyper)
            nn_type = CaDeLaN

    learned_model = nn_type(nv_dof_model, nn_config)

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
                                   save_checkpoint_model = save_checkpoint_model,
                                )

        # # Train Model
        rng, train_rng = jax.random.split(rng, num=2)

        train_model_state, _ = train_model(opt_state, train_dataset, train_rng, train_config, tb_writer)
        quit()
        print("after quit")

        # Get Trained parameters for Evaluation
        eval_params = train_model_state.params

        # Save the Model:
        if save_model:
            save_model_fn(train_model_state.params, hyper, model_name, folder_path)


    ################# Evaluate Model #################
    norm_tau_eval = norm_tau
    eval_results, eval_metrics = eval_components(learned_model, eval_params, eval_dataset, norm_tau_eval)


    ################# Plot Results #################
    # plot_components(eval_results, eval_dataset, test_labels, divider, model_folder, model_name, render, force_index=[0, 1, 2], repo_dir=repo_dir)
