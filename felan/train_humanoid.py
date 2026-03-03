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
from felan.data_scripts.data_loaders import load_dataset
from felan.models.np_math_utils import *

from felan.train import *
from felan.eval import *
from felan.models.felan import FeLaN, get_config_from_dict as get_felan_config
from felan.models.felan_branch_sparsity import FeLaNBranchSparsity, get_config_from_dict as get_felan_bs_config
from felan.models.delan import DeLaN, get_config_from_dict as get_delan_config
from felan.models.delan_pot_param import DeLaNPotParam, get_config_from_dict as get_delan_pp_config
from felan.models.mlp_black_box import MLPBlackBox, get_config_from_dict as get_mlp_config
from felan.models.mjx_dnea import MjxDNEA, get_config_from_dict as get_mjx_config

if __name__ == "__main__":

    # Read Command Line Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", nargs=1, type=int, required=False, default=[True, ], help="Training using CUDA.")
    parser.add_argument("-i", nargs=1, type=int, required=False, default=[0, ], help="Set the CUDA id.")
    parser.add_argument("-s", nargs=1, type=int, required=False, default=[0, ], help="Set the random seed")
    parser.add_argument("-r", nargs=1, type=int, required=False, default=[1, ], help="Render the figure")
    parser.add_argument("-l", nargs=1, type=int, required=False, default=[0, ], help="Load the model")
    parser.add_argument("-m", nargs=1, type=int, required=False, default=[1, ], help="Save the model")
    parser.add_argument("--robot", type=str, default="talos", choices=["talos", "talos_real"])
    parser.add_argument("--nn", type=str, default="FeLaN", choices=["FeLaN", "FeLaNBS", "DeLaN", "DeLaNPP", "MLP", "MjxDNEA"])
    parser.add_argument("--inertia-param", type=str, default="PrincipalTriangular", choices=["PrincipalTriangular", "PrincipalUnconstrained", "SpatialCov", "SpatialSpd", "SpatialLogCholesky"], help="Inertia parametrization",)

    args = parser.parse_args()
    seed, cuda, render, load_model, save_model = init_env(parser.parse_args())

    robot_prefix = args.robot
    nn_id = args.nn
    dyn_parametrization = args.inertia_param

    xml_path = repo_dir + '/data/robot_models/pal_talos/talos_motor_arm_reduced.xml'

    if nn_id == 'FeLaN':
        nn_type = FeLaN
        get_config_from_dict = get_felan_config
    if nn_id == 'FeLaNBS':
        nn_type = FeLaNBranchSparsity
        get_config_from_dict = get_felan_bs_config
    if nn_id == 'DeLaN':
        nn_type = DeLaN
        get_config_from_dict = get_delan_config
    if nn_id == 'DeLaNPP':
        nn_type = DeLaNPotParam
        get_config_from_dict = get_delan_pp_config
    if nn_id == 'MLP':
        nn_type = MLPBlackBox
        get_config_from_dict = get_mlp_config
    if nn_id == 'MjxDNEA':
        nn_type = MjxDNEA
        get_config_from_dict = get_mjx_config

    dataset_use = 1.0
    minibatch = 1024
    loss_power = False

    nq = 22
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

    print(f'nq_dof_model {nq_dof_model} nv_dof_model {nv_dof_model}')


    if sample_offset > 0:
        print(f'#####\n Using sample offset {sample_offset}!!\n#####')


    if robot_prefix == 'talos':
        dataset_name = 'talos_sim_freq_100hz'
        dataset_name_short = 'talos_sim_freq_100hz'
        tau_field = 'tau_dyn'

    elif robot_prefix == 'talos_real':
        dataset_name = 'talos_real_freq_100hz'
        dataset_name_short = 'talos_real_freq_100hz'

    model_folder = str(robot_prefix) + '/' + nn_id
    dataset_path = 'data/datasets/' + dataset_name + '.pkl'
    dataset_full_path = repo_dir + '/' + dataset_path

    test_labels = [0.1] # 10 % of all the environments
    if robot_prefix == 'talos':
        test_labels = ['env_0_run_3', 'env_0_run_16', 'env_0_run_24', 'env_0_run_34', 'env_0_run_45']
    else:
        test_labels = ['env_0_run_10', 'env_0_run_20']


    train_data, test_data, divider, dt_mean = load_dataset(filename=dataset_full_path, test_label=test_labels,
                                                           sample_offset=sample_offset, final_sample_offset=final_sample_offset,
                                                           dataset_use=dataset_use, nq_dof=nq_dof_full, nv_dof=nv_dof_full, tau_field=tau_field)


    train_labels, train_qp, train_qv, train_qa, train_tau = train_data
    test_labels, test_qp, test_qv, test_qa, test_tau, test_m, test_c, test_g = test_data

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
    print("")

    # Training Parameters:
    # Construct Hyperparameters:
    hyper = {
             'diagonal_epsilon': 0.01 if nn_id == 'MjxDNEA' else 0.1,
             'activation': 'Tanh',
             'net_arch_torso': [16, 16],
             'net_arch_arm': [16, 16],
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
             'mass_epsilon': 0.1,
             'rot_epsilon': 0.01,
             'penalize_diff_mass': 100.0,
             'penalize_rot_eigenvalue': 1.0,
             'penalize_extra': True if nn_id == "FeLaN" else False,
             'softplus_beta_base_rot': 5.0 if tri_ineq else 1.0,
             'final_sigma_epsilon': 1e-2,
             'dnea_inertia_epsilon': 1e-4,
             'dnea_mass_epsilon': 1e-2,
             # Robot
             'nq_dof': nq_dof_full,
             'nv_dof': nv_dof_full,
             'n_dof_torso': 2,
             'n_arms': 2,
             'n_dof_arm': 4,
             'n_legs': 2,
             'n_dof_leg': 6,
             'init_mass': 100.0,
             'pot_net': pot_net,
             'xml_path': xml_path,
             # MJX
             'dyn_parametrization': dyn_parametrization,
             # Data
             'dataset_path': dataset_path,
             'tau_field': tau_field,
             'train_labels': train_labels,
             'test_labels': test_labels,
             #
             'max_epoch': 3000,
            }

    if flag_normalize_tau:
        norm_tau = jnp.var(train_tau,axis=0) + norm_tau_epsilon
    else:
        norm_tau = jnp.ones(nv_dof_full)
    print(f'Norm Tau:\n{norm_tau}')

    ## Define Model Name
    model_name = 'epochs_' + str(hyper['max_epoch'])
    model_name += '_' + dataset_name_short


    if nn_id == 'MjxDNEA':
        model_name += '_' + hyper['dyn_parametrization']

    model_name += '_' + str(seed)

    rng = jax.random.PRNGKey(seed)

    train_input_list = [train_qp, train_qv, train_qa, train_tau]
    test_input_list = [test_qp, test_qv, test_qa, test_tau, test_m, test_c, test_g]

    train_dataset = create_dataset(train_input_list)
    eval_dataset = create_dataset(test_input_list)

    # Construct model:
    nn_config = get_config_from_dict(hyper)
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
        rng, param_rng = jax.random.split(rng, num=2)
        params = learned_model.init(param_rng, dumb_q, dumb_qd, dumb_qd)

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

        # Get Trained parameters for Evaluation
        eval_params = train_model_state.params

        # Save the Model:
        if save_model:
            save_model_fn(train_model_state.params, hyper, model_name, folder_path)


    ################# Evaluate Model #################
    norm_tau_eval = norm_tau
    eval_results, eval_metrics = eval_components(learned_model, eval_params, eval_dataset, norm_tau_eval)


    ################# Plot Results #################
    plot_components(eval_results, eval_dataset, test_labels, divider, model_folder, model_name, render, force_index=[0, 1, 2], repo_dir=repo_dir)