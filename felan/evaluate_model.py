import os
import argparse
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp

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
    parser.add_argument("--robot", type=str, default="go2", choices=["go2", "spot_real", "hyqreal2", "spot_arm_real", "talos", "talos_real"])
    parser.add_argument("--model_name", type=str, default="model")
    parser.add_argument("--nn", type=str, default="FeLaN", choices=["FeLaN", "FeLaNBS", "DeLaN", "DeLaNPP", "MLP", "MjxDNEA"])
    parser.add_argument("--model_folder", type=str, default="trained_models")

    args = parser.parse_args()
    seed, cuda, render, load_model, save_model = init_env(parser.parse_args())

    robot_prefix = args.robot
    model_name = args.model_name
    model_folder = args.model_folder
    nn_id = args.nn

    folder_path = repo_dir + f"/{model_folder}/{robot_prefix}/{nn_id}"
    eval_params, hyper = load_model_fn(model_name, folder_path)

    # Get Hyperparameters
    nq_dof = hyper['nq_dof']
    nv_dof = hyper['nv_dof']
    dataset_full_path = repo_dir + '/' + hyper['dataset_path']
    test_labels = hyper['test_labels']
    tau_field = hyper['tau_field']
    hyper['xml_path'] = repo_dir + '/' + hyper['xml_path']

    sample_offset = 10
    final_sample_offset = 10
    flag_normalize_tau = True
    norm_tau_epsilon = 1e-2

    train_data, test_data, divider, dt_mean = load_dataset(filename=dataset_full_path, test_label=test_labels,
                                                           sample_offset=sample_offset, final_sample_offset=final_sample_offset,
                                                           nq_dof=nq_dof, nv_dof=nv_dof, tau_field=tau_field)
    
    train_tau = train_data[4]
    test_labels, test_qp, test_qv, test_qa, test_tau, test_m, test_c, test_g = test_data

    if nn_id == 'MjxDNEA':
        print('Converting quaternion data to mujoco convention !!!!!')
        raw_test_qp = copy.deepcopy(test_qp)
        test_mj_quat = get_mj_quat_from_pin_quat(raw_test_qp[:,3:7])
        test_qp = jnp.hstack((raw_test_qp[:,0:3], test_mj_quat, raw_test_qp[:,7:]))
    else:
        raw_test_qp = copy.deepcopy(test_qp)
        test_euler = get_euler_from_pin_quat(raw_test_qp[:,3:7])
        test_qp = jnp.hstack((raw_test_qp[:,0:3], test_euler, raw_test_qp[:,7:]))

    if flag_normalize_tau:
        norm_tau_eval = jnp.var(train_tau,axis=0) + norm_tau_epsilon
    else:
        norm_tau_eval = jnp.ones(nv_dof)

    print("\n\n################################################")
    print("Runs:")
    print("   Test Runs = {0}".format(test_labels))
    print("# Test Samples = {0:05d}".format(int(test_qp.shape[0])))
    print("")

    rng = jax.random.PRNGKey(seed)
    test_input_list = [test_qp, test_qv, test_qa, test_tau, test_m, test_c, test_g]

    eval_dataset = create_dataset(test_input_list)

    # Construct model:
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

    nn_config = get_config_from_dict(hyper)
    learned_model = nn_type(nv_dof, nn_config)

    ################# Evaluate Model #################
    eval_results, eval_metrics = eval_components(learned_model, eval_params, eval_dataset, norm_tau_eval)


    ################# Plot Results #################
    if robot_prefix in ['talos', 'go2']:
        # Plot components only for simulated data when ground truth is available
        plot_components(eval_results, eval_dataset, test_labels, divider, model_folder, model_name, render, force_index=[0, 1, 2], repo_dir=repo_dir)
    else:
        plot_torques(eval_results, eval_dataset, test_labels, divider, model_folder, model_name, render, force_index=[0, 1, 2], repo_dir=repo_dir)