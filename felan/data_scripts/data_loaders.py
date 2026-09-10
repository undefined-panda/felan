import dill as pickle
import numpy as np
import copy
# change

def load_dataset(filename="data/datasets/go2_sim_n_runs_50_data_freq_100hz.pkl", test_label=["env_0_run_0","env_0_run_1","env_0_run_2"],
                sample_offset = 0, final_sample_offset = 0, dataset_use = 1.0, nq_dof = 31,
                nv_dof = 30, seed = None, tau_field = 'tau_ext_total'):

    with open(filename, 'rb') as f:
        data = pickle.load(f)
    org_time = copy.deepcopy(data['t'])

    if seed is not None:
        rng = np.random.default_rng(seed)

    # Split the dataset in train and test set:
    if len(test_label) == 1 and (type(test_label[0]) == type(0.1)):
        test_rate = test_label[0]
        n_test_label = test_rate * len(data["labels"])
        test_label = np.random.choice(data["labels"], size=int(n_test_label), replace=False)

    test_idx = [data["labels"].index(x) for x in test_label]

    dt = np.concatenate([org_time[idx][1:] - org_time[idx][:-1] for idx in test_idx])
    dt_mean, dt_var = np.mean(dt), np.var(dt)

    train_labels, test_labels = [], []
    train_qp, train_qv, train_qa, train_tau = np.zeros((0, nq_dof)), np.zeros((0, nv_dof)), np.zeros((0, nv_dof)), np.zeros((0, nv_dof))

    test_qp, test_qv, test_qa, test_tau = np.zeros((0, nq_dof)), np.zeros((0, nv_dof)), np.zeros((0, nv_dof)), np.zeros((0, nv_dof))
    test_m, test_c, test_g = np.zeros((0, nv_dof)), np.zeros((0, nv_dof)), np.zeros((0, nv_dof))

    divider = [0, ]   # Contains idx between characters for plotting

    for i in range(int(dataset_use*len(data["labels"]))):
        idxs = np.arange(sample_offset, len(data["q"][i]) - final_sample_offset)

        if i in test_idx:
            test_labels.append(data["labels"][i])
            test_qp = np.vstack((test_qp, data["q"][i][idxs]))
            test_qv = np.vstack((test_qv, data["qd"][i][idxs]))
            test_qa = np.vstack((test_qa, data["qdd"][i][idxs]))
            test_tau = np.vstack((test_tau, data[tau_field][i][idxs]))

            test_m = np.vstack((test_m, data["tau_m"][i][idxs]))
            test_c = np.vstack((test_c, data["tau_c"][i][idxs]))
            test_g = np.vstack((test_g, data["tau_g"][i][idxs]))

            divider.append(test_qp.shape[0])

        else:
            train_labels.append(data["labels"][i])
            train_qp = np.vstack((train_qp, data["q"][i][idxs]))
            train_qv = np.vstack((train_qv, data["qd"][i][idxs]))
            train_qa = np.vstack((train_qa, data["qdd"][i][idxs]))

            train_tau = np.vstack((train_tau, data[tau_field][i][idxs]))

    train_data = (train_labels, train_qp, train_qv, train_qa, train_tau)
    test_data = (test_labels, test_qp, test_qv, test_qa, test_tau, test_m, test_c, test_g)
    return train_data, test_data, divider, dt_mean

def load_custom_dataset(filename, sample_offset = 0, hist_length = 0, seed=None,
                            train_size=0.85, hist_stride = 1, history_input = "joint",
                            nq_dof = 19, nv_dof = 18):
    
    def create_historical_data(data, hist_length, list_key, stride=1):
        data_hist = {}
        span = hist_length * stride
        for key in list_key:
            data_hist[key] = [[] for _ in range(len(data[key]))]

            for i, run_data in enumerate(data[key]):
                window = np.lib.stride_tricks.sliding_window_view(run_data, window_shape=span, axis=0)
                strided_window = np.moveaxis(window, -1, 1)[:, ::stride]
                data_hist[key][i] = strided_window[:len(run_data) - span]

        return data_hist
    
    raw = np.load(filename)
    n_runs = raw["base_pos"].shape[0]
    base_mass = raw['base_mass'].mean(axis=1)

    # to keep it similar as load_dataset
    data = {
        'labels': [f'run_{i}' for i in range(n_runs)],
        'time': [raw['time'][i] for i in range(n_runs)],

        'qp': [np.concatenate([raw['base_pos'], raw['base_orient']],   axis=-1)[i] for i in range(n_runs)],
        'qv': [np.concatenate([raw['base_vel'], raw['base_ang_vel']],  axis=-1)[i] for i in range(n_runs)],
        'qa': [np.concatenate([raw['base_acc']],                       axis=-1)[i] for i in range(n_runs)],

        'diff_tau': [(raw['diff_tau_m_nom'] + raw['diff_tau_c_nom'] + raw['diff_tau_g_nom'])[..., :6][i] for i in range(n_runs)],
        'diff_tau_m': [raw['diff_tau_m_nom'][..., :6][i] for i in range(n_runs)],
        'diff_tau_c': [raw['diff_tau_c_nom'][..., :6][i] for i in range(n_runs)],
        'diff_tau_g': [raw['diff_tau_g_nom'][..., :6][i] for i in range(n_runs)],

        'joint_pos': [raw['joint_pos'][i] for i in range(n_runs)],
        'joint_vel': [raw['joint_vel'][i] for i in range(n_runs)],
        'base_orient': [raw['base_orient'][i] for i in range(n_runs)],
        'base_vel': [raw['base_vel'][i] for i in range(n_runs)],
        'base_ang_vel': [raw['base_ang_vel'][i] for i in range(n_runs)],
    }

    if history_input == "joint":
        list_key = ["joint_pos", "joint_vel", "diff_tau"]
    else:
        list_key = ["base_orient", "base_vel", "base_ang_vel", "diff_tau"]

    data_hist = create_historical_data(data, hist_length, list_key, stride=hist_stride)

    # Split the dataset in train and test set:
    rng = np.random.default_rng(seed)
    n_test = int(n_runs * (1 - train_size))
    test_run_indices = rng.choice(np.arange(n_runs), n_test, replace=False)
    test_label = [data["labels"][i] for i in test_run_indices]
    test_idx = [data["labels"].index(x) for x in test_label]

    dt = np.concatenate([data["time"][idx][1:] - data["time"][idx][:-1] for idx in test_idx])
    dt_mean, _ = np.mean(dt), np.var(dt)

    train_qp, train_qv, train_qa, train_tau = np.zeros((0, nq_dof)), np.zeros((0, nv_dof)), np.zeros((0, nv_dof)), np.zeros((0, 6))
    test_qp, test_qv, test_qa, test_tau = np.zeros((0, nq_dof)), np.zeros((0, nv_dof)), np.zeros((0, nv_dof)), np.zeros((0, 6))
    test_m, test_c, test_g = np.zeros((0, 6)), np.zeros((0, 6)), np.zeros((0, 6))

    train_hist_joint_pos, train_hist_joint_vel, train_hist_diff_tau_nom = np.zeros((0, hist_length, 12)), np.zeros((0, hist_length, 12)), np.zeros((0, hist_length, 6))
    train_hist_base_orient, train_hist_base_vel, train_hist_base_ang_vel = np.zeros((0, hist_length, 4)), np.zeros((0, hist_length, 3)), np.zeros((0, hist_length, 3))
    test_hist_joint_pos, test_hist_joint_vel, test_hist_diff_tau_nom = np.zeros((0, hist_length, 12)), np.zeros((0, hist_length, 12)), np.zeros((0, hist_length, 6))
    test_hist_base_orient, test_hist_base_vel, test_hist_base_ang_vel = np.zeros((0, hist_length, 4)), np.zeros((0, hist_length, 3)), np.zeros((0, hist_length, 3))
    
    divider = [0, ]
    test_base_mass = []
    train_labels, test_labels = [], []

    for i in range(int(len(data["labels"]))):
        if i in test_idx:
            test_labels.append(data["labels"][i])
            test_qp = np.vstack((test_qp, data["qp"][i][sample_offset:]))
            test_qv = np.vstack((test_qv, data["qv"][i][sample_offset:]))
            test_qa = np.vstack((test_qa, data["qa"][i][sample_offset:]))
            test_tau = np.vstack((test_tau, data["diff_tau"][i][sample_offset:]))

            test_m = np.vstack((test_m, data["diff_tau_m"][i][sample_offset:]))
            test_c = np.vstack((test_c, data["diff_tau_c"][i][sample_offset:]))
            test_g = np.vstack((test_g, data["diff_tau_g"][i][sample_offset:]))

            # add target to the end
            if history_input == "joint":
                test_hist_joint_pos = np.vstack((test_hist_joint_pos, data_hist["joint_pos"][i][sample_offset:]))
                test_hist_joint_vel = np.vstack((test_hist_joint_vel, data_hist["joint_vel"][i][sample_offset:]))
            else: # base
                test_hist_base_orient = np.vstack((test_hist_base_orient, data_hist["base_orient"][i][sample_offset:]))
                test_hist_base_vel = np.vstack((test_hist_base_vel, data_hist["base_vel"][i][sample_offset:]))
                test_hist_base_ang_vel = np.vstack((test_hist_base_ang_vel, data_hist["base_ang_vel"][i][sample_offset:]))

            test_hist_diff_tau_nom = np.vstack((test_hist_diff_tau_nom, data_hist["diff_tau"][i][sample_offset:]))

            divider.append(test_qp.shape[0])
            test_base_mass.append(base_mass[i])

        else:
            train_labels.append(data["labels"][i])
            train_qp = np.vstack((train_qp, data["qp"][i][sample_offset:]))
            train_qv = np.vstack((train_qv, data["qv"][i][sample_offset:]))
            train_qa = np.vstack((train_qa, data["qa"][i][sample_offset:]))

            train_tau = np.vstack((train_tau, data["diff_tau"][i][sample_offset:]))

            if history_input == "joint":
                train_hist_joint_pos = np.vstack((train_hist_joint_pos, data_hist["joint_pos"][i][sample_offset:]))
                train_hist_joint_vel = np.vstack((train_hist_joint_vel, data_hist["joint_vel"][i][sample_offset:]))
            else: # base
                train_hist_base_orient = np.vstack((train_hist_base_orient, data_hist["base_orient"][i][sample_offset:]))
                train_hist_base_vel = np.vstack((train_hist_base_vel, data_hist["base_vel"][i][sample_offset:]))
                train_hist_base_ang_vel = np.vstack((train_hist_base_ang_vel, data_hist["base_ang_vel"][i][sample_offset:]))

            train_hist_diff_tau_nom = np.vstack((train_hist_diff_tau_nom, data_hist["diff_tau"][i][sample_offset:]))

    if history_input == "joint":
        train_history = np.concatenate([train_hist_joint_pos, train_hist_joint_vel, train_hist_diff_tau_nom],axis=-1)
        test_history = np.concatenate([test_hist_joint_pos, test_hist_joint_vel, test_hist_diff_tau_nom],axis=-1)
    else:
        train_history = np.concatenate([train_hist_base_orient, train_hist_base_vel, train_hist_base_ang_vel, train_hist_diff_tau_nom], axis=-1)
        test_history = np.concatenate([test_hist_base_orient, test_hist_base_vel, test_hist_base_ang_vel, test_hist_diff_tau_nom],axis=-1)

    train_data = (train_labels, train_qp, train_qv, train_qa, train_tau, train_history)
    test_data = (test_labels, test_qp, test_qv, test_qa, test_tau, test_m, test_c, test_g, test_history)

    return train_data, test_data, divider, dt_mean, test_base_mass
