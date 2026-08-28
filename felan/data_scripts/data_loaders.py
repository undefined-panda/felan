import dill as pickle
import numpy as np
import copy

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

def add_historical_data(hist_length, data, list_key):
    
    new_data = copy.deepcopy(data)

    data_hist = {}
    for key in list_key:
        data_hist[key] = [[] for _ in range(len(data[key]))]

    for key in data.keys():
        if key in list_key:
            for index, run_data in enumerate(data[key]):
                hist_values = []
                for i in range(len(run_data)-hist_length):
                    hist_values.append(run_data[i:(i+hist_length),:])
                data_hist[key][index] = np.array(hist_values)

        # Delete data before hist_length
        if key != 'labels':
            for index, run_data in enumerate(data[key]):
                new_data[key][index] = new_data[key][index][hist_length:]
    
    return new_data, data_hist

def load_custom_dataset(filename, sample_offset = 0, dataset_use = 1.0,hist_length = 0,
                        add_noise = False, train_size=0.75, seed=None):

    raw = np.load(filename)

    qp = np.concatenate([raw['base_pos'], raw['base_orient']], axis=-1)
    base_pos_z = raw["base_pos"][..., 2:3]
    joint_pos = raw["joint_pos"]
    base_orient = raw["base_orient"]
    joint_vel = raw["joint_vel"]
    base_vel = raw["base_vel"]
    base_ang_vel = raw["base_ang_vel"]
    qv = np.concatenate([raw['base_vel'], raw['base_ang_vel']], axis=-1)
    qa = raw['base_acc']
    tau = raw['joint_torque']
    diff_tau = (raw['diff_tau_m_nom'] + raw['diff_tau_c_nom'] + raw['diff_tau_g_nom'])[..., :6]
    tau_m_all = raw['tau_m'][..., :6]
    tau_c_all = raw['tau_c'][..., :6]
    tau_g_all = raw['tau_g'][..., :6]
    diff_tau_m_all = raw['diff_tau_m_nom'][..., :6]
    diff_tau_c_all = raw['diff_tau_c_nom'][..., :6]
    diff_tau_g_all = raw['diff_tau_g_nom'][..., :6]
    time_all = raw['time']
    base_mass = raw['base_mass'].mean(axis=1)

    n_runs = qp.shape[0]
    labels_all = [f'run_{i}' for i in range(n_runs)]

    data = {
        'labels': labels_all,
        'qp': [qp[i] for i in range(n_runs)],
        'base_pos_z': [base_pos_z[i] for i in range(n_runs)],
        'joint_pos': [joint_pos[i] for i in range(n_runs)],
        'base_orient': [base_orient[i] for i in range(n_runs)],
        'qv': [qv[i] for i in range(n_runs)],
        'joint_vel': [joint_vel[i] for i in range(n_runs)],
        'base_vel': [base_vel[i] for i in range(n_runs)],
        'base_ang_vel': [base_ang_vel[i] for i in range(n_runs)],
        'qa': [qa[i] for i in range(n_runs)],
        'tau': [tau[i] for i in range(n_runs)],
        'diff_tau': [diff_tau[i] for i in range(n_runs)],
        'diff_tau_nom': [diff_tau[i] for i in range(n_runs)],
        'tau_m': [tau_m_all[i] for i in range(n_runs)],
        'tau_c': [tau_c_all[i] for i in range(n_runs)],
        'tau_g': [tau_g_all[i] for i in range(n_runs)],
        'diff_tau_m': [diff_tau_m_all[i] for i in range(n_runs)],
        'diff_tau_c': [diff_tau_c_all[i] for i in range(n_runs)],
        'diff_tau_g': [diff_tau_g_all[i] for i in range(n_runs)],
        'time': [time_all[i] for i in range(n_runs)],
    }

    if add_noise:
        # Noise scaled relative to each field's own per-run std, rather than
        # hard-coded per-dimension variances (those were sized for a 7-dof
        # dataset and silently break with a shape-mismatch on datasets whose
        # qv/qa/tau/diff_tau dimensions differ, e.g. this 6/6/12/6-dim
        # quadruped-base dataset).
        noise_rel_std = 0.01
        for run in range(len(data['qp'])):
            for key in ("qp", "qv", "qa", "tau", "diff_tau"):
                field = data[key][run]
                field_std = np.std(field, axis=0, keepdims=True)
                data[key][run] = field + np.random.normal(0, noise_rel_std * field_std, field.shape)

        print("\n################################################")
        print('Gaussian noise (1% of per-channel std) added to training data.')
        print("################################################")

    if hist_length > 0:
        data, data_hist = add_historical_data(hist_length, data, list_key=["joint_pos", "joint_vel", "diff_tau",
                                                                           "base_orient", "base_vel", "base_ang_vel", "base_pos_z"])

    # Split the dataset in train and test set:
    rng = np.random.default_rng(seed)
    n_test = int(n_runs * (1 - train_size))
    test_run_indices = rng.choice(np.arange(n_runs), n_test, replace=False)
    test_label = [labels_all[i] for i in test_run_indices]

    test_idx = [data["labels"].index(x) for x in test_label]

    dt = np.concatenate([data["time"][idx][1:] - data["time"][idx][:-1] for idx in test_idx])
    dt_mean, dt_var = np.mean(dt), np.var(dt)
    assert dt_var < 1.e-12

    train_labels, test_labels = [], []
    train_qp, train_qv, train_qa, train_tau = np.zeros((0, 7)), np.zeros((0, 6)), np.zeros((0, 6)), np.zeros((0, 6))

    test_qp, test_qv, test_qa, test_tau = np.zeros((0, 7)), np.zeros((0, 6)), np.zeros((0, 6)), np.zeros((0, 6))
    test_m, test_c, test_g = np.zeros((0, 6)), np.zeros((0, 6)), np.zeros((0, 6))

    if hist_length > 0:
        train_hist_qp, train_hist_qv, train_hist_diff_tau_nom = np.zeros((0, hist_length, 12)), np.zeros((0, hist_length, 12)), np.zeros((0, hist_length, 6))
        train_hist_base_orient, train_hist_base_vel, train_hist_base_ang_vel, train_hist_base_pos_z = np.zeros((0, hist_length, 4)), np.zeros((0, hist_length, 3)), np.zeros((0, hist_length, 3)), np.zeros((0, hist_length, 1))
        test_hist_qp, test_hist_qv, test_hist_diff_tau_nom = np.zeros((0, hist_length, 12)), np.zeros((0, hist_length, 12)), np.zeros((0, hist_length, 6))
        test_hist_base_orient, test_hist_base_vel, test_hist_base_ang_vel, test_hist_base_pos_z = np.zeros((0, hist_length, 4)), np.zeros((0, hist_length, 3)), np.zeros((0, hist_length, 3)), np.zeros((0, hist_length, 1))

    divider = [0, ]   # Contains idx between characters for plotting
    test_base_mass = []

    for i in range(int(dataset_use*len(data["labels"]))):

        if i in test_idx:
            test_labels.append(data["labels"][i])
            test_qp = np.vstack((test_qp, data["qp"][i][sample_offset:]))
            test_qv = np.vstack((test_qv, data["qv"][i][sample_offset:]))
            test_qa = np.vstack((test_qa, data["qa"][i][sample_offset:]))

            test_tau = np.vstack((test_tau, data["diff_tau"][i][sample_offset:]))

            test_m = np.vstack((test_m, data["diff_tau_m"][i][sample_offset:]))
            test_c = np.vstack((test_c, data["diff_tau_c"][i][sample_offset:]))
            test_g = np.vstack((test_g, data["diff_tau_g"][i][sample_offset:]))

            if hist_length > 0:
                test_hist_qp = np.vstack((test_hist_qp, data_hist["joint_pos"][i][sample_offset:]))
                test_hist_qv = np.vstack((test_hist_qv, data_hist["joint_vel"][i][sample_offset:]))
                test_hist_diff_tau_nom = np.vstack((test_hist_diff_tau_nom, data_hist["diff_tau"][i][sample_offset:]))
                test_hist_base_orient = np.vstack((test_hist_base_orient, data_hist["base_orient"][i][sample_offset:]))
                test_hist_base_vel = np.vstack((test_hist_base_vel, data_hist["base_vel"][i][sample_offset:]))
                test_hist_base_ang_vel = np.vstack((test_hist_base_ang_vel, data_hist["base_ang_vel"][i][sample_offset:]))
                test_hist_base_pos_z = np.vstack((test_hist_base_pos_z, data_hist["base_pos_z"][i][sample_offset:]))

            divider.append(test_qp.shape[0])
            test_base_mass.append(base_mass[i])

        else:
            train_labels.append(data["labels"][i])
            train_qp = np.vstack((train_qp, data["qp"][i][sample_offset:]))
            train_qv = np.vstack((train_qv, data["qv"][i][sample_offset:]))
            train_qa = np.vstack((train_qa, data["qa"][i][sample_offset:]))

            train_tau = np.vstack((train_tau, data["diff_tau"][i][sample_offset:]))

            if hist_length > 0:
                train_hist_qp = np.vstack((train_hist_qp, data_hist["joint_pos"][i][sample_offset:]))
                train_hist_qv = np.vstack((train_hist_qv, data_hist["joint_vel"][i][sample_offset:]))
                train_hist_diff_tau_nom = np.vstack((train_hist_diff_tau_nom, data_hist["diff_tau"][i][sample_offset:]))
                train_hist_base_orient = np.vstack((train_hist_base_orient, data_hist["base_orient"][i][sample_offset:]))
                train_hist_base_vel = np.vstack((train_hist_base_vel, data_hist["base_vel"][i][sample_offset:]))
                train_hist_base_ang_vel = np.vstack((train_hist_base_ang_vel, data_hist["base_ang_vel"][i][sample_offset:]))
                train_hist_base_pos_z = np.vstack((train_hist_base_pos_z, data_hist["base_pos_z"][i][sample_offset:]))

    if hist_length > 0:
        train_data = (train_labels, train_qp, train_qv, train_qa, train_tau, \
                    train_hist_qp, train_hist_qv, train_hist_diff_tau_nom, \
                    train_hist_base_orient, train_hist_base_vel, train_hist_base_ang_vel, train_hist_base_pos_z)
        test_data = (test_labels, test_qp, test_qv, test_qa, test_tau, test_m, test_c, test_g, \
                    test_hist_qp, test_hist_qv, test_hist_diff_tau_nom, \
                    test_hist_base_orient, test_hist_base_vel, test_hist_base_ang_vel, test_hist_base_pos_z)
    else:
        train_data = (train_labels, train_qp, train_qv, train_qa, train_tau)
        test_data = (test_labels, test_qp, test_qv, test_qa, test_tau, test_m, test_c, test_g)

    return train_data, test_data, divider, dt_mean, test_base_mass
