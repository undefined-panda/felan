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