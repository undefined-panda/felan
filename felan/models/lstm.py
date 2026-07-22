import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.training.train_state import TrainState
import numpy as np
import matplotlib.pyplot as plt
import optax
from functools import partial
import pickle 

class LSTM(nn.Module):
    """Implementation of a LSTM for residual torque prediction of quadrupeds in JAX.

    Input: [q, dq, tau_hat]
    Output (currently): residual torque (tau_hat) of next step

    Source: https://medium.com/@lucmccutcheon.home/lstm-in-flax-simple-but-effective-0193aadbc940
    """

    features: int
    num_layers: int
    z_dim: int

    @nn.compact
    def __call__(self, x):        
        # equivalent to for loops
        ScanLSTM = nn.scan(
            nn.OptimizedLSTMCell, variable_broadcast="params",
            split_rngs={"params": False}, in_axes=1, out_axes=1
        )

        for _ in range(self.num_layers):
            lstm = ScanLSTM(self.features) # create lstm object for each layer to get seperate params
            carry = lstm.initialize_carry(random.key(0), x[:, 0].shape) # reset memory cell for next sequence
            carry, x = lstm(carry, x) # x.shape = (batch, time, features)

        x = x[:,-1]
        x = nn.Dense(features=self.z_dim)(x)
        return x

@jax.jit
def update(lstm_state, x_batch, y_batch):
    def loss_fn(params):
        # MSE as loss function
        predictions = lstm_state.apply_fn(params, x_batch)
        return jnp.mean((predictions - y_batch)**2)
    
    loss, grads = jax.value_and_grad(loss_fn)(lstm_state.params) # finds gradient of loss function wrt model params
    lstm_state = lstm_state.apply_gradients(grads=grads) # apply gradient descent
    return lstm_state, loss

@partial(jax.jit, static_argnames=["time_window", "batch_size"]) # arguments do not change during compile time
def sample_batch(key, q, dq, tau_hat, time_window: int, batch_size: int):
    num_runs, num_datapoints = q.shape[0], q.shape[1]
    max_start = num_datapoints - time_window - 1

    k1, k2 = random.split(key)
    run_idx = random.randint(k1, (batch_size,), 0, num_runs)
    start_idx = random.randint(k2, (batch_size,), 0, max_start)

    def one_sample(run, start):
        joint_pos = jax.lax.dynamic_slice_in_dim(q[run], start, time_window, axis=0)
        joint_vel = jax.lax.dynamic_slice_in_dim(dq[run], start, time_window, axis=0)
        res_tau   = jax.lax.dynamic_slice_in_dim(tau_hat[run], start, time_window, axis=0)

        x = jnp.concatenate([joint_pos, joint_vel, res_tau], axis=-1)
        y = tau_hat[run, start + time_window, :]
        return x, y
    
    X, y = jax.vmap(one_sample)(run_idx, start_idx)
    return X, y

def extract_values(data):
    joint_pos = np.array(data["q"])[..., 7:]
    joint_vel = np.array(data["qd"])[..., 6:]

    tau_m = np.array(data["tau_m"])
    tau_dyn = np.array(data["tau_dyn"])
    tau_ext_total = np.array(data["tau_ext_total"])

    residual_torque = (tau_m - tau_dyn - tau_ext_total)[..., 6:]

    return joint_pos, joint_vel, residual_torque

def create_train_state(data, key, features, num_layers, z_dim):
    # create quick example batch, pass through model to set some params
    q, dq, tau_hat = extract_values(data)
    # x_batch, _ = generate_batch(rng, x_range, batch_size)
    x_batch, _ = sample_batch(key, q, dq, tau_hat, time_window, batch_size)
    model = LSTM(features=features, num_layers=num_layers, z_dim=z_dim)
    params = model.init(key, x_batch)
    tx = optax.adam(1e-3) # optimizer
    return TrainState.create(params=params, apply_fn=jax.jit(model.apply), tx=tx)

def train(lstm_state, data, key, time_window, batch_size, epochs, log_step=1000):
    losses = []
    q, dq, tau_hat = extract_values(data)

    for epoch in range(epochs):
        rng, key = random.split(key)
        # x_batch, y_batch = generate_batch(key, x_range, batch_size)
        x_batch, y_batch = sample_batch(key, q, dq, tau_hat, time_window, batch_size)
        lstm_state, loss = update(lstm_state, x_batch, y_batch)
        losses.append(loss)
        if epoch % log_step == 0 and epoch > 0:
            print(f"epoch {epoch}: mean loss (last {log_step}) = {np.mean(losses[-log_step:]):.6f}")
    
    return lstm_state

if __name__ == "__main__":
    rng = random.PRNGKey(0)
    batch_size = 64
    time_window = 20 # sequence length
    num_layers = 5
    hidden_size = 10
    z_dim = 12
    x_range = 2*jnp.pi
    epochs = 100
    log_step = 10
    
    with open('felan/data/datasets/go2_sim_n_runs_50_data_freq_100hz.pkl', 'rb') as f: 
        data = pickle.load(f)
        print("File loaded")

    rng, key = random.split(rng)
    lstm_state = create_train_state(data=data, key=key, features=hidden_size, num_layers=num_layers, z_dim=z_dim)
    print("LSTM state created")

    rng, key = random.split(rng)
    print("Start training")
    lstm_state = train(lstm_state, data, key, time_window, batch_size, epochs, log_step)
    print("End training")
