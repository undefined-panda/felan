# Legged Robot Localization under Uncertain Dynamics

This repo implements a physics-encoded neural network for the _**Robot Learning: Integrated Project Part 2**_ at TU Darmstadt. The overall goal is to estimate the state of a quadruped when the dynamics are uncertain, e.g. due to an additional payload.

This repo was forked from [felan](https://github.com/Schulze18/felan) and extended with an implementation of a **Context-Aware Deep Lagrangian Network** (CaDeLaN). It is used to predict the residual torque, inertia matrix and bias forces which are then applied to the Rigid-Body-Dynamics of quadrupeds inside the state estimator.

## Setup
Clone the repo and then create a conda environment:
```
conda env create -f felan_env.yml
conda activate felan
```
Install with pip:
```
pip install -e .
```

## IP 2 Contribution
The goal of IP2 is to develop a method that improves the state estimation of the Kalman Filter for uncertain dynamics, when carrying an unknown payload. These payloads are simulated by sampling an offset for the base mass. This repo combines a context encoder with a physics-encoded neural network (PENN) that derives the payload from the most recent motion history, and returns not only the residual torque but also the residual inertia matrix and residual bias forces, which can be fed directly into the rigid-body dynamics of the Kalman Filter.

**LSTM-Encoder**: An _LSTM_ is fed with a history of joint or base values to produce an environmental context vector $z \in \mathbb{R}^{10}$. The default architecture has 5 layers with a hidden size of 10 each, followed by a dense layer. As a standalone model, the network outputs the residual torque directly, with a head size of 6.

**DeLaN**: The used PENN is a _Deep-Lagrangian Network_ with the same structure as `delan_pot_param.py`, but the inertia network receives $z$ instead of the joint angles. Since this project only works with floating base (`nq = 0, nq_full = 6`), the spatial inertia only depends idk. At the end we have a 6x6 inertia matrix.

This architecture is based on _CaDeLaC_, which implements a Context-Aware DeLaN for MPC. As no MPC is used here, we have only a **Context-Aware DeLaN** (CaDeLaN).

#### Log-Cholesky Parametrization
The standard Cholesky decomposition $M = LL^\top$ guarantees $M \succeq 0$, not that the 6×6 matrix represents a physically realisable rigid-body inertia matrix. For this reason, the log-Cholesky form of the pseudo-inertia matrix is used instead.

The inertia net returns 10 parameters, $\theta = \begin{bmatrix}\alpha & d_1 & d_2 & d_3 & s_{12} & s_{13} & s_{23} & t_1 & t_2 & t_3\end{bmatrix}$, which are used to build a log-Cholesky parametrization of the pseudo inertia $\mathbf{J = UU^T}$ with
$
\begin{equation}
    \mathbf{U} = e^{\alpha}
    \begin{bmatrix}
        e^{d_1} & s_{12} & s_{13} & t_1\\
        0 & e^{d_2} & s_{23} & t_2 \\
        0 & 0 & e^{d_3} & t_3 \\
        0 & 0 & 0 & 1
    \end{bmatrix}
\end{equation}
$
The pseudo-inertia matrix is defined as:
$
\begin{equation}
    \mathbf{J} =
    \begin{bmatrix}
        \Sigma & \mathbf{h} \\ 
        \mathbf{h}^\top & m
    \end{bmatrix},
    \quad \text{where} \quad \Sigma = \frac{1}{2}\text{Tr}(\mathbf{I})\mathbf{1}_3 - \mathbf{I}
\end{equation}
$

$U$ is an upper triangular matrix with a strictly positive diagonal, so $J \succ 0$ by construction — for every network output.

## Training

Run the training script with:
```
python -m felan.train_quad_cadelan
```
The trained are models are stored in [/trained_models](/felan/trained_models/aliengo/).

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--epochs` | `3000` | Number of training epochs |
| `--nn` | `LSTM` | Network architecture: `LSTM`, `CaDeLaN`, or `LogCholCaDeLaN` |
| `--lstm_num_layers` | `5` | Number of stacked LSTM layers in the context encoder |
| `--lstm_hidden_size` | `10` | Hidden size of each LSTM layer |
| `--delan_size` | `[16, 16]` | Layer sizes of the DeLaN network (list of ints) |
| `--history_span` | `0.5` | Wall-clock span of the LSTM context window [s] |
| `--history_stride` | `4` | Spacing between context-window samples [samples] |
| `--history_input` | `joint` | Input data for the history: `joint` or `base` |

### Example

```bash
python -m felan.train_quad_cadelan --nn CaDeLaN --epochs 5000 --history_input base --delan_size 32 32
```
