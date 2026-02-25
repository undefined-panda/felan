from typing import List, Optional
from flax import struct

import jax
from flax import linen as nn
import jax.numpy as jnp

from felan.models.math_utils import *

@struct.dataclass
class MLPConfig:
    n_output: int
    nq: int = 1
    net_arch: List[int] = struct.field(default_factory=list)
    activation_name: str = 'tanh'

class MLP(nn.Module):
    config: MLPConfig

    @nn.compact
    def __call__(self, x):
        for hidden_dim in self.config.net_arch:
            x = nn.Dense(hidden_dim, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.constant(0.01))(x)
            x = apply_act_fn(x, self.config.activation_name)

        x = nn.Dense(
            self.config.n_output,
            kernel_init=nn.initializers.normal(stddev=1e-2),
            bias_init=nn.initializers.zeros
        )(x)

        return x

@struct.dataclass
class MLPBlackBoxConfig:
    # Robot
    nq : int
    n_legs : int
    nq_leg : int
    n_arms : int = 0
    nq_arm : int = 0
    nq_torso : int = 0
    lin_vel_dof : int = 3
    ang_vel_dof : int = 3
    # Activations
    activation_name: str = 'tanh'
    # Components Config
    mlp_config: Optional[MLPConfig] = None

def get_config_from_dict(kwargs):
    lin_vel_dof = 3
    ang_vel_dof = 3

    n_legs = kwargs.get('n_legs')
    nq_leg = kwargs.get('n_dof_leg')
    n_arms = kwargs.get('n_arms', 0)
    nq_arm = kwargs.get('n_dof_arm', 0)
    nq_torso = kwargs.get('n_dof_torso', 0)
    nq = nq_torso + n_legs * nq_leg + n_arms * nq_arm
    nq_full = lin_vel_dof + ang_vel_dof + nq
    
    mlp_config = MLPConfig(n_output=nq_full,
                           nq=nq,
                           net_arch=kwargs.get('net_arch_mlp', kwargs.get('net_arch', [16, 16])),
                           activation_name=kwargs.get('activation', 'tanh'),
                        )
    
    config = MLPBlackBoxConfig(
        nq=nq,
        n_legs=n_legs,
        nq_leg=nq_leg,
        n_arms=n_arms,
        nq_arm=nq_arm,
        nq_torso=nq_torso,
        lin_vel_dof=lin_vel_dof,
        ang_vel_dof=ang_vel_dof,
        # Activations
        activation_name=kwargs.get('activation', 'tanh'),
        # Components Config
        mlp_config=mlp_config,
    )
    return config

class MLPBlackBox(nn.Module):
    n_dof: int
    config: MLPBlackBoxConfig

    def setup(self):

        self.mlp = MLP(config=self.config.mlp_config)
        
        ## Constants
        self.omega_to_euler_rate_mat = omega_B_to_euler_rate_mat
        self.euler_rate_to_omega_mat = euler_rate_to_omega_B_mat

        # Define Vmaps
        def omega_to_euler_rate_scalar(euler_angles):
            return jnp.squeeze(self.omega_to_euler_rate_mat(euler_angles[None,:]))
        self.jac_Wn_dq_method = jax.vmap(jax.jacfwd(omega_to_euler_rate_scalar))
        self.vmap_mlp = jax.vmap(self.mlp, in_axes=(0, None))

    def get_euler_rates_and_acc(self, q_full, qd_full, qdd_full):
        euler_angle = q_full[:,3:6]

        Wn = self.omega_to_euler_rate_mat(euler_angle)
        euler_rates = Wn @ (qd_full[:,3:6, None])

        v_full = jnp.concatenate([qd_full[:,:3, None],
                                  euler_rates,
                                  qd_full[:,6:, None],
                                ], axis=1)

        jac_Wn_dq = self.jac_Wn_dq_method(euler_angle)

        jac_Wn_dt_term = jnp.einsum('bijk,bk->bij', jac_Wn_dq, euler_rates.squeeze(-1))
        euler_acc = jac_Wn_dt_term @ (qd_full[:,3:6, None]) + Wn @ (qdd_full[:, 3:6, None])

        acc_full = jnp.concatenate([qdd_full[:,:3, None],
                                    euler_acc,
                                    qdd_full[:,6:, None],
                                    ], axis=1)

        return v_full, acc_full

    def dyn_model(self, q_full, qd_full, qdd_full):

        v_euler_rates, acc_euler_rates = self.get_euler_rates_and_acc(q_full, qd_full, qdd_full)
        mlp_input = jnp.concatenate([q_full[:,3:], v_euler_rates.squeeze(), acc_euler_rates.squeeze()], axis=-1)
        tau_pred = self.mlp(mlp_input)

        dEdt = jnp.sum(qd_full * tau_pred, axis=1)

        return tau_pred.squeeze(), dEdt

    def __call__(self, q, qd, qdd):
        out = self.dyn_model(q, qd, qdd)
        tau_pred = out[0]
        dEdt = out[1]
        extras = {}
        return tau_pred, dEdt, extras
