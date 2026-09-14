from typing import List, Optional, Tuple
import jax
import jax.numpy as jnp
from flax import struct, linen as nn

from felan.models.math_utils import *
from felan.models.lstm_black_box import LSTMBlackBox, LSTMBlackBoxConfig
from felan.models.delan_pot_param import ComponentNN, ComponentConfig, DeLaNPotParam, DeLaNPotParamConfig, get_config_from_dict as get_delan_pp_config, get_L_from_output

class _DeLaNPotParam(DeLaNPotParam):
    def setup(self):
        self.inertia_net = ComponentNN(config=self.config.inertia_config)

        # Init Potential Network
        if self.config.pot_config is not None:
            self.potential_net = ComponentNN(config=self.config.pot_config)
        
        ## Constants
        gravity_value = 9.81
        self.gravity_array13 = jnp.array([0.0, 0.0, gravity_value]).reshape((1,3))
        self.omega_to_euler_rate_mat = omega_B_to_euler_rate_mat
        self.euler_rate_to_omega_mat = euler_rate_to_omega_B_mat

        # Models
        self.dyn_model = self.dyn_model_hessian

        # Define Vmaps
        self.vmap_get_L_from_output = jax.vmap(get_L_from_output, in_axes=(0, None))

        def omega_to_euler_rate_scalar(euler_angles):
            return jnp.squeeze(self.omega_to_euler_rate_mat(euler_angles[None,:]))
        self.jac_Wn_dq_method = jax.vmap(jax.jacfwd(omega_to_euler_rate_scalar))

        def lagrangian_scalar(q, qd, z):
            return jnp.squeeze(self.lagrangian_euler_rates_vb_fn(q[None, :], qd[None, :], z[None, :]))
        self.vmap_dLdq_fn = jax.vmap(jax.jacrev(lagrangian_scalar, argnums=0), in_axes=(0, 0, 0))
        self.vmap_d2L_fn = jax.vmap(jax.jacfwd(jax.jacrev(lagrangian_scalar, argnums=1), argnums=(0, 1)), in_axes=(0, 0, 0))

    def get_inertia_matrix(self, z):
        Lfinal = self.vmap_get_L_from_output(
            self.inertia_net(z),
            self.inertia_net.config
        )

        # Final mass matrix: H = LL' (standard Cholesky)
        H = Lfinal @ jnp.transpose(Lfinal, (0, 2, 1))
        mass = jnp.trace(H[:,:3,:3], axis1=1, axis2=2) / 3.0
        return H, mass
    
    def convert_inertia_to_vw_euler_rate(self, q_full, inertia_mat, rot_base_to_world, Wn_inv):
        n_batch = q_full.shape[0]
        rot_map = Wn_inv
        z3_3 = jnp.zeros((n_batch, 3, 3))

        row1 = jnp.concatenate([
            jnp.transpose(rot_base_to_world, (0, 2, 1)),
            z3_3,
        ], axis=-1)

        row2 = jnp.concatenate([
            z3_3,
            rot_map,
        ], axis=-1)

        Tmap = jnp.concatenate([row1, row2], axis=-2)
        Tmap_T = jnp.transpose(Tmap, (0, 2, 1))
        
        out = Tmap_T @ inertia_mat @ Tmap
        return out
    
    def lagrangian_euler_rates_vb_fn(self, q_full, qd_full, z):
        rot_base_to_world = get_rot_base_to_world(q_full[:,3:6])
        Wn_inv = self.euler_rate_to_omega_mat(q_full[:,3:6])
        
        inertia_mat_raw, mass = self.get_inertia_matrix(z)

        inertia_mat = self.convert_inertia_to_vw_euler_rate(q_full, inertia_mat_raw, rot_base_to_world, Wn_inv)

        inertia_skew = inertia_mat_raw[:,:3,3:6]
        pot_comp = -get_vector_from_skew(inertia_skew)

        # Compute Energies
        e_kin = self.kinetic_energy(qd_full, inertia_mat)
        e_pot = self.potential_energy(q_full[:,:3], rot_base_to_world, mass, pot_comp)

        return e_kin - e_pot

    def get_euler_rates_and_acc(self, q_full, qd_full, qdd_full):
        euler_angle = q_full[:,3:6]

        Wn = self.omega_to_euler_rate_mat(euler_angle)
        euler_rates = Wn @ (qd_full[:,3:6, None])

        v_full = jnp.concatenate([qd_full[:,:3, None],
                                  euler_rates,
                                  ], axis=1)

        jac_Wn_dq = self.jac_Wn_dq_method(euler_angle)

        jac_Wn_dt_term = jnp.einsum('bijk,bk->bij', jac_Wn_dq, euler_rates.squeeze(-1))
        euler_acc = jac_Wn_dt_term @ (qd_full[:,3:6, None]) + Wn @ (qdd_full[:, 3:6, None])

        acc_full = jnp.concatenate([qdd_full[:,:3, None],
                                    euler_acc,
                                    ], axis=1)

        return v_full, acc_full

    def dyn_model_hessian(self, q_full, qd_full, qdd_full, z):
        v_euler_rates, acc_euler_rates = self.get_euler_rates_and_acc(q_full, qd_full, qdd_full)

        dLdq = self.vmap_dLdq_fn(q_full, v_euler_rates, z)[:, :, None]
        d2L_dqddq, d2Ld2qd = self.vmap_d2L_fn(q_full, v_euler_rates, z) 

        # Compute the predicted generalized force:
        tau_pred = jnp.matmul(d2Ld2qd.squeeze(), acc_euler_rates) + jnp.matmul(d2L_dqddq.squeeze(), v_euler_rates) - dLdq

        tau_pred = self.gen_force_to_angular_frame(q_full, tau_pred).squeeze()

        dEdt = jnp.sum(qd_full * tau_pred, axis=1)

        return tau_pred, dEdt
    
    def __call__(self, q, qd, qdd, z):
        out = self.dyn_model(q, qd, qdd, z)
        tau_pred = out[0]
        dEdt = out[1]
        extras = {}
        return tau_pred, dEdt, extras

@struct.dataclass
class CaDelaNConfig:
    lstm_config: LSTMBlackBoxConfig
    delan_config: DeLaNPotParamConfig

def get_config_from_dict(kwargs):
    lin_vel_dof = 3
    ang_vel_dof = 3
    pot_config = None

    n_legs = kwargs.get('n_legs')
    nq_leg = kwargs.get('n_dof_leg')
    n_arms = kwargs.get('n_arms', 0)
    nq_arm = kwargs.get('n_dof_arm', 0)
    nq_torso = kwargs.get('n_dof_torso', 0)
    nq = nq_torso + n_legs * nq_leg + n_arms * nq_arm
    nq_full = lin_vel_dof + ang_vel_dof 

    ## Inertia Net
    l_output_size = int((nq_full ** 2 + nq_full) / 2)
    l_diag_size = nq_full
    l_lower_size = l_output_size - l_diag_size

    l_tril_indices_nq = jnp.tril_indices(nq_full)
    l_idx_nq = get_idx_triangular(nq_full, l_output_size)

    inertia_config = ComponentConfig(
        n_output=l_output_size,
        nq=nq_full,
        net_arch=kwargs.get('net_arch_inertia_full', [16, 16]),
        init_tf=kwargs.get('init_tf', True),
        epsilon=kwargs.get('diagonal_epsilon', 1e-5),
        shift=kwargs.get('diagonal_shift', 0.0),
        act_ld_name=kwargs.get('act_ld', 'Softplus'),
        softplus_beta=kwargs.get('softplus_beta', 1.0),
        activation_name=kwargs.get('activation', 'tanh'),
        l_output_size=l_output_size,
        l_diag_size=l_diag_size,
        l_lower_size=l_lower_size,
        tril_indices_nq=l_tril_indices_nq,
        idx_nq=l_idx_nq,
    )
    

    # Need Potential Network
    pot_config = ComponentConfig(
        n_output=1,
        net_arch=kwargs.get('net_arch_pot', kwargs.get('net_arch', [16, 16])),
        init_tf=kwargs.get('init_tf', True),
        activation_name=kwargs.get('activation', 'tanh'),
    )

    delan_config = DeLaNPotParamConfig(
        nq=nq,
        nq_full=nq_full,
        # Activations
        activation_name=kwargs.get('activation', 'tanh'),
        epsilon=kwargs.get('diagonal_epsilon', 1e-5),
        shift=kwargs.get('diagonal_shift', 0.0),
        init_tf=kwargs.get('init_tf', True),
        act_ld_name=kwargs.get('act_ld', 'Softplus'),
        softplus_beta=kwargs.get('softplus_beta', 1.0),
        # Components Config
        inertia_config=inertia_config,
        pot_config=pot_config,
    )
    
    lstm_config = LSTMBlackBoxConfig(
        n_output=kwargs.get('n_output'),
        hidden_size=kwargs.get('lstm_hidden_size', 10),
        num_layers=kwargs.get('lstm_num_layers', 5),
        dropout_rate=kwargs.get('lstm_dropout', 0.0),
    )

    return CaDelaNConfig(lstm_config=lstm_config, delan_config=delan_config)

class CaDeLaN(nn.Module):
    n_dof: int
    config: CaDelaNConfig

    def setup(self):
        self.lstm  = LSTMBlackBox(n_dof=self.n_dof, config=self.config.lstm_config)
        self.delan = _DeLaNPotParam(n_dof=self.n_dof, config=self.config.delan_config)

    def __call__(self, q, qd, qdd, history):
        z, _, _ = self.lstm(q, qd, qdd, history)
        tau_pred, dEdt, extras = self.delan(q, qd, qdd, z)
        return tau_pred, dEdt, extras
