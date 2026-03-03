from typing import List, Optional, Tuple
import jax
import jax.numpy as jnp
from flax import struct, linen as nn

from felan.models.math_utils import *

@struct.dataclass
class ComponentConfig:
    n_output: int
    nq: int = 1
    net_arch: List[int] = struct.field(default_factory=list)
    init_tf: bool = True
    activation_name: str = 'tanh'
    epsilon: float = 1e-5
    shift: float = 0.0
    act_ld_name: str = 'Softplus'
    softplus_beta: float = 1.0
    # Inertia dimensions
    l_output_size: int = 0
    l_diag_size: int = 0
    l_lower_size: int = 0
    idx_nq: jnp.ndarray = 0
    tril_indices_nq: Tuple[jnp.ndarray, jnp.ndarray] = struct.field(default_factory=tuple)

class ComponentNN(nn.Module):
    config: ComponentConfig

    @nn.compact
    def __call__(self, x = None):
        # Apply Input Transformation
        if self.config.init_tf:
            x = apply_input_tf(x)

        for hidden_dim in self.config.net_arch:
            x = nn.Dense(hidden_dim, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.constant(0.01))(x)
            x = apply_act_fn(x, self.config.activation_name)
        
        x = nn.Dense(
            self.config.n_output,
            kernel_init=nn.initializers.normal(stddev=1e-2),
            bias_init=nn.initializers.zeros
        )(x)

        return x
    
class PotNN(nn.Module):
    config: ComponentConfig

    @nn.compact
    def __call__(self, x = None):

        # Apply Input Transformation
        if self.config.init_tf:
            x_lin = x[:,:3]
            x_tf = apply_input_tf(x[:,3:])
            x = jnp.concatenate([x_lin, x_tf], axis=-1)

        for hidden_dim in self.config.net_arch:
            x = nn.Dense(hidden_dim, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.constant(0.01))(x)
            x = apply_act_fn(x, self.config.activation_name)
        x = nn.Dense(self.config.n_output, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.constant(0.01))(x)

        return x

@struct.dataclass
class DeLaNConfig:
    # Robot
    nq : int
    nq_full : int
    lin_vel_dof : int = 3
    ang_vel_dof : int = 3
    # Activations
    activation_name: str = 'tanh'
    epsilon: float = 1e-5
    shift: float = 0.0
    init_tf: bool = True
    act_ld_name: str = 'Softplus'
    softplus_beta: float = 1.0
    # Components Config
    inertia_config: Optional[ComponentConfig] = None
    pot_config: Optional[ComponentConfig] = None

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
    nq_full = lin_vel_dof + ang_vel_dof + nq

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
        net_arch=kwargs.get('net_arch_pot', kwargs.get('net_arch_pot', [16, 16])),
        init_tf=kwargs.get('init_tf', True),
        activation_name=kwargs.get('activation', 'tanh'),
    )

    config = DeLaNConfig(
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
    return config

def get_L_from_output(L_output, config):
    l_diagonal, l_off_diagonal = jnp.split(L_output, [config.l_diag_size], axis=-1)
    l_diagonal = apply_act_fn(l_diagonal + config.shift, config.act_ld_name, config.softplus_beta) + config.epsilon

    # Reassemble L vector
    L_vec = jnp.concatenate([l_diagonal, l_off_diagonal], axis=-1)
    L_vec = L_vec[..., config.idx_nq]

    # Create L matrix
    L = jnp.zeros((config.nq, config.nq), dtype=jnp.float32)
    L = L.at[config.tril_indices_nq[0], config.tril_indices_nq[1]].set(L_vec)
    return L


class DeLaN(nn.Module):
    n_dof: int
    config: DeLaNConfig

    def setup(self):

        self.inertia_net = ComponentNN(config=self.config.inertia_config)

        # Init Potential Network
        if self.config.pot_config is not None:
            self.potential_net = PotNN(config=self.config.pot_config)
        
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

        def lagrangian_scalar(q, qd):
            return jnp.squeeze(self.lagrangian_euler_rates_vb_fn(q[None, :], qd[None, :]))
        self.vmap_dLdq_fn = jax.vmap(jax.jacrev(lagrangian_scalar, argnums=0))
        self.vmap_d2L_fn = jax.vmap(jax.jacfwd(jax.jacrev(lagrangian_scalar, argnums=1), argnums=(0, 1)))

 
    
    def get_inertia_matrix(self, q_full):
        # q_full: base + actuated joints
        # q_base: base
        # q: actuated joints
        _, q = split_by_lengths(q_full, [6, self.config.nq], axis=1)

        Lfinal = self.vmap_get_L_from_output(self.inertia_net(q), self.inertia_net.config)

        # Final mass matrix: H = LL' (standard Cholesky)
        H = Lfinal @ jnp.transpose(Lfinal, (0, 2, 1))
        mass = jnp.trace(H[:,:3,:3], axis1=1, axis2=2) / 3.0
        return H, mass

    def convert_inertia_to_vw_euler_rate(self, q_full, inertia_mat, rot_base_to_world, Wn_inv):

        n_batch = q_full.shape[0]
        rot_map = Wn_inv

        z3_3 = jnp.zeros((n_batch, 3, 3))
        z3_nq = jnp.zeros((n_batch, 3, self.config.nq))
        znq_6 = jnp.zeros((n_batch, self.config.nq, 6))
        eye_nq = jnp.broadcast_to(jnp.eye(self.config.nq), (n_batch, self.config.nq, self.config.nq))

        row1 = jnp.concatenate([
            jnp.transpose(rot_base_to_world, (0, 2, 1)),
            z3_3,
            z3_nq,
        ], axis=-1)

        row2 = jnp.concatenate([
            z3_3,
            rot_map,
            z3_nq,
        ], axis=-1)

        row3 = jnp.concatenate([
            znq_6,
            eye_nq,
        ], axis=-1)

        # Stack rows into Tmap: [B, D, D]
        Tmap = jnp.concatenate([row1, row2, row3], axis=-2)

        # Now do: T' @ I @ T
        Tmap_T = jnp.transpose(Tmap, (0, 2, 1))

        out = Tmap_T @ inertia_mat @ Tmap
        return out

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
    
    def gen_force_to_angular_frame(self, q_full, gen_force):
        gen_ang_force = gen_force[:,3:6,:]
        Wn = self.omega_to_euler_rate_mat(q_full[:,3:6])

        ang_torque = jnp.transpose(Wn, (0, 2, 1)) @ gen_ang_force
        gen_force = gen_force.at[:, 3:6, :].set(ang_torque)

        return gen_force

    def kinetic_energy(self, qd: jnp.ndarray, H) -> jnp.ndarray:
        return 0.5 * jnp.matmul(jnp.transpose(qd, (0, 2, 1)), H @ qd).squeeze(-1)

    def potential_energy_NN(self, q: jnp.ndarray):
        # Init Potential Network
        V = self.potential_net(q)
        return V

    def lagrangian_euler_rates_vb_fn(self, q_full, qd_full):
        
        rot_base_to_world = get_rot_base_to_world(q_full[:,3:6])
        Wn_inv = self.euler_rate_to_omega_mat(q_full[:,3:6])

        inertia_mat_raw, mass = self.get_inertia_matrix(q_full)

        inertia_mat = self.convert_inertia_to_vw_euler_rate(q_full, inertia_mat_raw, rot_base_to_world, Wn_inv)
       
        # Compute Energies
        e_kin = self.kinetic_energy(qd_full, inertia_mat)
        e_pot = self.potential_energy_NN(q_full)

        return e_kin - e_pot

    def dyn_model_hessian(self, q_full, qd_full, qdd_full = None):

        v_euler_rates, acc_euler_rates = self.get_euler_rates_and_acc(q_full, qd_full, qdd_full)

        dLdq = self.vmap_dLdq_fn(q_full, v_euler_rates)[:, :, None]
        d2L_dqddq, d2Ld2qd = self.vmap_d2L_fn(q_full, v_euler_rates) 

        # Compute the predicted generalized force:
        tau_pred = jnp.matmul(d2Ld2qd.squeeze(), acc_euler_rates) + jnp.matmul(d2L_dqddq.squeeze(), v_euler_rates) - dLdq

        tau_pred = self.gen_force_to_angular_frame(q_full, tau_pred).squeeze()

        dEdt = jnp.sum(qd_full * tau_pred, axis=1)

        return tau_pred, dEdt

    def __call__(self, q, qd, qdd):
        out = self.dyn_model(q, qd, qdd)
        tau_pred = out[0]
        dEdt = out[1]
        extras = {}
        return tau_pred, dEdt, extras
