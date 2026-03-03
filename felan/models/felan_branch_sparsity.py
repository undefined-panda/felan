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
    def __call__(self, x):
        
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

@struct.dataclass
class FeLaNBranchSparsityConfig:
    # Robot
    nq : int
    n_legs : int
    nq_leg : int
    n_arms : int = 0
    nq_arm : int = 0
    nq_torso : int = 0
    lin_vel_dof : int = 3
    ang_vel_dof : int = 3
    robot_type : str = 'humanoid'
    # Activations
    activation_name: str = 'tanh'
    epsilon: float = 1e-5
    shift: float = 0.0
    init_tf: bool = True
    act_ld_name: str = 'Softplus'
    softplus_beta: float = 1.0
    mass_epsilon: float = 1.0
    # Components Config
    base_config: Optional[ComponentConfig] = None
    torso_arm_config: Optional[ComponentConfig] = None
    torso_config: Optional[ComponentConfig] = None
    arm_config: Optional[ComponentConfig] = None
    leg_config: Optional[ComponentConfig] = None
    pot_config: Optional[ComponentConfig] = None

def get_config_from_dict(kwargs):
    lin_vel_dof = 3
    ang_vel_dof = 3
    torso_arm_config = None
    torso_config = None
    arm_config = None
    pot_config = None

    n_legs = kwargs.get('n_legs')
    nq_leg = kwargs.get('n_dof_leg')
    n_arms = kwargs.get('n_arms', 0)
    nq_arm = kwargs.get('n_dof_arm', 0)
    nq_torso = kwargs.get('n_dof_torso', 0)
    nq = nq_torso + n_legs * nq_leg + n_arms * nq_arm

    ## Base Net
    nq_base = ang_vel_dof + lin_vel_dof
    lbase_output_size = int((nq_base ** 2 + nq_base) / 2)
    lbase_diag_size = nq_base
    lbase_lower_size = lbase_output_size - lbase_diag_size

    lbase_tril_indices_nq = jnp.tril_indices(nq_base)
    lbase_idx_nq = get_idx_triangular(nq_base, lbase_output_size)

    base_config = ComponentConfig(
                            n_output=lbase_output_size,
                            nq=nq_base,
                            net_arch=kwargs.get('net_arch_base', [16, 16]),
                            init_tf=kwargs.get('init_tf', True),
                            epsilon=kwargs.get('diagonal_epsilon', 1e-5),
                            shift=kwargs.get('diagonal_shift', 0.0),
                            act_ld_name=kwargs.get('act_ld', 'Softplus'),
                            softplus_beta=kwargs.get('softplus_beta', 1.0),
                            activation_name=kwargs.get('activation', 'tanh'),
                            l_output_size=lbase_output_size,
                            l_diag_size=lbase_diag_size,
                            l_lower_size=lbase_lower_size,
                            tril_indices_nq=lbase_tril_indices_nq,
                            idx_nq=lbase_idx_nq,
                    )
    

    if n_arms == 2:
        robot_type = 'humanoid'
    elif n_arms == 1:
        robot_type = 'quad_with_arm'
    elif n_arms == 0:
        robot_type = 'quad'
    else:
        raise ValueError(f"Invalid number of arms: {n_arms}. Expected 0, 1, or 2.")

    if robot_type == 'humanoid':
        ## Torso Config
        ltorso_output_size = int((nq_torso ** 2 + nq_torso) / 2)
        ltorso_diag_size = nq_torso
        ltorso_lower_size = ltorso_output_size - ltorso_diag_size
        # UT, RT, LT
        ltorso_full_output_size = nq_torso * (lin_vel_dof + ang_vel_dof) + ltorso_output_size

        ltorso_tril_indices_nq = jnp.tril_indices(nq_torso)
        ltorso_idx_nq = get_idx_triangular(nq_torso, ltorso_output_size)

        torso_config = ComponentConfig(
                                    n_output=ltorso_full_output_size,
                                    nq=nq_torso,
                                    net_arch=kwargs.get('net_arch_torso', [16, 16]),
                                    init_tf=kwargs.get('init_tf', True),
                                    epsilon=kwargs.get('diagonal_epsilon', 1e-5),
                                    shift=kwargs.get('diagonal_shift', 0.0),
                                    act_ld_name=kwargs.get('act_ld', 'Softplus'),
                                    softplus_beta=kwargs.get('softplus_beta', 1.0),
                                    activation_name=kwargs.get('activation', 'tanh'),
                                    l_output_size=ltorso_output_size,
                                    l_diag_size=ltorso_diag_size,
                                    l_lower_size=ltorso_lower_size,
                                    tril_indices_nq=ltorso_tril_indices_nq,
                                    idx_nq=ltorso_idx_nq,
                            )

        ## Arm Config
        larm_output_size = int((nq_arm ** 2 + nq_arm) / 2)
        larm_diag_size = nq_arm
        larm_lower_size = larm_output_size - larm_diag_size

        # LAi
        larm_full_output_size = larm_output_size

        larm_tril_indices_nq = jnp.tril_indices(nq_arm)
        larm_idx_nq = get_idx_triangular(nq_arm, larm_output_size)

        arm_config = ComponentConfig(
                            n_output=larm_full_output_size,
                            nq=nq_arm,
                            net_arch=kwargs.get('net_arch_arm', [16, 16]),
                            init_tf=kwargs.get('init_tf', True),
                            epsilon=kwargs.get('diagonal_epsilon', 1e-5),
                            shift=kwargs.get('diagonal_shift', 0.0),
                            act_ld_name=kwargs.get('act_ld', 'Softplus'),
                            softplus_beta=kwargs.get('softplus_beta', 1.0),
                            activation_name=kwargs.get('activation', 'tanh'),
                            l_output_size=larm_output_size,
                            l_diag_size=larm_diag_size,
                            l_lower_size=larm_lower_size,
                            tril_indices_nq=larm_tril_indices_nq,
                            idx_nq=larm_idx_nq,
                    )
        
        ## Torso Arm Config
        ltorso_arm_full_output_size = nq_arm * (lin_vel_dof + ang_vel_dof + nq_torso)
        torso_arm_config = ComponentConfig(
                            n_output=ltorso_arm_full_output_size,
                            nq=nq_torso+nq_arm,
                            net_arch=kwargs.get('net_arch_torso_arm', [16, 16]),
                            init_tf=kwargs.get('init_tf', True),
                            epsilon=kwargs.get('diagonal_epsilon', 1e-5),
                            shift=kwargs.get('diagonal_shift', 0.0),
                            act_ld_name=kwargs.get('act_ld', 'Softplus'),
                            softplus_beta=kwargs.get('softplus_beta', 1.0),
                            activation_name=kwargs.get('activation', 'tanh'),
                    )

        ## Leg Config
        lleg_output_size = int((nq_leg ** 2 + nq_leg) / 2)
        lleg_diag_size = nq_leg
        lleg_lower_size = lleg_output_size - lleg_diag_size
        # LLiL, LLiR, LLi
        lleg_full_output_size = nq_leg * (lin_vel_dof + ang_vel_dof) + lleg_output_size

        lleg_tril_indices_nq = jnp.tril_indices(nq_leg)
        lleg_idx_nq = get_idx_triangular(nq_leg, lleg_output_size)

        leg_config = ComponentConfig(
                                    n_output=lleg_full_output_size,
                                    nq=nq_leg,
                                    net_arch=kwargs.get('net_arch_leg', [16, 16]),
                                    init_tf=kwargs.get('init_tf', True),
                                    epsilon=kwargs.get('diagonal_epsilon', 1e-5),
                                    shift=kwargs.get('diagonal_shift', 0.0),
                                    act_ld_name=kwargs.get('act_ld', 'Softplus'),
                                    softplus_beta=kwargs.get('softplus_beta', 1.0),
                                    activation_name=kwargs.get('activation', 'tanh'),
                                    l_output_size=lleg_output_size,
                                    l_diag_size=lleg_diag_size,
                                    l_lower_size=lleg_lower_size,
                                    tril_indices_nq=lleg_tril_indices_nq,
                                    idx_nq=lleg_idx_nq,
                            )
    elif robot_type == 'quad_with_arm':

        ## Arm Config
        larm_output_size = int((nq_arm ** 2 + nq_arm) / 2)
        larm_diag_size = nq_arm
        larm_lower_size = larm_output_size - larm_diag_size

        # LAi
        larm_full_output_size = nq_arm * (lin_vel_dof + ang_vel_dof) + larm_output_size

        larm_tril_indices_nq = jnp.tril_indices(nq_arm)
        larm_idx_nq = get_idx_triangular(nq_arm, larm_output_size)

        arm_config = ComponentConfig(
                            n_output=larm_full_output_size,
                            nq=nq_arm,
                            net_arch=kwargs.get('net_arch_arm', [16, 16]),
                            init_tf=kwargs.get('init_tf', True),
                            epsilon=kwargs.get('diagonal_epsilon', 1e-5),
                            shift=kwargs.get('diagonal_shift', 0.0),
                            act_ld_name=kwargs.get('act_ld', 'Softplus'),
                            softplus_beta=kwargs.get('softplus_beta', 1.0),
                            activation_name=kwargs.get('activation', 'tanh'),
                            l_output_size=larm_output_size,
                            l_diag_size=larm_diag_size,
                            l_lower_size=larm_lower_size,
                            tril_indices_nq=larm_tril_indices_nq,
                            idx_nq=larm_idx_nq,
                    )

        lleg_output_size = int((nq_leg ** 2 + nq_leg) / 2)
        lleg_diag_size = nq_leg
        lleg_lower_size = lleg_output_size - lleg_diag_size
        # LLiL, LLiR, LLi
        lleg_full_output_size = nq_leg * (lin_vel_dof + ang_vel_dof) + lleg_output_size

        lleg_tril_indices_nq = jnp.tril_indices(nq_leg)
        lleg_idx_nq = get_idx_triangular(nq_leg, lleg_output_size)

        leg_config = ComponentConfig(
                                    n_output=lleg_full_output_size,
                                    nq=nq_leg,
                                    net_arch=kwargs.get('net_arch_leg', [16, 16]),
                                    init_tf=kwargs.get('init_tf', True),
                                    epsilon=kwargs.get('diagonal_epsilon', 1e-5),
                                    shift=kwargs.get('diagonal_shift', 0.0),
                                    act_ld_name=kwargs.get('act_ld', 'Softplus'),
                                    softplus_beta=kwargs.get('softplus_beta', 1.0),
                                    activation_name=kwargs.get('activation', 'tanh'),
                                    l_output_size=lleg_output_size,
                                    l_diag_size=lleg_diag_size,
                                    l_lower_size=lleg_lower_size,
                                    tril_indices_nq=lleg_tril_indices_nq,
                                    idx_nq=lleg_idx_nq,
                            )

    elif robot_type == 'quad':

        lleg_output_size = int((nq_leg ** 2 + nq_leg) / 2)
        lleg_diag_size = nq_leg
        lleg_lower_size = lleg_output_size - lleg_diag_size
        # LiL, LiR, Li
        lleg_full_output_size = nq_leg * (lin_vel_dof + ang_vel_dof) + lleg_output_size

        lleg_tril_indices_nq = jnp.tril_indices(nq_leg)
        lleg_idx_nq = get_idx_triangular(nq_leg, lleg_output_size)

        leg_config = ComponentConfig(
                                    n_output=lleg_full_output_size,
                                    nq=nq_leg,
                                    net_arch=kwargs.get('net_arch_leg', [16, 16]),
                                    init_tf=kwargs.get('init_tf', True),
                                    epsilon=kwargs.get('diagonal_epsilon', 1e-5),
                                    shift=kwargs.get('diagonal_shift', 0.0),
                                    act_ld_name=kwargs.get('act_ld', 'Softplus'),
                                    softplus_beta=kwargs.get('softplus_beta', 1.0),
                                    activation_name=kwargs.get('activation', 'tanh'),
                                    l_output_size=lleg_output_size,
                                    l_diag_size=lleg_diag_size,
                                    l_lower_size=lleg_lower_size,
                                    tril_indices_nq=lleg_tril_indices_nq,
                                    idx_nq=lleg_idx_nq,
                            )

    if kwargs.get('pot_net', False):
        pot_config = ComponentConfig(
            n_output=1,
            net_arch=kwargs.get('net_arch_pot', kwargs.get('net_arch', [16, 16])),
            init_tf=kwargs.get('init_tf', True),
            activation_name=kwargs.get('activation', 'tanh'),
        )

    config = FeLaNBranchSparsityConfig(
        nq=nq,
        n_legs=n_legs,
        nq_leg=nq_leg,
        n_arms=n_arms,
        nq_arm=nq_arm,
        nq_torso=nq_torso,
        lin_vel_dof=lin_vel_dof,
        ang_vel_dof=ang_vel_dof,
        robot_type=robot_type,
        # Activations
        activation_name=kwargs.get('activation', 'tanh'),
        epsilon=kwargs.get('diagonal_epsilon', 1e-5),
        shift=kwargs.get('diagonal_shift', 0.0),
        init_tf=kwargs.get('init_tf', True),
        act_ld_name=kwargs.get('act_ld', 'Softplus'),
        softplus_beta=kwargs.get('softplus_beta', 1.0),
        mass_epsilon=kwargs.get("mass_epsilon", 1.0),
        # Components Config
        base_config=base_config,
        torso_arm_config=torso_arm_config,
        torso_config=torso_config,
        arm_config=arm_config,
        leg_config=leg_config,
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


class FeLaNBranchSparsity(nn.Module):
    n_dof: int
    config: FeLaNBranchSparsityConfig

    def setup(self):

        # LL, LRL, LR
        self.lbase_net = ComponentNN(config=self.config.base_config)

        # LiL, LiR, Li
        if self.config.robot_type == 'humanoid':
            self.ltorso_net = ComponentNN(config=self.config.torso_config)
            self.ltorso_arm_left_net = ComponentNN(config=self.config.torso_arm_config)
            self.ltorso_arm_right_net = ComponentNN(config=self.config.torso_arm_config)

            self.larm_left_net = ComponentNN(config=self.config.arm_config)
            self.larm_right_net = ComponentNN(config=self.config.arm_config)

            self.lleg_left_net = ComponentNN(config=self.config.leg_config)
            self.lleg_right_net = ComponentNN(config=self.config.leg_config)

            self.get_inertia_matrix = self.get_inertia_matrix_humanoid

        elif self.config.robot_type == 'quad_with_arm':
            self.larm_net = ComponentNN(config=self.config.arm_config)

            self.lleg_LF_net = ComponentNN(config=self.config.leg_config)
            self.lleg_RF_net = ComponentNN(config=self.config.leg_config)
            self.lleg_LH_net = ComponentNN(config=self.config.leg_config)
            self.lleg_RH_net = ComponentNN(config=self.config.leg_config)

            self.get_inertia_matrix = self.get_inertia_matrix_quad_with_arm

        elif self.config.robot_type == 'quad':
            self.lleg_LF_net = ComponentNN(config=self.config.leg_config)
            self.lleg_RF_net = ComponentNN(config=self.config.leg_config)
            self.lleg_LH_net = ComponentNN(config=self.config.leg_config)
            self.lleg_RH_net = ComponentNN(config=self.config.leg_config)

            self.get_inertia_matrix = self.get_inertia_matrix_quad
        else:
            raise AssertionError(f'Robot type {self.config.robot_type} not implemented!')

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

        def lagrangian_scalar(q, qd):
            return jnp.squeeze(self.lagrangian_euler_rates_vb_fn(q[None, :], qd[None, :]))
        self.vmap_dLdq_fn = jax.vmap(jax.jacrev(lagrangian_scalar, argnums=0))
        self.vmap_d2L_fn = jax.vmap(jax.jacfwd(jax.jacrev(lagrangian_scalar, argnums=1), argnums=(0, 1)))

    def get_base_nn_output(self, q):
        Ubase = self.vmap_get_L_from_output(self.lbase_net(q), self.lbase_net.config)
        LL = Ubase[:,:3,:3]
        LRL = Ubase[:,3:6,:3]
        LR = Ubase[:,3:6,3:6]
        return LL, LRL, LR
    
    def get_torso_nn_output(self, qtorso):
        output_lengths = [self.config.nq_torso * self.config.lin_vel_dof,
                          self.config.nq_torso * self.config.ang_vel_dof,
                          self.ltorso_net.config.l_output_size]

        LiL, LiR, Li = split_by_lengths(self.ltorso_net(qtorso),
                                      output_lengths,
                                      axis=1)
        Li = self.vmap_get_L_from_output(Li, self.ltorso_net.config)
        return LiL.reshape(-1, self.ltorso_net.config.nq, self.config.lin_vel_dof), LiR.reshape(-1, self.ltorso_net.config.nq, self.config.ang_vel_dof), Li

    def get_arm_nn_output(self, qtorso, qarm, arm_net, torso_arm_net):
        Li = arm_net(qarm)
        Li = self.vmap_get_L_from_output(Li, arm_net.config)

        output_lengths = [self.config.nq_arm * self.config.lin_vel_dof,
                          self.config.nq_arm * self.config.ang_vel_dof,
                          self.config.nq_arm * self.config.nq_torso]

        q_input = jnp.concatenate([qtorso,
                                     qarm], axis=1)
        LAiL, LAiR, LTAi = split_by_lengths(torso_arm_net(q_input), 
                                         output_lengths, 
                                         axis=1)
        return LAiL.reshape(-1, self.config.nq_arm, self.config.lin_vel_dof), LAiR.reshape(-1, self.config.nq_arm, self.config.ang_vel_dof), LTAi.reshape(-1, self.config.nq_arm, self.config.nq_torso), Li

    def get_leg_nn_output(self, qleg, leg_net):
        output_lengths = [leg_net.config.nq * self.config.lin_vel_dof,
                          leg_net.config.nq * self.config.ang_vel_dof,
                          leg_net.config.l_output_size]

        LiL, LiR, Li = split_by_lengths(leg_net(qleg), 
                                         output_lengths, 
                                         axis=1)
        Li = self.vmap_get_L_from_output(Li, leg_net.config)
        return LiL.reshape(-1, leg_net.config.nq, self.config.lin_vel_dof), LiR.reshape(-1, leg_net.config.nq, self.config.ang_vel_dof), Li

    def get_inertia_matrix_humanoid(self, q_full):
        # q_full: base + actuated joints
        # q_base: base
        # q: actuated joints
        _, q = split_by_lengths(q_full, [6, self.config.nq], axis=1)
        q_torso_arms, qleg_left, qleg_right = split_by_lengths(q, [self.config.nq_torso + self.config.n_arms*self.config.nq_arm,
                                                                   self.config.nq_leg,
                                                                   self.config.nq_leg])

        qtorso, qarm_left, qarm_right = split_by_lengths(q_torso_arms, [self.config.nq_torso,
                                                                        self.config.nq_arm,
                                                                        self.config.nq_arm])

        # Neural Networks outputs
        LL, LRL, LR = self.get_base_nn_output(q)
        # T, A0, ..., L1 correspond to the index i in LiL, LiR, and Li.
        LTL, LTR, LT = self.get_torso_nn_output(q_torso_arms)
        LA0L, LA0R, LTA0, LA0 = self.get_arm_nn_output(qtorso, qarm_left, self.larm_left_net, self.ltorso_arm_left_net)
        LA1L, LA1R, LTA1, LA1 = self.get_arm_nn_output(qtorso, qarm_right, self.larm_right_net, self.ltorso_arm_right_net)
        LL0L, LL0R, LL0 = self.get_leg_nn_output(qleg_left, self.lleg_left_net)
        LL1L, LL1R, LL1 = self.get_leg_nn_output(qleg_right, self.lleg_right_net)

        #### LR ####
        K = jnp.concatenate([
            LTL,
            LA0L,
            LA1L,
            LL0L,
            LL1L,
        ], axis=1)

        W = jnp.concatenate([
            LTR,
            LA0R,
            LA1R,
            LL0R,
            LL1R,
        ], axis=1)

        #### U ####
        U = jnp.concatenate([
            LRL,
            K,
        ], axis=1)

        ########### Build L ###########
        n_batch = LL.shape[0]

        # Linear Velocity Column
        col_lin = jnp.concatenate([
            LL,
            U,
        ], axis=1)

        # Angular Velocity Column
        col_ang = jnp.concatenate([
            jnp.zeros((n_batch, 3, 3)),
            LR,
            W,
        ], axis=1)

        # Torso Column
        col_torso = jnp.concatenate([
            jnp.zeros((n_batch, 3 + self.config.ang_vel_dof, self.config.nq_torso)),
            LT,
            LTA0,
            LTA1,
            jnp.zeros((n_batch, 2*self.config.nq_leg, self.config.nq_torso)),
        ], axis=1)

        # Arm Left Column
        col_arm_left = jnp.concatenate([
            jnp.zeros((n_batch, 3 + self.config.ang_vel_dof + self.config.nq_torso, self.config.nq_arm)),
            LA0,
            jnp.zeros((n_batch, self.config.nq_arm + 2*self.config.nq_leg, self.config.nq_arm)),
        ], axis=1)

        # Arm Right Column
        col_arm_right = jnp.concatenate([
            jnp.zeros((n_batch, 3 + self.config.ang_vel_dof + self.config.nq_torso + self.config.nq_arm, self.config.nq_arm)),
            LA1,
            jnp.zeros((n_batch, 2*self.config.nq_leg, self.config.nq_arm)),
        ], axis=1)

        # Leg Left Column
        col_leg_left = jnp.concatenate([
            jnp.zeros((n_batch, 3 + self.config.ang_vel_dof + self.config.nq_torso + 2*self.config.nq_arm, self.config.nq_leg)),
            LL0,
            jnp.zeros((n_batch, self.config.nq_leg, self.config.nq_leg)),
        ], axis=1)

        # Leg Right Column
        col_leg_right = jnp.concatenate([
            jnp.zeros((n_batch, 3 + self.config.ang_vel_dof + self.config.nq_torso + 2*self.config.nq_arm + self.config.nq_leg, self.config.nq_leg)),
            LL1,
        ], axis=1)

        # === Final L matrix ===
        Lfinal = jnp.concatenate([
            col_lin,
            col_ang,
            col_torso,
            col_arm_left,
            col_arm_right,
            col_leg_left,
            col_leg_right,
        ], axis=2)

        # Final mass matrix: H = L'L
        H = jnp.transpose(Lfinal, (0, 2, 1)) @ Lfinal
        mass = jnp.trace(H[:,:3,:3], axis1=1, axis2=2) / 3.0
        return H, mass
    
    def get_inertia_matrix_quad_with_arm(self, q_full):
        # q_full: base + actuated joints
        # q_base: base
        # q: actuated joints
        _, q = split_by_lengths(q_full, [6, self.config.nq], axis=1)
        qarm, qLF, qRF, qLH, qRH = split_by_lengths(q, [self.config.nq_arm, self.config.nq_leg, self.config.nq_leg, self.config.nq_leg, self.config.nq_leg])

        # Neural Networks outputs
        LL, LRL, LR = self.get_base_nn_output(q)

        LA0L, LA0R, LA0 = self.get_leg_nn_output(qarm, self.larm_net)
        LL0L, LL0R, LL0 = self.get_leg_nn_output(qLF, self.lleg_LF_net)
        LL1L, LL1R, LL1 = self.get_leg_nn_output(qRF, self.lleg_RF_net)
        LL2L, LL2R, LL2 = self.get_leg_nn_output(qLH, self.lleg_LH_net)
        LL3L, LL3R, LL3 = self.get_leg_nn_output(qRH, self.lleg_RH_net)

        #### LR ####
        K = jnp.concatenate([
            LA0L,
            LL0L,
            LL1L,
            LL2L,
            LL3L,
        ], axis=1)

        W = jnp.concatenate([
            LA0R,
            LL0R,
            LL1R,
            LL2R,
            LL3R,
        ], axis=1)

        #### U ####
        U = jnp.concatenate([
            LRL,
            K,
        ], axis=1)

        ########### Build L ###########
        n_batch = LL.shape[0]

        # Linear Velocity Column
        col_lin = jnp.concatenate([
            LL,
            U,
        ], axis=1)

        # Angular Velocity Column
        col_ang = jnp.concatenate([
            jnp.zeros((n_batch, 3, 3)),
            LR,
            W,
        ], axis=1)

        # Arm Column
        col_arm = jnp.concatenate([
            jnp.zeros((n_batch, 3 + self.config.ang_vel_dof, self.config.nq_arm)),
            LA0,
            jnp.zeros((n_batch, 4*self.config.nq_leg, self.config.nq_arm)),
        ], axis=1)

        # Leg 0 Column
        col_l0 = jnp.concatenate([
            jnp.zeros((n_batch, 3 + self.config.ang_vel_dof, self.config.nq_leg)),
            jnp.zeros((n_batch, self.config.nq_arm, self.config.nq_leg)),
            LL0,
            jnp.zeros((n_batch, 3*self.config.nq_leg, self.config.nq_leg)),
        ], axis=1)

        # Leg 1 Column
        col_l1 = jnp.concatenate([
            jnp.zeros((n_batch, 3 + self.config.ang_vel_dof, self.config.nq_leg)),
            jnp.zeros((n_batch, self.config.nq_arm, self.config.nq_leg)),
            jnp.zeros((n_batch, self.config.nq_leg, self.config.nq_leg)),
            LL1,
            jnp.zeros((n_batch, 2*self.config.nq_leg, self.config.nq_leg)),
        ], axis=1)

        # Leg 2 Column
        col_l2 = jnp.concatenate([
            jnp.zeros((n_batch, 3 + self.config.ang_vel_dof, self.config.nq_leg)),
            jnp.zeros((n_batch, self.config.nq_arm, self.config.nq_leg)),
            jnp.zeros((n_batch, 2*self.config.nq_leg, self.config.nq_leg)),
            LL2,
            jnp.zeros((n_batch, self.config.nq_leg, self.config.nq_leg)),
        ], axis=1)

        # Leg 3 Column
        col_l3 = jnp.concatenate([
            jnp.zeros((n_batch, 3 + self.config.ang_vel_dof, self.config.nq_leg)),
            jnp.zeros((n_batch, self.config.nq_arm, self.config.nq_leg)),
            jnp.zeros((n_batch, 3*self.config.nq_leg, self.config.nq_leg)),
            LL3,
        ], axis=1)

        # === Final L matrix ===
        Lfinal = jnp.concatenate([
            col_lin,
            col_ang,
            col_arm,
            col_l0,
            col_l1,
            col_l2,
            col_l3,
        ], axis=2)

        # Final mass matrix: H = L'L
        H = jnp.transpose(Lfinal, (0, 2, 1)) @ Lfinal
        mass = jnp.trace(H[:,:3,:3], axis1=1, axis2=2) / 3.0
        return H, mass

    def get_inertia_matrix_quad(self, q_full):
        # q_full: base + actuated joints
        # q_base: base
        # q: actuated joints
        _, q = split_by_lengths(q_full, [6, self.config.nq], axis=1)
        qLF, qRF, qLH, qRH = split_by_lengths(q, [self.config.nq_leg, self.config.nq_leg, self.config.nq_leg, self.config.nq_leg])

        # Neural Networks outputs
        LL, LRL, LR = self.get_base_nn_output(q)

        L0L, L0R, L0 = self.get_leg_nn_output(qLF, self.lleg_LF_net)
        L1L, L1R, L1 = self.get_leg_nn_output(qRF, self.lleg_RF_net)
        L2L, L2R, L2 = self.get_leg_nn_output(qLH, self.lleg_LH_net)
        L3L, L3R, L3 = self.get_leg_nn_output(qRH, self.lleg_RH_net)

        #### LR ####
        K = jnp.concatenate([
            L0L,
            L1L,
            L2L,
            L3L,
        ], axis=1)

        W = jnp.concatenate([
            L0R,
            L1R,
            L2R,
            L3R,
        ], axis=1)

        #### U ####
        U = jnp.concatenate([
            LRL,
            K,
        ], axis=1)

        ########### Build L ###########
        n_batch = LL.shape[0]

        # Linear Velocity Column
        col_lin = jnp.concatenate([
            LL,
            U,
        ], axis=1)

        # Angular Velocity Column
        col_ang = jnp.concatenate([
            jnp.zeros((n_batch, 3, 3)),
            LR,
            W,
        ], axis=1)

        # Leg 0 Column
        col_l0 = jnp.concatenate([
            jnp.zeros((n_batch, 3 + self.config.ang_vel_dof, self.config.nq_leg)),
            L0,
            jnp.zeros((n_batch, 3*self.config.nq_leg, self.config.nq_leg)),
        ], axis=1)

        # Leg 1 Column
        col_l1 = jnp.concatenate([
            jnp.zeros((n_batch, 3 + self.config.ang_vel_dof, self.config.nq_leg)),
            jnp.zeros((n_batch, self.config.nq_leg, self.config.nq_leg)),
            L1,
            jnp.zeros((n_batch, 2*self.config.nq_leg, self.config.nq_leg)),
        ], axis=1)

        # Leg 2 Column
        col_l2 = jnp.concatenate([
            jnp.zeros((n_batch, 3 + self.config.ang_vel_dof, self.config.nq_leg)),
            jnp.zeros((n_batch, 2*self.config.nq_leg, self.config.nq_leg)),
            L2,
            jnp.zeros((n_batch, self.config.nq_leg, self.config.nq_leg)),
        ], axis=1)

        # Leg 3 Column
        col_l3 = jnp.concatenate([
            jnp.zeros((n_batch, 3 + self.config.ang_vel_dof, self.config.nq_leg)),
            jnp.zeros((n_batch, 3*self.config.nq_leg, self.config.nq_leg)),
            L3,
        ], axis=1)

        # === Final L matrix ===
        Lfinal = jnp.concatenate([
            col_lin,
            col_ang,
            col_l0,
            col_l1,
            col_l2,
            col_l3,
        ], axis=2)

        # Final mass matrix: H = L'L
        H = jnp.transpose(Lfinal, (0, 2, 1)) @ Lfinal
        mass = jnp.trace(H[:,:3,:3], axis1=1, axis2=2) / 3.0
        return H, mass

    def convet_inertia_to_vw_euler_rate(self, q_full, inertia_mat, rot_base_to_world, Wn_inv):

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
    
    def potential_energy(self, rb, rot_base_to_world, mass_value, prod_mass_r):
        total_value = mass_value * rb[..., None] + rot_base_to_world @ prod_mass_r
        P = jnp.dot(self.gravity_array13, total_value)
        return jnp.squeeze(P, axis=-1)
    
    def lagrangian_euler_rates_vb_fn(self, q_full, qd_full):
        
        rot_base_to_world = get_rot_base_to_world(q_full[:,3:6])
        Wn_inv = self.euler_rate_to_omega_mat(q_full[:,3:6])

        inertia_mat_raw, mass = self.get_inertia_matrix(q_full)

        inertia_mat = self.convet_inertia_to_vw_euler_rate(q_full, inertia_mat_raw, rot_base_to_world, Wn_inv)

        inertia_skew = inertia_mat_raw[:,:3,3:6]
        pot_comp = -get_vector_from_skew(inertia_skew)

        # Compute Energies
        e_kin = self.kinetic_energy(qd_full, inertia_mat)
        e_pot = self.potential_energy(q_full[:,:3], rot_base_to_world, mass, pot_comp)

        return e_kin - e_pot

    def dyn_model_hessian(self, q_full, qd_full, qdd_full = None):

        v_euler_rates, acc_euler_rates = self.get_euler_rates_and_acc(q_full, qd_full, qdd_full)

        dLdq = self.vmap_dLdq_fn(q_full, v_euler_rates)[:, :, None]
        d2L_dqddq, d2Ld2qd = self.vmap_d2L_fn(q_full, v_euler_rates) 

        # Compute the predicted generalized force:
        tau_pred = jnp.matmul(d2Ld2qd.squeeze(), acc_euler_rates) + jnp.matmul(d2L_dqddq.squeeze(), v_euler_rates) - dLdq

        tau_pred = self.gen_force_to_angular_frame(q_full, tau_pred).squeeze()

        dEdt = jnp.sum(qd_full * tau_pred, axis=1)

        return tau_pred.squeeze(), dEdt

    def __call__(self, q, qd, qdd):
        out = self.dyn_model(q, qd, qdd)
        tau_pred = out[0]
        dEdt = out[1]
        extras = {}
        return tau_pred, dEdt, extras
