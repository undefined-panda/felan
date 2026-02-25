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
    init_mass: float = 90.0
    l_output_size: int = 0
    l_diag_size: int = 0
    l_lower_size: int = 0
    idx_nq: jnp.ndarray = 0
    tril_indices_nq: Tuple[jnp.ndarray, jnp.ndarray] = struct.field(default_factory=tuple)

class ComponentNN(nn.Module):
    config: ComponentConfig

    @nn.compact
    def __call__(self, x = None):
        if x is None:
            m_sqrt = self.param('w', nn.initializers.constant(jnp.sqrt(self.config.init_mass)), ())
            return jnp.power(m_sqrt, 2)

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
class FeLaNConfig:
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
    H_epsilon: float = 0.01
    mass_epsilon: float = 1.0
    rot_epsilon: float = 0.01
    # Mass - Inequality
    mass_ineq: bool = False
    # Skew Symmetric - First principal moment
    skew_sym_ineq: bool = False
    # Triangle Inequality
    tri_ineq: bool = False
    tri_ineq_act_name: str = 'Softplus'
    tri_ineq_softplus_beta: float = 6.0
    tri_ineq_softplus_shift: float = 0.0
    diff_mass_softplus_beta: float = 5.0
    diff_mass_shift_loss: float = 1.0
    penalize_extra: bool = True
    penalize_diff_mass: float = 10.0
    penalize_rot_eigenvalue: float = 1.0
    final_sigma_epsilon: float = 1e-2
    # Components Config
    mass_config: Optional[ComponentConfig] = None
    base_rot_config: Optional[ComponentConfig] = None
    base_mass_config: Optional[ComponentConfig] = None
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

    ## Mass Net
    mass_config = ComponentConfig(n_output=1,
                                  init_mass=kwargs.get('init_mass', 90.0),
                                )

    ## Base Rot Net
    lbase_rot_output_size = int((ang_vel_dof ** 2 + ang_vel_dof) / 2)
    lbase_rot_diag_size = ang_vel_dof
    lbase_rot_lower_size = lbase_rot_output_size - lbase_rot_diag_size

    lbase_tril_indices_nq = jnp.tril_indices(ang_vel_dof)
    lbase_rot_idx_nq = get_idx_triangular(ang_vel_dof, lbase_rot_output_size)

    skew_sym_ineq = kwargs.get('skew_sym_ineq', False)
    if skew_sym_ineq:
        # Skew-symmetric matrix input + RR
        lbase_rot_full_size = ang_vel_dof + lbase_rot_output_size
    else:
        # Estimate the entire LRL
        lbase_rot_full_size = ang_vel_dof ** 2 + lbase_rot_output_size

    mass_ineq = kwargs.get('mass_ineq', False)
    if mass_ineq == False:
        lbase_rot_full_size += int((lin_vel_dof ** 2 + lin_vel_dof) / 2)


    tri_ineq = kwargs.get('tri_ineq', False)
    if tri_ineq:
        base_rot_epsilon = kwargs.get('hr_sigma_epsilon', 0.1)
    else:
        base_rot_epsilon = kwargs.get('diagonal_epsilon', 1e-5)

    base_rot_config = ComponentConfig(
                                n_output=lbase_rot_full_size,
                                nq=ang_vel_dof,
                                net_arch=kwargs.get('net_arch_base_rot', [16, 16]),
                                init_tf=kwargs.get('init_tf', True),
                                epsilon=base_rot_epsilon,
                                shift=kwargs.get('diagonal_shift', 0.0),
                                act_ld_name=kwargs.get('act_ld', 'Softplus'),
                                softplus_beta=kwargs.get('softplus_beta_base_rot', 1.0),
                                activation_name=kwargs.get('activation', 'tanh'),
                                l_output_size=lbase_rot_output_size,
                                l_diag_size=lbase_rot_diag_size,
                                l_lower_size=lbase_rot_lower_size,
                                tril_indices_nq=lbase_tril_indices_nq,
                                idx_nq=lbase_rot_idx_nq,
                        )
    
    # Same sizes as we partially use them
    base_mass_config = ComponentConfig(
                                n_output=lbase_rot_full_size,
                                nq=ang_vel_dof,
                                net_arch=kwargs.get('net_arch_base_rot', [16, 16]),
                                init_tf=kwargs.get('init_tf', True),
                                epsilon=kwargs.get('diagonal_epsilon', 1e-5),
                                shift=kwargs.get('diagonal_shift', 0.0),
                                act_ld_name=kwargs.get('act_ld', 'Softplus'),
                                softplus_beta=kwargs.get('softplus_beta', 1.0),
                                activation_name=kwargs.get('activation', 'tanh'),
                                l_output_size=lbase_rot_output_size,
                                l_diag_size=lbase_rot_diag_size,
                                l_lower_size=lbase_rot_lower_size,
                                tril_indices_nq=lbase_tril_indices_nq,
                                idx_nq=lbase_rot_idx_nq,
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
        # ULi, RLi, LLi
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
        # ULi, RLi, LLi
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
        # ULi, RLi, LLi
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

    config = FeLaNConfig(
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
        H_epsilon=kwargs.get("H_epsilon", 0.01),
        mass_epsilon=kwargs.get("mass_epsilon", 1.0),
        rot_epsilon=kwargs.get("rot_epsilon", 0.01),
        # Mass Inequality
        mass_ineq=mass_ineq,
        # Skew Symmetric - First principal moment
        skew_sym_ineq=skew_sym_ineq,
        # Triangle Inequality
        tri_ineq=tri_ineq,
        tri_ineq_act_name=kwargs.get('tri_ineq_act_name', 'Softplus'),
        tri_ineq_softplus_beta=kwargs.get('tri_ineq_softplus_beta', 6.0),
        tri_ineq_softplus_shift=kwargs.get('tri_ineq_softplus_shift', 0.0),
        diff_mass_softplus_beta=kwargs.get("diff_mass_softplus_beta", 5.0),
        diff_mass_shift_loss=kwargs.get("diff_mass_shift_loss", 1.0),
        penalize_extra=kwargs.get("penalize_extra", True),
        penalize_diff_mass=kwargs.get("penalize_diff_mass", 10.0),
        penalize_rot_eigenvalue=kwargs.get("penalize_rot_eigenvalue", 1.0),
        final_sigma_epsilon=kwargs.get("final_sigma_epsilon", 1e-2),
        # Components Config
        mass_config=mass_config,
        base_rot_config=base_rot_config,
        base_mass_config=base_mass_config,
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


class FeLaN(nn.Module):
    n_dof: int
    config: FeLaNConfig

    def setup(self):
        # UL + US
        self.lmass_net = ComponentNN(config=self.config.mass_config)

        # S, RR
        self.lbase_rot_net = ComponentNN(config=self.config.base_rot_config)

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

            # ULi, RLi, LLi
            self.lleg_LF_net = ComponentNN(config=self.config.leg_config)
            self.lleg_RF_net = ComponentNN(config=self.config.leg_config)
            self.lleg_LH_net = ComponentNN(config=self.config.leg_config)
            self.lleg_RH_net = ComponentNN(config=self.config.leg_config)

            self.get_inertia_matrix = self.get_inertia_matrix_quad_with_arm

        elif self.config.robot_type == 'quad':
            # ULi, RLi, LLi
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
            return jnp.squeeze(self.lagrangian_euler_rates_vb_fn(q[None, :], qd[None, :])[0])
        self.vmap_dLdq_fn = jax.vmap(jax.jacrev(lagrangian_scalar, argnums=0))
        self.vmap_d2L_fn = jax.vmap(jax.jacfwd(jax.jacrev(lagrangian_scalar, argnums=1), argnums=(0, 1)))

        def full_lagrangian_scalar(q, qd):
            full_output = self.lagrangian_euler_rates_vb_fn(q[None, :], qd[None, :])
            return jnp.squeeze(full_output[0]), full_output[1]
        self.vmap_L_fn = jax.vmap(full_lagrangian_scalar)

        self.vmap_get_rot_inertia_from_tri_ineq_sigma = jax.vmap(self.get_rot_inertia_from_tri_ineq_sigma)
        self.vmap_get_linear_factor_from_mass = jax.vmap(self.get_linear_factor_from_mass)

    def get_mass_nn_output(self):
        return apply_act_fn(self.lmass_net() + self.config.shift, self.config.act_ld_name, self.config.softplus_beta) + self.config.epsilon
    
    def get_base_rot_nn_output(self, q):
        first_output_length = self.config.ang_vel_dof if self.config.skew_sym_ineq else self.config.ang_vel_dof ** 2
        output_lengths = [first_output_length,
                          self.lbase_rot_net.config.l_output_size]
        S_input, RR_input = split_by_lengths(self.lbase_rot_net(q),
                                             output_lengths, 
                                             axis=1)
        
        if self.config.skew_sym_ineq:
            S = skew_sym_matrix(S_input)
        else:
            S = S_input.reshape(-1, self.config.ang_vel_dof, self.config.ang_vel_dof)

        RR = self.vmap_get_L_from_output(RR_input, self.lbase_rot_net.config)
        
        RR_diff = jnp.diagonal(RR, axis1=1, axis2=2) - RR_input[:,:3] - self.lbase_rot_net.config.epsilon
        return S, RR, RR_diff
    
    def get_base_nn_output(self, q):
        first_output_length = self.config.ang_vel_dof if self.config.skew_sym_ineq else self.config.ang_vel_dof ** 2
        output_lengths = [first_output_length,
                          self.lbase_rot_net.config.l_output_size,
                          self.lbase_rot_net.config.l_output_size]
        S_input, UL_input, RR_input = split_by_lengths(self.lbase_rot_net(q),
                                             output_lengths,
                                             axis=1)
        
        if self.config.skew_sym_ineq:
            S = skew_sym_matrix(S_input)
        else:
            S = S_input.reshape(-1, self.config.ang_vel_dof, self.config.ang_vel_dof)

        RR = self.vmap_get_L_from_output(RR_input, self.lbase_rot_net.config)
        UL = self.vmap_get_L_from_output(UL_input, self.config.base_mass_config)

        RR_diff = jnp.diagonal(RR, axis1=1, axis2=2) - RR_input[:,:3] - self.lbase_rot_net.config.epsilon
        return S, UL, RR, RR_diff
    
    def get_torso_nn_output(self, qtorso):
        output_lengths = [self.config.nq_torso * self.config.lin_vel_dof,
                          self.config.nq_torso * self.config.ang_vel_dof,
                          self.ltorso_net.config.l_output_size]

        UT, RT, LT = split_by_lengths(self.ltorso_net(qtorso),
                                      output_lengths,
                                      axis=1)
        LT = self.vmap_get_L_from_output(LT, self.ltorso_net.config)
        return UT.reshape(-1, self.ltorso_net.config.nq, self.config.lin_vel_dof), RT.reshape(-1, self.ltorso_net.config.nq, self.config.ang_vel_dof), LT

    def get_arm_nn_output(self, qtorso, qarm, arm_net, torso_arm_net):
        LAi = arm_net(qarm)
        LAi = self.vmap_get_L_from_output(LAi, arm_net.config)

        output_lengths = [self.config.nq_arm * self.config.lin_vel_dof,
                          self.config.nq_arm * self.config.ang_vel_dof,
                          self.config.nq_arm * self.config.nq_torso]

        q_input = jnp.concatenate([qtorso,
                                     qarm], axis=1)
        UAi, RAi, LTAi = split_by_lengths(torso_arm_net(q_input), 
                                         output_lengths, 
                                         axis=1)
        return UAi.reshape(-1, self.config.nq_arm, self.config.lin_vel_dof), RAi.reshape(-1, self.config.nq_arm, self.config.ang_vel_dof), LTAi.reshape(-1, self.config.nq_arm, self.config.nq_torso), LAi

    def get_leg_nn_output(self, qleg, leg_net):
        output_lengths = [leg_net.config.nq * self.config.lin_vel_dof,
                          leg_net.config.nq * self.config.ang_vel_dof,
                          leg_net.config.l_output_size]

        ULi, RLi, LLi = split_by_lengths(leg_net(qleg), 
                                         output_lengths, 
                                         axis=1)
        LLi = self.vmap_get_L_from_output(LLi, leg_net.config)
        return ULi.reshape(-1, leg_net.config.nq, self.config.lin_vel_dof), RLi.reshape(-1, leg_net.config.nq, self.config.ang_vel_dof), LLi

    def get_rot_inertia_from_tri_ineq_sigma(self, Sigma, W):
        eye_3 = jnp.eye(3)
        W_prod = jnp.transpose(W, (1, 0)) @ W

        HR = jnp.trace(Sigma) * eye_3 - Sigma

        LR_prod = HR - W_prod
        LR_prod = LR_prod

        LR_prod_eigvalues = jnp.real(jnp.linalg.eigvalsh(LR_prod))
        LR_prod_min_eigvalue = jnp.min(LR_prod_eigvalues)
        LR_prod_min_eigvalue = jax.lax.stop_gradient(LR_prod_min_eigvalue)

        LR_prod_shift = apply_act_fn(- LR_prod_min_eigvalue + self.config.tri_ineq_softplus_shift, 
                                     self.config.tri_ineq_act_name,
                                     self.config.tri_ineq_softplus_beta)

        LR_prod_new = HR - W_prod + (self.config.rot_epsilon + LR_prod_shift) * eye_3

        RR_pen = LR_prod_shift

        return LR_prod_new, HR, RR_pen, LR_prod_eigvalues
    
    def get_linear_factor_from_mass(self, US, raw_mass):
        Us_prod = jnp.transpose(US, (1, 0)) @ US

        Us_prod_eig = jnp.real(jnp.linalg.eigvalsh(Us_prod))
        Us_prod_max_eigvalue = jax.lax.stop_gradient(jnp.max(Us_prod_eig))

        mass = apply_act_fn(raw_mass - Us_prod_max_eigvalue, 'Softplus', self.config.diff_mass_softplus_beta) + self.config.mass_epsilon + Us_prod_max_eigvalue
       
        LL_input = mass * jnp.eye(3) - Us_prod

        return LL_input, mass, Us_prod_eig

    def get_inertia_matrix_humanoid(self, q_full, Wn):
        n_batch = q_full.shape[0]

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

        if self.config.mass_ineq:
            raw_mass = self.lmass_net()
            # Neural Networks outputs
            S, Sigma_RR, RR_diff = self.get_base_rot_nn_output(q)
            Sigma_mat = jnp.transpose(Sigma_RR, (0, 2, 1)) @ Sigma_RR + self.config.final_sigma_epsilon * jnp.eye(Sigma_RR.shape[-1])

        else:
            S, UL, Sigma_RR, RR_diff = self.get_base_nn_output(q)
            Sigma_mat = jnp.transpose(Sigma_RR, (0, 2, 1)) @ Sigma_RR + self.config.final_sigma_epsilon * jnp.eye(Sigma_RR.shape[-1])

        # Neural Networks outputs
        UT, RT, LT = self.get_torso_nn_output(q_torso_arms)
        UA0, RA0, LTA0, LA0 = self.get_arm_nn_output(qtorso, qarm_left, self.larm_left_net, self.ltorso_arm_left_net)
        UA1, RA1, LTA1, LA1 = self.get_arm_nn_output(qtorso, qarm_right, self.larm_right_net, self.ltorso_arm_right_net)
        UL0, RL0, LL0 = self.get_leg_nn_output(qleg_left, self.lleg_left_net)
        UL1, RL1, LL1 = self.get_leg_nn_output(qleg_right, self.lleg_right_net)

        #### UR ####
        USR = jnp.concatenate([
            UT,
            UA0,
            UA1,
            UL0,
            UL1,
        ], axis=1)

        RSR = jnp.concatenate([
            RT,
            RA0,
            RA1,
            RL0,
            RL1,
        ], axis=1)


        ## Triangle Inequality constraint
        if self.config.tri_ineq:
            RR_prod_new, HR, RR_pen, LR_prod_eigvalues = self.vmap_get_rot_inertia_from_tri_ineq_sigma(Sigma_mat, RSR)
            RR = reordered_cholesky(RR_prod_new)

        else:        
            # Default case
            RR = Sigma_RR
            RR_pen = jnp.zeros((n_batch, 1))

        ## Skew-symmetric constraint
        if self.config.skew_sym_ineq:
            UR_prod_term = jnp.transpose(USR, (0, 2, 1)) @ RSR

            UR_input = jnp.transpose(S, (0, 2, 1)) - UR_prod_term
            UR = jax.scipy.linalg.solve_triangular(jnp.transpose(RR, (0, 2, 1)), jnp.transpose(UR_input, (0, 2, 1)), lower=False)

        else:
            UR = S

        #### US ####
        US = jnp.concatenate([
            UR,
            USR,
        ], axis=1)

        ## Constant mass constraint
        #### UL ####
        if self.config.mass_ineq:
            UL_input, mass, Us_prod_eig = self.vmap_get_linear_factor_from_mass(US, raw_mass[...,None])
            diff_mass = jnp.sum(jax.nn.softplus(-(raw_mass - Us_prod_eig) + self.config.diff_mass_shift_loss))
            UL = reordered_cholesky(UL_input)

        ########### Build L ###########
        n_batch = UL.shape[0]

        # Linear Velocity Column
        col_lin = jnp.concatenate([
            UL,
            US,
        ], axis=1)

        # Angular Velocity Column
        col_ang = jnp.concatenate([
            jnp.zeros((n_batch, 3, 3)),
            RR,
            RSR,
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
        eye_H = jnp.eye(H.shape[-1], dtype=H.dtype)
        eye_H = jnp.broadcast_to(eye_H, H.shape)
        H = H + self.config.H_epsilon * eye_H

        if self.config.mass_ineq == False:
            mass = jnp.trace(H[:,:3,:3], axis1=1, axis2=2) / 3.0
            diff_mass = jnp.zeros_like(mass)

        Sigma_mat_cond = jnp.linalg.cond(Sigma_mat)
        RR_cond = jnp.linalg.cond(RR)
        UR_cond = jnp.linalg.cond(UR)
        S_cond = jnp.linalg.cond(UR)
        UL_input_cond = jnp.zeros_like(UR_cond)
        UR_input_cond = jnp.zeros_like(UR_cond)
        UR_prod_cond = jnp.zeros_like(UR_cond)
        USR_cond = jnp.linalg.cond(USR)
        RSR_cond = jnp.linalg.cond(RSR)
        if self.config.skew_sym_ineq:
            UR_input_cond =  jnp.linalg.cond(UR_input)
            UR_prod_cond = jnp.linalg.cond(UR_prod_term)

        if self.config.mass_ineq:
            UL_input_cond = jnp.linalg.cond(UL_input)

        extras = {
            'diff_mass': diff_mass.squeeze(),
            'diff_rot_eigenvalue': RR_pen.squeeze(),
            'Sigma_mat_cond': Sigma_mat_cond.squeeze(),
            'RR_cond': RR_cond.squeeze(),
            'UR_cond': UR_cond.squeeze(),
            'UL_input_cond': UL_input_cond.squeeze(),
            'UR_input_cond': UR_input_cond.squeeze(),
            'S_cond': S_cond.squeeze(),
            'UR_prod_cond': UR_prod_cond.squeeze(),
            'USR_cond': USR_cond.squeeze(),
            'RSR_cond': RSR_cond.squeeze(),
        }
        return H, mass, extras

    def get_inertia_matrix_quad_with_arm(self, q_full, Wn):
        n_batch = q_full.shape[0]
        # q_full: base + actuated joints
        # q_base: base
        # q: actuated joints
        _, q = split_by_lengths(q_full, [6, self.config.nq], axis=1)
        qarm, qLF, qRF, qLH, qRH = split_by_lengths(q, [self.config.nq_arm, self.config.nq_leg, self.config.nq_leg, self.config.nq_leg, self.config.nq_leg])


        if self.config.mass_ineq:
            raw_mass = self.lmass_net()
            
            # Neural Networks outputs
            S, Sigma_RR, RR_diff = self.get_base_rot_nn_output(q)
            Sigma_mat = jnp.transpose(Sigma_RR, (0, 2, 1)) @ Sigma_RR + self.config.final_sigma_epsilon * jnp.eye(Sigma_RR.shape[-1])

        else:
            S, UL, Sigma_RR, RR_diff = self.get_base_nn_output(q)
            Sigma_mat = jnp.transpose(Sigma_RR, (0, 2, 1)) @ Sigma_RR + self.config.final_sigma_epsilon * jnp.eye(Sigma_RR.shape[-1])

        UA0, RA0, LA0 = self.get_leg_nn_output(qarm, self.larm_net)
        UL0, RL0, LL0 = self.get_leg_nn_output(qLF, self.lleg_LF_net)
        UL1, RL1, LL1 = self.get_leg_nn_output(qRF, self.lleg_RF_net)
        UL2, RL2, LL2 = self.get_leg_nn_output(qLH, self.lleg_LH_net)
        UL3, RL3, LL3 = self.get_leg_nn_output(qRH, self.lleg_RH_net)

        #### UR ####
        USR = jnp.concatenate([
            UA0,
            UL0,
            UL1,
            UL2,
            UL3,
        ], axis=1)

        RSR = jnp.concatenate([
            RA0,
            RL0,
            RL1,
            RL2,
            RL3,
        ], axis=1)

        ## Triangle Inequality constraint
        if self.config.tri_ineq:

            RR_prod_new, HR, RR_pen, LR_prod_eigvalues = self.vmap_get_rot_inertia_from_tri_ineq_sigma(Sigma_mat, RSR)
            RR = reordered_cholesky(RR_prod_new)
        
        else:
            RR = Sigma_RR
            RR_pen = jnp.zeros((n_batch, 1))


        ## Skew-symmetric constraint
        if self.config.skew_sym_ineq:
            UR_prod_term = jnp.transpose(USR, (0, 2, 1)) @ RSR

            UR_input = jnp.transpose(S, (0, 2, 1)) - UR_prod_term
            UR = jax.scipy.linalg.solve_triangular(jnp.transpose(RR, (0, 2, 1)), jnp.transpose(UR_input, (0, 2, 1)), lower=False)

        else:
            UR = S

        #### US ####
        US = jnp.concatenate([
            UR,
            USR,
        ], axis=1)

        ## Constant mass constraint
        #### UL ####
        if self.config.mass_ineq:
            UL_input, mass, Us_prod_eig = self.vmap_get_linear_factor_from_mass(US, raw_mass[...,None])
            diff_mass = jnp.sum(jax.nn.softplus(-(raw_mass - Us_prod_eig) + self.config.diff_mass_shift_loss))
            UL = reordered_cholesky(UL_input)

        ########### Build L ###########
        n_batch = UL.shape[0]

        # Linear Velocity Column
        col_lin = jnp.concatenate([
            UL,
            US,
        ], axis=1)

        # Angular Velocity Column
        col_ang = jnp.concatenate([
            jnp.zeros((n_batch, 3, 3)),
            RR,
            RSR,
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
        eye_H = jnp.eye(H.shape[-1], dtype=H.dtype)
        eye_H = jnp.broadcast_to(eye_H, H.shape)
        H = H + self.config.H_epsilon * eye_H

        if self.config.mass_ineq == False:
            mass = jnp.trace(H[:,:3,:3], axis1=1, axis2=2) / 3.0
            diff_mass = jnp.zeros_like(mass)
        
        Sigma_mat_cond = jnp.linalg.cond(Sigma_mat)
        RR_cond = jnp.linalg.cond(RR)
        UR_cond = jnp.linalg.cond(UR)
        S_cond = jnp.linalg.cond(UR)
        UL_input_cond = jnp.zeros_like(UR_cond)
        UR_input_cond = jnp.zeros_like(UR_cond)
        UR_prod_cond = jnp.zeros_like(UR_cond)
        USR_cond = jnp.linalg.cond(USR)
        RSR_cond = jnp.linalg.cond(RSR)
        if self.config.skew_sym_ineq:
            UR_input_cond =  jnp.linalg.cond(UR_input)
            UR_prod_cond = jnp.linalg.cond(UR_prod_term)

        if self.config.mass_ineq:
            UL_input_cond = jnp.linalg.cond(UL_input)

        extras = {
            'diff_mass': diff_mass.squeeze(),
            'diff_rot_eigenvalue': RR_pen.squeeze(),
            'Sigma_mat_cond': Sigma_mat_cond.squeeze(),
            'RR_cond': RR_cond.squeeze(),
            'UR_cond': UR_cond.squeeze(),
            'UL_input_cond': UL_input_cond.squeeze(),
            'UR_input_cond': UR_input_cond.squeeze(),
            'S_cond': S_cond.squeeze(),
            'UR_prod_cond': UR_prod_cond.squeeze(),
            'USR_cond': USR_cond.squeeze(),
            'RSR_cond': RSR_cond.squeeze(),
        }

        return H, mass, extras

    def get_inertia_matrix_quad(self, q_full, Wn):
        n_batch = q_full.shape[0]
        # q_full: base + actuated joints
        # q_base: base
        # q: actuated joints
        _, q = split_by_lengths(q_full, [6, self.config.nq], axis=1)
        qLF, qRF, qLH, qRH = split_by_lengths(q, [self.config.nq_leg, self.config.nq_leg, self.config.nq_leg, self.config.nq_leg])

        if self.config.mass_ineq:
            raw_mass = self.lmass_net()
            
            # Neural Networks outputs
            S, Sigma_RR, RR_diff = self.get_base_rot_nn_output(q)
            Sigma_mat = jnp.transpose(Sigma_RR, (0, 2, 1)) @ Sigma_RR + self.config.final_sigma_epsilon * jnp.eye(Sigma_RR.shape[-1])

        else:
            S, UL, Sigma_RR, RR_diff = self.get_base_nn_output(q)
            Sigma_mat = jnp.transpose(Sigma_RR, (0, 2, 1)) @ Sigma_RR + self.config.final_sigma_epsilon * jnp.eye(Sigma_RR.shape[-1])

        UL0, RL0, LL0 = self.get_leg_nn_output(qLF, self.lleg_LF_net)
        UL1, RL1, LL1 = self.get_leg_nn_output(qRF, self.lleg_RF_net)
        UL2, RL2, LL2 = self.get_leg_nn_output(qLH, self.lleg_LH_net)
        UL3, RL3, LL3 = self.get_leg_nn_output(qRH, self.lleg_RH_net)

        #### UR ####
        USR = jnp.concatenate([
            UL0,
            UL1,
            UL2,
            UL3,
        ], axis=1)

        RSR = jnp.concatenate([
            RL0,
            RL1,
            RL2,
            RL3,
        ], axis=1)

        ## Triangle Inequality constraint
        if self.config.tri_ineq:

            RR_prod_new, HR, RR_pen_2, LR_prod_eigvalues = self.vmap_get_rot_inertia_from_tri_ineq_sigma(Sigma_mat, RSR)
            RR_pen = RR_pen_2
            RR = reordered_cholesky(RR_prod_new)
        
        else:
            RR = Sigma_RR
            RR_pen = jnp.zeros((n_batch, 1))

        ## Skew-symmetric constraint
        if self.config.skew_sym_ineq:
            UR_prod_term = jnp.transpose(USR, (0, 2, 1)) @ RSR
            UR_input = jnp.transpose(S, (0, 2, 1)) - UR_prod_term
            UR = jax.scipy.linalg.solve_triangular(jnp.transpose(RR, (0, 2, 1)), jnp.transpose(UR_input, (0, 2, 1)), lower=False)
        else:
            UR = S

        #### US ####
        US = jnp.concatenate([
            UR,
            USR,
        ], axis=1)

        ## Constant mass constraint
        #### UL ####
        if self.config.mass_ineq:
            UL_input, mass, Us_prod_eig = self.vmap_get_linear_factor_from_mass(US, raw_mass[...,None])
            diff_mass = jnp.sum(jax.nn.softplus(-(raw_mass - Us_prod_eig) + self.config.diff_mass_shift_loss))
            UL = reordered_cholesky(UL_input)

        ########### Build L ###########
        n_batch = UL.shape[0]

        # Linear Velocity Column
        col_lin = jnp.concatenate([
            UL,
            US,
        ], axis=1)

        # Angular Velocity Column
        col_ang = jnp.concatenate([
            jnp.zeros((n_batch, 3, 3)),
            RR,
            RSR,
        ], axis=1)

        # Leg 0 Column
        col_l0 = jnp.concatenate([
            jnp.zeros((n_batch, 3 + self.config.ang_vel_dof, self.config.nq_leg)),
            LL0,
            jnp.zeros((n_batch, 3*self.config.nq_leg, self.config.nq_leg)),
        ], axis=1)

        # Leg 1 Column
        col_l1 = jnp.concatenate([
            jnp.zeros((n_batch, 3 + self.config.ang_vel_dof, self.config.nq_leg)),
            jnp.zeros((n_batch, self.config.nq_leg, self.config.nq_leg)),
            LL1,
            jnp.zeros((n_batch, 2*self.config.nq_leg, self.config.nq_leg)),
        ], axis=1)

        # Leg 2 Column
        col_l2 = jnp.concatenate([
            jnp.zeros((n_batch, 3 + self.config.ang_vel_dof, self.config.nq_leg)),
            jnp.zeros((n_batch, 2*self.config.nq_leg, self.config.nq_leg)),
            LL2,
            jnp.zeros((n_batch, self.config.nq_leg, self.config.nq_leg)),
        ], axis=1)

        # Leg 3 Column
        col_l3 = jnp.concatenate([
            jnp.zeros((n_batch, 3 + self.config.ang_vel_dof, self.config.nq_leg)),
            jnp.zeros((n_batch, 3*self.config.nq_leg, self.config.nq_leg)),
            LL3,
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
        eye_H = jnp.eye(H.shape[-1], dtype=H.dtype)
        eye_H = jnp.broadcast_to(eye_H, H.shape)
        H = H + self.config.H_epsilon * eye_H

        if self.config.mass_ineq == False:
            mass = jnp.trace(H[:,:3,:3], axis1=1, axis2=2) / 3.0
            diff_mass = jnp.zeros_like(mass)

        Sigma_mat_cond = jnp.linalg.cond(Sigma_mat)
        RR_cond = jnp.linalg.cond(RR)
        UR_cond = jnp.linalg.cond(UR)
        S_cond = jnp.linalg.cond(UR)
        UL_input_cond = jnp.zeros_like(UR_cond)
        UR_input_cond = jnp.zeros_like(UR_cond)
        UR_prod_cond = jnp.zeros_like(UR_cond)
        USR_cond = jnp.linalg.cond(USR)
        RSR_cond = jnp.linalg.cond(RSR)
        if self.config.skew_sym_ineq:
            UR_input_cond =  jnp.linalg.cond(UR_input)
            UR_prod_cond = jnp.linalg.cond(UR_prod_term)

        if self.config.mass_ineq:
            UL_input_cond = jnp.linalg.cond(UL_input)

        extras = {
            'diff_mass': diff_mass.squeeze(),
            'diff_rot_eigenvalue': RR_pen.squeeze(),
            'Sigma_mat_cond': Sigma_mat_cond.squeeze(),
            'RR_cond': RR_cond.squeeze(),
            'UR_cond': UR_cond.squeeze(),
            'UL_input_cond': UL_input_cond.squeeze(),
            'UR_input_cond': UR_input_cond.squeeze(),
            'S_cond': S_cond.squeeze(),
            'UR_prod_cond': UR_prod_cond.squeeze(),
            'USR_cond': USR_cond.squeeze(),
            'RSR_cond': RSR_cond.squeeze(),
        }
        return H, mass, extras

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

        # Now do: Tᵀ @ I @ T
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

    def kinetic_energy(self, qd: jnp.ndarray, M) -> jnp.ndarray:
        T = 0.5 * jnp.matmul(jnp.transpose(qd, (0, 2, 1)), M @ qd).squeeze(-1)
        return T

    def potential_energy(self, rb, rot_base_to_world, mass_value, prod_mass_r):
        total_value = mass_value * rb[..., None] + rot_base_to_world @ prod_mass_r
        V = jnp.dot(self.gravity_array13, total_value)
        return jnp.squeeze(V, axis=-1)

    def lagrangian_euler_rates_vb_fn(self, q_full, qd_full):
        
        rot_base_to_world = get_rot_base_to_world(q_full[:,3:6])
        Wn = self.omega_to_euler_rate_mat(q_full[:,3:6])
        Wn_inv = self.euler_rate_to_omega_mat(q_full[:,3:6])

        inertia_mat_raw, mass, extras = self.get_inertia_matrix(q_full, Wn)

        inertia_mat = self.convet_inertia_to_vw_euler_rate(q_full, inertia_mat_raw, rot_base_to_world, Wn_inv)
    
        inertia_skew = inertia_mat_raw[:,:3,3:6]
        pot_comp = -get_vector_from_skew(inertia_skew)

        # Compute Energies
        e_kin = self.kinetic_energy(qd_full, inertia_mat)
        e_pot = self.potential_energy(q_full[:,:3], rot_base_to_world, mass, pot_comp)

        return e_kin - e_pot, extras

    def dyn_model_hessian(self, q_full, qd_full, qdd_full):

        v_euler_rates, acc_euler_rates = self.get_euler_rates_and_acc(q_full, qd_full, qdd_full)

        L, extras = self.vmap_L_fn(q_full, v_euler_rates)

        dLdq = self.vmap_dLdq_fn(q_full, v_euler_rates)[:, :, None]
        d2L_dqddq, d2Ld2qd = self.vmap_d2L_fn(q_full, v_euler_rates) 

        # Compute the predicted generalized force:
        tau_pred = jnp.matmul(d2Ld2qd.squeeze(), acc_euler_rates) + jnp.matmul(d2L_dqddq.squeeze(), v_euler_rates) - dLdq

        tau_pred = self.gen_force_to_angular_frame(q_full, tau_pred).squeeze()

        dEdt = jnp.sum(qd_full * tau_pred, axis=1)

        return tau_pred, extras, dEdt

    def __call__(self, q, qd, qdd):
        out = self.dyn_model(q, qd, qdd)
        tau_pred = out[0]
        extras = out[1]
        dEdt = out[2]
        return tau_pred, dEdt, extras
