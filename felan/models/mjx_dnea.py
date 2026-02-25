from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

from flax import struct
from flax import linen as nn

import mujoco
from mujoco import mjx

from pathlib import Path
from .custom_mjx import *

def quaternion_wxyz_to_xyzw(quat):
    return jnp.concatenate([quat[..., 1:], quat[..., :1]], axis=-1)

def quaternion_xyzw_to_wxyz(quat):
    return jnp.concatenate([quat[..., -1:], quat[..., :-1]], axis=-1)

def apply_act_fn(input, act_name: str, softplus_beta: float = 1.0) -> jnp.ndarray:
    if act_name == 'Tanh':
        return nn.tanh(input)
    elif act_name == 'ReLu':
        return nn.relu(input)
    elif act_name == 'Softplus':
        return nn.softplus(softplus_beta * input) / (softplus_beta)
    else:
        raise ValueError(f"Unsupported activation: {act_name}")

def get_idx_triangular(n_rows, inertia_output_size):
    # Diagonal indices of a lower triangular matrix
    idx_diag = jnp.arange(n_rows) + 1
    idx_diag = idx_diag * (idx_diag + 1) // 2 - 1
    idx_diag = idx_diag.astype(jnp.int32)

    # All indices
    all_idx = jnp.arange(inertia_output_size)

    # Mask out non-diagonal indices
    mask = ~jnp.isin(all_idx, idx_diag)
    idx_tril = all_idx[mask]

    # Concatenate and sort
    cat_idx = jnp.concatenate([idx_diag, idx_tril])
    order = jnp.argsort(cat_idx)

    return jnp.arange(cat_idx.size)[order]

def decompose_single(Ic):
    eigvals, eigvecs = jnp.linalg.eigh(Ic)

    # Sort eigenvalues and eigenvectors to maintain consistent axis ordering
    idx = jnp.argsort(eigvals)
    J = eigvals#[idx]
    R = eigvecs#[:, idx]  # Columns are the principal axes in body frame

    # Ensure R is a proper rotation matrix (det = +1)
    det = jnp.linalg.det(R)
    R = jax.lax.cond(det < 0.0,
                     lambda R_: R_.at[:, 0].set(-R_[:, 0]),
                     lambda R_: R_,
                     R)

    return J, R
decompose_batch = jax.vmap(decompose_single, in_axes=(0,))

def canonicalize_eigvecs(Ic):
    eigvals, eigvecs = jnp.linalg.eigh(Ic)

    # eigvecs[:, i] is the i-th eigenvector
    # Match each to canonical axes
    body_axes = jnp.eye(3)  # x, y, z unit vectors
    projections = jnp.abs(eigvecs.T @ body_axes)  # (3, 3), each row = dot products with body axes

    # For each canonical axis (x, y, z), find the best matching eigenvector
    assignment = jnp.argmax(projections, axis=0)  # (3,), index of best eigvec for x, y, z

    # Check that assignment is unique (no duplicates)
    def is_unique(indices):
        return jnp.all(jnp.bincount(indices, length=3) == 1)

    # If ambiguous (two axes assigned the same eigvec), fallback to sort by value
    fallback = jnp.argsort(eigvals)

    assignment = jax.lax.cond(
        is_unique(assignment),
        lambda: assignment,
        lambda: fallback
    )

    # Reorder eigenvalues and eigenvectors
    J = eigvals[assignment]
    R = eigvecs[:, assignment]

    # Ensure right-handed frame
    det = jnp.linalg.det(R)
    R = jax.lax.cond(det < 0.0,
                     lambda R_: R_.at[:, 0].set(-R_[:, 0]),
                     lambda R_: R_,
                     R)

    return J, R  # J = [Jx, Jy, Jz], R = corresponding principal axes
decompose_batch = jax.vmap(canonicalize_eigvecs, in_axes=(0,))


def batch_expmap_to_quat(theta_rot):
    """
    Convert a batch of exponential maps (rotation vectors) to quaternions.
    Args:
        theta_rot: (n_batch, 3, 1) array of rotation vectors
    Returns:
        quat: (n_batch, 4) array of quaternions in (w, x, y, z) format
    """
    theta_rot = theta_rot  # (n, 3)
    theta = jnp.linalg.norm(theta_rot, axis=-1, keepdims=True)  # (n, 1)

    half_theta = 0.5 * theta
    axis = jnp.where(theta > 1e-8, theta_rot / theta, jnp.zeros_like(theta_rot))  # (n, 3)

    sin_half = jnp.sin(half_theta)  # (n, 1)
    cos_half = jnp.cos(half_theta)  # (n, 1)

    vec_part = axis * sin_half  # (n, 3)
    quat = jnp.concatenate([cos_half, vec_part], axis=-1)  # (n, 4), (w, x, y, z)

    return quat

def get_log_cholesky(params):
    alpha, d1, d2, d3, s12, s13, s23, t1, t2, t3 = params
    mat = jnp.array([
        [jnp.exp(d1), s12,       s13,       t1],
        [0.0,         jnp.exp(d2), s23,       t2],
        [0.0,         0.0,       jnp.exp(d3), t3],
        [0.0,         0.0,       0.0,       1.0],
    ])

    U = jnp.exp(alpha) * mat
    pseudo_J = U @ jnp.transpose(U, (1,0))

    mass = pseudo_J[3,3]
    h = pseudo_J[3,:3]
    com_pos = h / mass

    Sigma_c = pseudo_J[:3,:3]
    trace_sigma_c = jnp.trace(Sigma_c, axis1=0, axis2=1)
    Ic = trace_sigma_c * jnp.eye(3) - Sigma_c

    return mass, com_pos, Ic
get_log_cholesky_batch = jax.vmap(get_log_cholesky, in_axes=(0,))

batched_diag = jax.vmap(jnp.diag)

@struct.dataclass
class MjxDNEAConfig:
    xml_path: str
    epsilon: float = 1e-5
    inertia_epsilon: float = 1e-4
    mass_epsilon: float = 1e-4
    shift: float = 0.0
    act_ld_name: str = 'Softplus'
    softplus_beta: float = 1.0
    idx_3: jnp.ndarray = 0
    tril_indices_3: Tuple[jnp.ndarray, jnp.ndarray] = struct.field(default_factory=tuple)
    dyn_param: str = 'PrincipalTriangular'

def get_config_from_dict(kwargs):
    default_xml_path = Path(__file__).parent.parent.parent.parent / 'data' / 'robot_models' / 'go2' / 'go2.xml'

    lrot_tril_indices_3 = jnp.tril_indices(3)
    lrot_output_size = int((3 ** 2 + 3) / 2)
    lrot_idx_3 = get_idx_triangular(3, lrot_output_size)

    config = MjxDNEAConfig(
                        xml_path=kwargs.get('xml_path', str(default_xml_path.resolve())),
                        epsilon=kwargs.get('diagonal_epsilon', 1e-5),
                        inertia_epsilon=kwargs.get('dnea_inertia_epsilon', 1e-4),
                        mass_epsilon=kwargs.get('dnea_mass_epsilon', 1e-2),
                        shift=kwargs.get('diagonal_shift', 0.0),
                        act_ld_name=kwargs.get('act_ld', 'Softplus'),
                        softplus_beta=kwargs.get('softplus_beta', 1.0),
                        tril_indices_3=lrot_tril_indices_3,
                        idx_3=lrot_idx_3,
                        dyn_param=kwargs.get('dyn_parametrization', 'PrincipalTriangular')
                        )
    
    if 'Principal' in config.dyn_param:
        print(f'Principal Inertia parametrization, method: {config.dyn_param}')
    elif 'Spatial' in config.dyn_param:
        print(f'Spatial Inertia parametrization, method: {config.dyn_param}')

    return config

class MjxDNEA(nn.Module):
    n_dof: int
    config: MjxDNEAConfig

    def setup(self):
        dyn_param_fns = {
            # Principal Inertia Methods
            'PrincipalTriangular': self.get_tri_dyn_params,
            'PrincipalUnconstrained': self.get_unconstrained_dyn_params,
            # Spatial Inertia
            'SpatialCov': self.get_cov_dyn_params,
            'SpatialSpd': self.get_spd_dyn_params,
            'SpatialLogCholesky': self.get_log_chol_params,
        }

        # Load and convert MuJoCo model
        model = mujoco.MjModel.from_xml_path(self.config.xml_path)
        self.raw_mjx_model = mjx.put_model(model)
        self.mjx_model = mjx.put_model(model)
        self.n_bodies = self.mjx_model.body_mass.shape[0] - 1

        init_body_quat = quaternion_wxyz_to_xyzw(self.mjx_model.body_iquat[1:,:])
        init_body_euler = Rotation.from_quat(init_body_quat).as_euler("xyz")

        self.init_values = {
            'body_mass': self.mjx_model.body_mass[1:],
            'body_inertia': self.mjx_model.body_inertia[1:,:],
            'body_ipos': self.mjx_model.body_ipos[1:,:],
            'body_iquat': self.mjx_model.body_iquat[1:,:],
            'body_pos': self.mjx_model.body_pos[1:,:],
            'body_euler': init_body_euler,
        }


        self.get_trainable_dyn_params = dyn_param_fns[self.config.dyn_param]
        if 'Principal' in self.config.dyn_param:
            self.get_dyn_params = self.get_dyn_principal_inertia_params
        elif 'Spatial' in self.config.dyn_param:
            self.get_dyn_params = self.get_dyn_spatial_inertia_params

        self.get_trainable_kin_params = self.get_unconstrained_kin_params

    def get_unconstrained_dyn_params(self):
        body_mass = self.param(
            "body_mass",
            lambda rng: jax.random.uniform(rng, minval=1.0, maxval=10.0, shape=self.init_values['body_mass'].shape)
        )

        body_inertia = self.param(
            "body_inertia",
            lambda rng: jax.random.uniform(rng, shape=self.init_values['body_inertia'].shape)
        )

        body_inertia_pos = self.param(
            "body_inertia_pos",
            lambda rng: jax.random.uniform(rng, minval=-1.0, maxval=1.0, shape=self.init_values['body_ipos'].shape)
        )

        body_inertia_euler = self.param(
            "body_inertia_euler",
            lambda rng: jax.random.uniform(rng, minval=-1.0, maxval=1.0, shape=self.init_values['body_ipos'].shape)
        )

        body_inertia_quat_xyzw = Rotation.from_euler("xyz", body_inertia_euler).as_quat()
        body_inertia_quat = quaternion_xyzw_to_wxyz(body_inertia_quat_xyzw)

        return body_mass, body_inertia, body_inertia_pos, body_inertia_quat
    
    def get_log_chol_params(self):
        log_chol_params = self.param(
            "log_chol_params",
            lambda rng: jax.random.uniform(rng, shape=(self.n_bodies, 10))
        )

        body_mass, body_inertia_pos, body_Ic = get_log_cholesky_batch(log_chol_params)
        return body_mass, body_inertia_pos, body_Ic
    
    def get_tri_dyn_params(self):
        theta_mass = self.param(
            "theta_mass",
            lambda rng: jax.random.uniform(rng, minval=1.0, maxval=3.0, shape=self.init_values['body_mass'].shape)
        )
        body_mass = jnp.power(theta_mass, 2) + self.config.mass_epsilon

        theta_inertia_chol = self.param(
            "theta_inertia_chol",
            lambda rng: jax.random.uniform(rng, shape=(self.n_bodies, 6))
        )
        inertia_densities, inertia_rot_exp_map_input = jnp.split(theta_inertia_chol, [3], axis=-1)
        inertia_densities = jnp.power(inertia_densities, 2) + self.config.inertia_epsilon
    
        J0 = inertia_densities[:,1] + inertia_densities[:,2]
        J1 = inertia_densities[:,0] + inertia_densities[:,2]
        J2 = inertia_densities[:,0] + inertia_densities[:,1]

        body_inertia = jnp.stack([J0, J1, J2], axis=-1)
        body_inertia_quat = batch_expmap_to_quat(inertia_rot_exp_map_input)

        body_inertia_pos = self.param(
            "body_inertia_pos",
            lambda rng: jax.random.uniform(rng, shape=(self.n_bodies,3))
        )

        return body_mass, body_inertia, body_inertia_pos, body_inertia_quat

    def get_cov_dyn_params(self):
        theta_mass = self.param(
            "theta_mass",
            lambda rng: jax.random.uniform(rng, minval=1.0, maxval=3.0, shape=self.init_values['body_mass'].shape)
        )
        body_mass = jnp.power(theta_mass, 2) + self.config.mass_epsilon

        theta_inertia_chol = self.param(
            "theta_inertia_chol",
            lambda rng: jax.random.uniform(rng, shape=(self.n_bodies, 6))
        )

        ##
        l_diagonal, l_off_diagonal = jnp.split(theta_inertia_chol, [3], axis=-1)
        l_diagonal = apply_act_fn(l_diagonal + self.config.shift, self.config.act_ld_name, self.config.softplus_beta) + self.config.epsilon
        # Reassemble L vector
        L_vec = jnp.concatenate([l_diagonal, l_off_diagonal], axis=-1)
        L_vec = L_vec[..., self.config.idx_3]

        # Create L matrix
        L = jnp.zeros((self.n_bodies, 3,3), dtype=jnp.float32)
        L = L.at[:, self.config.tril_indices_3[0], self.config.tril_indices_3[1]].set(L_vec)
        Sigma_c = L @ jnp.transpose(L, (0, 2, 1))
        trace_sigma_c = jnp.trace(Sigma_c, axis1=1, axis2=2)
        body_Ic = trace_sigma_c[:, None, None] * jnp.eye(3) - Sigma_c

        body_inertia_pos = self.param(
            "body_inertia_pos",
            lambda rng: jax.random.uniform(rng, minval=-1.0, maxval=1.0, shape=(self.n_bodies, 3))
        )
        return body_mass, body_inertia_pos, body_Ic
    
    def get_spd_dyn_params(self):
        theta_mass = self.param(
            "theta_mass",
            lambda rng: jax.random.uniform(rng, minval=1.0, maxval=3.0, shape=self.init_values['body_mass'].shape)
        )
        body_mass = jnp.power(theta_mass, 2) + self.config.mass_epsilon

        theta_inertia_chol = self.param(
            "theta_inertia_chol",
            lambda rng: jax.random.uniform(rng, shape=(self.n_bodies, 6))
        )

        ##
        l_diagonal, l_off_diagonal = jnp.split(theta_inertia_chol, [3], axis=-1)
        l_diagonal = apply_act_fn(l_diagonal + self.config.shift, self.config.act_ld_name, self.config.softplus_beta) + self.config.epsilon
        # Reassemble L vector
        L_vec = jnp.concatenate([l_diagonal, l_off_diagonal], axis=-1)
        L_vec = L_vec[..., self.config.idx_3]

        # Create L matrix
        L = jnp.zeros((self.n_bodies, 3,3), dtype=jnp.float32)
        L = L.at[:, self.config.tril_indices_3[0], self.config.tril_indices_3[1]].set(L_vec)
        body_Ic = L @ jnp.transpose(L, (0, 2, 1))

        body_inertia_pos = self.param(
            "body_inertia_pos",
            lambda rng: jax.random.uniform(rng, minval=-1.0, maxval=1.0, shape=(self.n_bodies, 3))
        )
        return body_mass, body_inertia_pos, body_Ic

    def get_dyn_principal_inertia_params(self):
        body_mass, body_inertia, body_inertia_pos, body_inertia_quat = self.get_trainable_dyn_params()

        full_mass = jnp.concatenate([
            jnp.zeros(1), # root link
            body_mass
        ], axis=0)

        full_inertia = jnp.concatenate([
            jnp.zeros((1,3)), # root link
            body_inertia
        ], axis=0)

        full_ipos = jnp.concatenate([
            jnp.zeros((1,3)), # root link
            body_inertia_pos
        ], axis=0)

        full_iquat = jnp.concatenate([
            jnp.array([[1.0,0.0,0.0,0.0]]), # root link
            body_inertia_quat
        ], axis=0)

        xyzw_full_iquat = quaternion_wxyz_to_xyzw(full_iquat)
        full_irot_mat = Rotation.from_quat(xyzw_full_iquat).as_matrix()
        full_spatial_inertia = full_irot_mat @ batched_diag(full_inertia) @ jnp.transpose(full_irot_mat, (0, 2, 1))

        return full_mass, full_ipos, full_spatial_inertia
    
    def get_dyn_spatial_inertia_params(self):
        body_mass, body_inertia_pos, body_spatial_inertia = self.get_trainable_dyn_params()

        full_mass = jnp.concatenate([
            jnp.zeros(1), # root link
            body_mass
        ], axis=0)

        full_ipos = jnp.concatenate([
            jnp.zeros((1,3)), # root link
            body_inertia_pos
        ], axis=0)

        full_spatial_inertia = jnp.concatenate([
            jnp.zeros((1,3,3)), # root link
            body_spatial_inertia
        ], axis=0)

        return full_mass, full_ipos, full_spatial_inertia
    
    def get_unconstrained_kin_params(self):
        body_pos = self.param(
            "body_pos",
            lambda rng: jax.random.uniform(rng, minval=-1.0, maxval=1.0, shape=(self.n_bodies, 3))
        )
        body_euler = self.param(
            "body_euler",
            lambda rng: jax.random.uniform(rng, minval=-1.0, maxval=1.0, shape=(self.n_bodies, 3))
        )
        body_quat = Rotation.from_euler("xyz", body_euler).as_quat()
        body_quat = quaternion_xyzw_to_wxyz(body_quat)

        return body_pos, body_quat

    def get_kin_params(self):
        body_pos, body_quat = self.get_trainable_kin_params()

        full_body_pos = jnp.concatenate([
            jnp.zeros((1,3)), # root link
            body_pos
        ], axis=0)

        full_body_quat = jnp.concatenate([
            jnp.array([[1.0,0.0,0.0,0.0]]), # root link
            body_quat
        ], axis=0)

        return full_body_pos, full_body_quat

    def dyn_model(self, q, qd, qdd):

        full_mass, full_ipos, full_spatial_inertia = self.get_dyn_params()

        # Replace model body mass
        if self.get_trainable_kin_params is None:
            model = self.mjx_model.replace(body_mass=full_mass,
                                           body_ipos=full_ipos)
        
        else:
            full_body_pos, full_body_quat = self.get_kin_params()
            model = self.mjx_model.replace(body_pos=full_body_pos,
                                           body_quat=full_body_quat,
                                           body_mass=full_mass,
                                           body_ipos=full_ipos)

        # Define the actual dynamics function (no params inside)
        def dyn_fn(q_, qd_, qdd_):
            data = mjx.make_data(model)
            data = data.replace(qpos=q_, qvel=qd_)
            data = fwd_position_inertia(model, data, full_spatial_inertia)
            data = mjx.fwd_velocity(model, data)
            return data.qM @ qdd_ + data.qfrc_bias
        tau_pred = jax.vmap(dyn_fn)(q, qd, qdd)

        dEdt = jnp.sum(qd * tau_pred, axis=1)

        return tau_pred.squeeze(), dEdt

    @nn.compact
    def __call__(self, q, qd, qdd):
        out = self.dyn_model(q, qd, qdd)
        tau_pred = out[0]
        dEdt = out[1]
        extras = {}
        return tau_pred, dEdt, extras