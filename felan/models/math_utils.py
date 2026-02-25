
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

def split_by_lengths(x: jnp.ndarray,
                     lengths: List[int],
                     axis: int = -1
                    ) -> List[jnp.ndarray]:
    # compute split points as cumulative sums (dropping the last total)
    cuts = np.cumsum(np.array(lengths))[:-1].tolist()
    return jnp.split(x, cuts, axis=axis)

def get_rot_base_to_world(euler_angles):
    R = get_rot_world_to_base(euler_angles)
    if R.ndim == 2:
        return get_rot_world_to_base(euler_angles).T
    else:
        return jnp.transpose(R, axes=(0, 2, 1))

def get_rot_world_to_base(euler_angles: jnp.ndarray) -> jnp.ndarray:
    roll, pitch, yaw = (euler_angles[..., i] for i in range(3))

    cos_roll, sin_roll = jnp.cos(roll), jnp.sin(roll)
    cos_pitch, sin_pitch = jnp.cos(pitch), jnp.sin(pitch)
    cos_yaw, sin_yaw = jnp.cos(yaw), jnp.sin(yaw)
    zero = jnp.zeros_like(roll)
    one = jnp.ones_like(roll)

    # R_roll
    row0 = jnp.stack([ one,       zero,     zero], axis=-1)
    row1 = jnp.stack([ zero,  cos_roll, sin_roll], axis=-1)
    row2 = jnp.stack([ zero, -sin_roll, cos_roll], axis=-1)
    R_roll = jnp.stack([row0, row1, row2], axis=-2)

    # R_pitch
    row0 = jnp.stack([ cos_pitch, zero, -sin_pitch], axis=-1)
    row1 = jnp.stack([      zero,  one,       zero], axis=-1)
    row2 = jnp.stack([ sin_pitch, zero,  cos_pitch], axis=-1)
    R_pitch = jnp.stack([row0, row1, row2], axis=-2)

    # R_yaw
    row0 = jnp.stack([  cos_yaw, sin_yaw, zero], axis=-1)
    row1 = jnp.stack([ -sin_yaw, cos_yaw, zero], axis=-1)
    row2 = jnp.stack([     zero,    zero,  one], axis=-1)
    R_yaw = jnp.stack([row0, row1, row2], axis=-2)

    return R_roll @ R_pitch @ R_yaw

def euler_rate_to_omega_B_mat(euler_angles):
    roll, pitch, yaw = (euler_angles[..., i] for i in range(3))

    cos_roll, sin_roll = jnp.cos(roll), jnp.sin(roll)
    cos_pitch, sin_pitch = jnp.cos(pitch), jnp.sin(pitch)
    zero = jnp.zeros_like(roll)
    one = jnp.ones_like(roll)

    row0 = jnp.stack([  one,      zero,         -sin_pitch], axis=-1)
    row1 = jnp.stack([ zero,  cos_roll, sin_roll*cos_pitch], axis=-1)
    row2 = jnp.stack([ zero, -sin_roll, cos_roll*cos_pitch], axis=-1)
    Wn_inv = jnp.stack([row0, row1, row2], axis=-2)
    return Wn_inv

def omega_B_to_euler_rate_mat(euler_angles):
    roll, pitch, yaw = (euler_angles[..., i] for i in range(3))

    cos_roll, sin_roll = jnp.cos(roll), jnp.sin(roll)
    cos_pitch, tan_pitch = jnp.cos(pitch), jnp.tan(pitch)
    zero = jnp.zeros_like(roll)
    one = jnp.ones_like(roll)

    row0 = jnp.stack([  one, sin_roll * tan_pitch, cos_roll * tan_pitch], axis=-1)
    row1 = jnp.stack([ zero,             cos_roll,            -sin_roll], axis=-1)
    row2 = jnp.stack([ zero, sin_roll / cos_pitch, cos_roll / cos_pitch], axis=-1)
    Wn = jnp.stack([row0, row1, row2], axis=-2)

    return Wn

def skew_sym_matrix(v):
    v0, v1, v2 = v[..., 0], v[..., 1], v[..., 2]
    zero = jnp.zeros_like(v0)
    shape = v.shape[:-1] + (3, 3)
    S = jnp.zeros(shape, dtype=v.dtype)
    S = S.at[..., 0, 1].set(-v2)
    S = S.at[..., 0, 2].set( v1)
    S = S.at[..., 1, 0].set( v2)
    S = S.at[..., 1, 2].set(-v0)
    S = S.at[..., 2, 0].set(-v1)
    S = S.at[..., 2, 1].set( v0)
    return S

def get_vector_from_skew(S: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack([S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]], axis=-1)[..., None]

@jax.jit
def reordered_cholesky(H_input: jnp.ndarray) -> jnp.ndarray:
    # only keep the lower triangle of H_input
    H = jnp.tril(H_input)

    # Step k=2
    L22 = jnp.sqrt(H[..., 2, 2])
    L21 = H[..., 2, 1] / L22
    L20 = H[..., 2, 0] / L22

    H11_new = H[..., 1, 1] - L21 * L21
    H10_new = H[..., 1, 0] - L21 * L20
    H00_int  = H[..., 0, 0] - L20 * L20

    # Step k=1
    L11 = jnp.sqrt(H11_new)
    L10 = H10_new / L11
    H00_new = H00_int - L10 * L10

    # Step k=0
    L00 = jnp.sqrt(H00_new)

    # --- assemble rows of L ---
    zero = jnp.zeros_like(L00)
    row0 = jnp.stack([L00, zero, zero], axis=-1)          # (..., 3)
    row1 = jnp.stack([L10, L11,  zero], axis=-1)
    row2 = jnp.stack([L20, L21,  L22  ], axis=-1)

    # final L: shape (..., 3, 3)
    return jnp.stack([row0, row1, row2], axis=-2)

def check_triangle_inequality(matrix, tol = 1e-5):
    """
    Checks if the eigenvalues of a 3x3 symmetric matrix satisfy the triangle inequality:
        l1 ≤ l2 + l3
        l2 ≤ l1 + l3
        l3 ≤ l1 + l2
    Assumes symmetric matrix
    """
    # Compute eigenvalues (jax.numpy.linalg.eigh returns sorted eigenvalues)
    eigenvalues = jnp.linalg.eigh(matrix)[0]  # eigh returns (eigenvalues, eigenvectors)
    l1, l2, l3 = eigenvalues

    # Check triangle inequality
    satisfied = (
        (l1 <= l2 + l3 + tol) &
        (l2 <= l1 + l3 + tol) &
        (l3 <= l1 + l2 + tol)
    )

    return satisfied

def eig(input_mat):
    return jnp.real(jnp.linalg.eigvalsh(input_mat))

def apply_input_tf(input):
    return jnp.concatenate([jnp.cos(input), jnp.sin(input)], axis=-1)

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