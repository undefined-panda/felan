import functools

import jax
from jax import numpy as jp
import mujoco
from mujoco.mjx._src import collision_driver
from mujoco.mjx._src import constraint
from mujoco.mjx._src import derivative
from mujoco.mjx._src import math
from mujoco.mjx._src import passive
from mujoco.mjx._src import scan
from mujoco.mjx._src import sensor
from mujoco.mjx._src import smooth
from mujoco.mjx._src import solver
from mujoco.mjx._src import support
# pylint: disable=g-importing-member
from mujoco.mjx._src.types import BiasType
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import DataJAX
from mujoco.mjx._src.types import DisableBit
from mujoco.mjx._src.types import DynType
from mujoco.mjx._src.types import GainType
from mujoco.mjx._src.types import IntegratorType
from mujoco.mjx._src.types import JointType
from mujoco.mjx._src.types import Model
from mujoco.mjx._src.types import ModelJAX
from mujoco.mjx._src.types import TrnType
# pylint: enable=g-importing-member
import numpy as np

# RK4 tableau
_RK4_A = np.array([
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 1.0],
])
_RK4_B = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])


def named_scope(fn, name: str = ''):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with jax.named_scope(name or getattr(fn, '__name__')):
            res = fn(*args, **kwargs)
        return res

    return wrapper


@named_scope
def default_fwd_position(m: Model, d: Data) -> Data:
  """Position-dependent computations."""
  # TODO(robotics-simulation): tendon
  d = smooth.kinematics(m, d)
  d = smooth.com_pos(m, d)
  d = smooth.camlight(m, d)
  d = smooth.tendon(m, d)
  d = smooth.crb(m, d)
  d = smooth.tendon_armature(m, d)
  d = smooth.factor_m(m, d)
  d = collision_driver.collision(m, d)
  d = constraint.make_constraint(m, d)
  d = smooth.transmission(m, d)
  return d

@named_scope
def fwd_position_inertia(m: Model, d: Data, body_spatial_inertia) -> Data:
  """Position-dependent computations."""
  # TODO(robotics-simulation): tendon
  d = smooth.kinematics(m, d)
  d = new_com_pos(m, d, body_spatial_inertia)
  d = smooth.camlight(m, d)
  d = smooth.tendon(m, d)
  d = smooth.crb(m, d)
  d = smooth.tendon_armature(m, d)
  d = smooth.factor_m(m, d)
  d = collision_driver.collision(m, d)
  d = constraint.make_constraint(m, d)
  d = smooth.transmission(m, d)
  return d


def new_com_pos(m: Model, d: Data, body_spatial_inertia) -> Data:
  """Maps inertias and motion dofs to global frame centered at subtree-CoM."""
  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
    raise ValueError('com_pos requires JAX backend implementation.')

  # calculate center of mass of each subtree
  def subtree_sum(carry, xipos, body_mass):
    pos, mass = xipos * body_mass, body_mass
    if carry is not None:
      subtree_pos, subtree_mass = carry
      pos, mass = pos + subtree_pos, mass + subtree_mass
    return pos, mass

  pos, mass = scan.body_tree(
      m, subtree_sum, 'bb', 'bb', d.xipos, m.body_mass, reverse=True
  )
  cond = jp.tile(mass < mujoco.mjMINVAL, (3, 1)).T
  # take maximum to avoid NaN in gradient of jp.where
  subtree_com = jax.vmap(jp.divide)(pos, jp.maximum(mass, mujoco.mjMINVAL))
  subtree_com = jp.where(cond, d.xipos, subtree_com)
  d = d.replace(subtree_com=subtree_com)

  @jax.vmap
  def inert_com(spatial_inert, world_quat, off, mass, org_inertia, ord_ximat):
    org_inert = math.matmul_unroll((ord_ximat * org_inertia), ord_ximat.T)
    h = jp.cross(off, -jp.eye(3))
    rot_mat_world = math.quat_to_mat(world_quat)
    inert = rot_mat_world @ spatial_inert @ jp.transpose(rot_mat_world, (1, 0))
    inert += math.matmul_unroll(h, h.T) * mass
    inert = inert[([0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2])]
    return jp.concatenate([inert, off * mass, mass[None]])

  root_com = subtree_com[m.body_rootid]
  offset = d.xipos - root_com
  cinert = inert_com(body_spatial_inertia, d.xquat, offset, m.body_mass, m.body_inertia, d.ximat)
  d = d.tree_replace({'_impl.cinert': cinert})

  # map motion dofs to global frame centered at subtree_com
  def cdof_fn(jnt_typs, root_com, xmat, xanchor, xaxis):
    cdofs = []

    dof_com_fn = lambda a, o: jp.concatenate([a, jp.cross(a, o)])

    for i, jnt_typ in enumerate(jnt_typs):
      offset = root_com - xanchor[i]
      if jnt_typ == JointType.FREE:
        cdofs.append(jp.eye(3, 6, 3))  # free translation
        cdofs.append(jax.vmap(dof_com_fn, in_axes=(0, None))(xmat.T, offset))
      elif jnt_typ == JointType.BALL:
        cdofs.append(jax.vmap(dof_com_fn, in_axes=(0, None))(xmat.T, offset))
      elif jnt_typ == JointType.HINGE:
        cdof = dof_com_fn(xaxis[i], offset)
        cdofs.append(jp.expand_dims(cdof, 0))
      elif jnt_typ == JointType.SLIDE:
        cdof = jp.concatenate((jp.zeros((3,)), xaxis[i]))
        cdofs.append(jp.expand_dims(cdof, 0))
      else:
        raise RuntimeError(f'unrecognized joint type: {jnt_typ}')

    cdof = jp.concatenate(cdofs) if cdofs else jp.empty((0, 6))

    return cdof

  cdof = scan.flat(
      m,
      cdof_fn,
      'jbbjj',
      'v',
      m.jnt_type,
      root_com,
      d.xmat,
      d.xanchor,
      d.xaxis,
  )
  d = d.tree_replace({'_impl.cdof': cdof})

  return d