import numpy as np
from scipy.spatial.transform import Rotation


def get_omega_to_euler_rates(euler_angle):
    roll, pitch, yaw = euler_angle
    Wn_inv = np.eye(3)
    Wn_inv[0,1] = np.sin(roll) * np.tan(pitch)
    Wn_inv[0,2] = np.cos(roll) * np.tan(pitch)
    Wn_inv[1,1] = np.cos(roll)
    Wn_inv[1,2] = -np.sin(roll)
    Wn_inv[2,1] = np.sin(roll) / np.cos(pitch)
    Wn_inv[2,2] = np.cos(roll) / np.cos(pitch)
    return Wn_inv

def convert_omega_to_euler_rates(euler_angle, omega):
    Wn_inv = get_omega_to_euler_rates(euler_angle)
    return Wn_inv @ omega

def get_euler_from_mj_quat(mj_quat):
        def quaternion_wxyz_to_xyzw(quat):
            # return np.hstack([quat[:,1:], quat[:,0].reshape((-1,1))])
            return np.concatenate([quat[..., 1:], quat[..., :1]], axis=-1)
        """
        Rotation of the base in euler zyx angles.
        """
        reshaped_quat = mj_quat.reshape(-1, 4)
        euler_angles = Rotation.from_quat(
            quaternion_wxyz_to_xyzw(reshaped_quat)
        ).as_euler("xyz")
        return euler_angles.reshape(mj_quat.shape[:-1] + (3,))

def get_euler_from_pin_quat(pin_quat):
        reshaped_quat = pin_quat.reshape(-1, 4)
        euler_angles = Rotation.from_quat(
            reshaped_quat
        ).as_euler("xyz")
        return euler_angles.reshape(pin_quat.shape[:-1] + (3,))

def get_mj_quat_from_pin_quat(pin_quat):
    return np.concatenate([pin_quat[..., 3:], pin_quat[..., :3]], axis=-1)

def get_rot_base_to_world(euler_angles):
    return get_rot_world_to_base(euler_angles).T

def get_rot_world_to_base(euler_angles):
    roll, pitch, yaw = euler_angles[0], euler_angles[1], euler_angles[2]

    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

    R_roll = np.array([
        [1.0,      0.0,       0.0],
        [0.0, cos_roll,  sin_roll],
        [0.0, -sin_roll, cos_roll]
    ])

    R_pitch = np.array([
        [cos_pitch, 0.0, -sin_pitch],
        [0.0,       1.0,      0.0],
        [sin_pitch, 0.0,  cos_pitch]
    ])

    R_yaw = np.array([
        [ cos_yaw, sin_yaw, 0.0],
        [-sin_yaw, cos_yaw, 0.0],
        [     0.0,     0.0, 1.0]
    ])

    # Combined rotation matrix: R = R_roll @ R_pitch @ R_yaw
    rotation_matrix = R_roll @ R_pitch @ R_yaw

    return rotation_matrix

def get_skew_sym_mat(v):
    v0, v1, v2 = v
    return np.array([[0.0, -v2, v1],
                     [v2, 0.0, -v0],
                     [-v1, v0, 0.0]])

def convert_ang_acc_from_base_to_world(ang_acc_B, omega_B, rot_base_to_world):
    skew_omega_B = get_skew_sym_mat(omega_B)
    ang_acc_W = rot_base_to_world @ skew_omega_B @ omega_B + rot_base_to_world @ ang_acc_B
    return ang_acc_W