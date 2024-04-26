"""
Collection of utility functions for computing kinematic and probabilistic
functions using the CasADi symbolic framework.
"""

import casadi as ca
import numpy as np


def gaussian_pdf(x, mu, sigma, zero_lim=1e-25):
    """
    Computes the Gaussian probability density function value.
    """
    d = mu.shape[0]
    coef = 1.0 / ca.sqrt((2.0 * np.pi)**d * ca.det(sigma))
    exp = ca.exp(-0.5 * ca.mtimes(
        (x - mu).T, ca.mtimes(ca.inv(sigma), x - mu)))
    # return ca.fmax(coef * exp, zero_lim)
    return coef * exp


def mahalanobis_distance(x, mu, sigma):
    """
    Computes Mahalanobis distance between a sample and a Gaussian distribution.
    """
    diff = x - mu
    inv = ca.inv(sigma)
    dist = ca.sqrt(ca.mtimes(diff.T, ca.mtimes(inv, diff)))
    return dist


def quaternion_rotation_matrix(q):
    """
    See "Visualizing Quaternions" by Andrew Hanson, pg. 63
    """
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q0 = q[3]
    Q = np.array([[q0, -q1, -q2, -q3], [q1, q0, -q3, q2], [q2, q3, q0, -q1],
                  [q3, -q2, q1, q0]])
    return Q


def quaternion_to_tf(q):
    """
    Converts a quaternion to a homogeneous transformation matrix.
    """
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q0 = q[3]
    tf = ca.SX.zeros(4, 4)
    tf[0, 0] = q0**2 + q1**2 - q2**2 - q3**2
    tf[0, 1] = 2 * (q1 * q2 - q0 * q3)
    tf[0, 2] = 2 * (q1 * q3 + q0 * q2)
    tf[1, 0] = 2 * (q1 * q2 + q0 * q3)
    tf[1, 1] = q0**2 - q1**2 + q2**2 - q3**2
    tf[1, 2] = 2 * (q2 * q3 - q0 * q1)
    tf[2, 0] = 2 * (q1 * q3 - q0 * q2)
    tf[2, 1] = 2 * (q2 * q3 + q0 * q1)
    tf[2, 2] = q0**2 - q1**2 - q2**2 + q3**2
    tf[3, 3] = 1
    return tf


def tf_to_quaternion(tf):
    """
    Converts a homogeneous transformation matrix to a quaternion.
    """
    r11 = tf[0, 0]
    r12 = tf[0, 1]
    r13 = tf[0, 2]
    r21 = tf[1, 0]
    r22 = tf[1, 1]
    r23 = tf[1, 2]
    r31 = tf[2, 0]
    r32 = tf[2, 1]
    r33 = tf[2, 2]

    q = ca.SX.zeros(4)

    q0q0 = 0.25 * (1 + r11 + r22 + r33)
    q1q1 = 0.25 * (1 + r11 - r22 - r33)
    q2q2 = 0.25 * (1 - r11 + r22 - r33)
    q3q3 = 0.25 * (1 - r11 - r22 + r33)

    q0q1 = 0.25 * (r32 - r23)
    q0q2 = 0.25 * (r13 - r31)
    q0q3 = 0.25 * (r21 - r12)
    q1q2 = 0.25 * (r12 + r21)
    q1q3 = 0.25 * (r13 + r31)
    q2q3 = 0.25 * (r23 + r32)

    choose_q0 = ca.logic_and(q0q0 >= q1q1,
                             ca.logic_and(q0q0 >= q2q2, q0q0 >= q3q3))
    choose_q1 = ca.logic_and(q1q1 >= q0q0,
                             ca.logic_and(q1q1 >= q2q2, q1q1 >= q3q3))
    choose_q2 = ca.logic_and(q2q2 >= q0q0,
                             ca.logic_and(q2q2 >= q1q1, q2q2 >= q3q3))
    choose_q3 = ca.logic_and(q3q3 >= q0q0,
                             ca.logic_and(q3q3 >= q1q1, q3q3 >= q2q2))

    q[0] = ca.if_else(choose_q0, ca.sqrt(q0q0), q[0])
    q[1] = ca.if_else(choose_q0, q0q1 / q[0], q[1])
    q[2] = ca.if_else(choose_q0, q0q2 / q[0], q[2])
    q[3] = ca.if_else(choose_q0, q0q3 / q[0], q[3])

    q[1] = ca.if_else(choose_q1, ca.sqrt(q1q1), q[1])
    q[0] = ca.if_else(choose_q1, q0q1 / q[1], q[0])
    q[2] = ca.if_else(choose_q1, q1q2 / q[1], q[2])
    q[3] = ca.if_else(choose_q1, q1q3 / q[1], q[3])

    q[2] = ca.if_else(choose_q2, ca.sqrt(q2q2), q[2])
    q[0] = ca.if_else(choose_q2, q0q2 / q[2], q[0])
    q[1] = ca.if_else(choose_q2, q1q2 / q[2], q[1])
    q[3] = ca.if_else(choose_q2, q2q3 / q[2], q[3])

    q[3] = ca.if_else(choose_q3, ca.sqrt(q3q3), q[3])
    q[0] = ca.if_else(choose_q3, q0q3 / q[3], q[0])
    q[1] = ca.if_else(choose_q3, q1q3 / q[3], q[1])
    q[2] = ca.if_else(choose_q3, q2q3 / q[3], q[2])

    # TODO hack to get in correct order x,y,z,w
    q_mod = ca.SX.zeros(4)
    q_mod[0] = q[1]
    q_mod[1] = q[2]
    q_mod[2] = q[3]
    q_mod[3] = q[0]
    return q_mod


def tf_to_position(tf):
    """
    Extracts the position components from a homogeneous transformation matrix.
    """
    return tf[:3, 3]


def TF_mu(base_TF_object, mu, suffix=""):
    """
    Creates symbolic transformation for the end-effector pose in the base
    frame given the pose of the object in the base frame.
    """
    base_pose_ee = ca.SX.sym("base_mu_ee_{}".format(suffix), 7)
    object_q_ee = mu[3:]
    object_q_ee /= ca.norm_2(object_q_ee)
    object_TF_ee = quaternion_to_tf(object_q_ee)
    object_TF_ee[0, 3] = mu[0]
    object_TF_ee[1, 3] = mu[1]
    object_TF_ee[2, 3] = mu[2]
    base_TF_ee = ca.mtimes(base_TF_object, object_TF_ee)
    base_q_ee = tf_to_quaternion(base_TF_ee)
    base_pos_ee = tf_to_position(base_TF_ee)
    for i in range(3):
        base_pose_ee[i] = base_pos_ee[i]
    for i in range(4):
        base_pose_ee[i + 3] = base_q_ee[i]
    return base_pose_ee


def TF_sigma(base_TF_object, sigma):
    """
    Applies rotation based on quaternions. 7x7 block-diagonal matrix, where 
    first block is 3x3 position rotation (standard rotation matrix derived from 
    quaternion) and the second block is 4x4 quaternion rotation (interpreting 
    quaternion as a real-valued 4x4 matrix by appropriately setting the 
    components essentially as composition of unit quaternions).

    See "Loose-limbed People: Estimating 3D Human Pose and Motion Using 
    Non-parametric Belief Propagation", Sigal et al. 2012 Appendix A.
    """
    M = ca.SX.zeros(7, 7)
    base_q_object = tf_to_quaternion(base_TF_object)
    T = quaternion_to_tf(base_q_object)
    M[:3, :3] = T[:3, :3]
    M[3:, 3:] = quaternion_rotation_matrix(base_q_object)
    return ca.mtimes(M, ca.mtimes(sigma, M.T))


def euler_to_quaternion(roll, pitch, yaw):
    """
    Converts RPY Euler angles to quaternion. 

    See https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    
    Args:
        roll (float): Roll angle about x-axis
        pitch (float): Pitch angle about y-axis
        yaw (float): Yaw angle about z-axis
    """
    cy = ca.cos(yaw * 0.5)
    sy = ca.sin(yaw * 0.5)
    cp = ca.cos(pitch * 0.5)
    sp = ca.sin(pitch * 0.5)
    cr = ca.cos(roll * 0.5)
    sr = ca.sin(roll * 0.5)
    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = sy * cp * sr + cy * sp * cr
    qz = sy * cp * cr - cy * sp * sr
    return qx, qy, qz, qw


def euler_yaw_to_tf(yaw):
    """
    Converts yaw Euler angle to a homogenous transformation matrix.
    """
    tf = ca.SX.zeros(4, 4)
    tf[0, 0] = ca.cos(yaw)
    tf[0, 1] = -ca.sin(yaw)
    tf[1, 0] = ca.sin(yaw)
    tf[1, 1] = ca.cos(yaw)
    tf[2, 2] = 1
    tf[3, 3] = 1
    return tf


def table_planar_to_table_tf(obj_x, obj_y, obj_z, obj_theta):
    """
    Converts a planar pose to a homogeneous transformation matrix.
    """
    tf = euler_yaw_to_tf(obj_theta)
    tf[0, 3] = obj_x
    tf[1, 3] = obj_y
    tf[2, 3] = obj_z
    return tf


def table_planar_to_base_tf(obj_x, obj_y, obj_z, obj_theta, table_position, table_quaternion):
    """
    Converts a planar pose to a homogenous transformation matrix and
    applies a transformation to put it in the base frame.
    """
    base_TF_table = quaternion_to_tf(table_quaternion)
    base_TF_table[0, 3] = table_position[0]
    base_TF_table[1, 3] = table_position[1]
    base_TF_table[2, 3] = table_position[2]
    table_TF_object = table_planar_to_table_tf(obj_x, obj_y, obj_z, obj_theta)
    base_TF_object = ca.mtimes(base_TF_table, table_TF_object)
    return base_TF_object
