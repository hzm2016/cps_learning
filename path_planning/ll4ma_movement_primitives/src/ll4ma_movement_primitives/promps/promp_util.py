import numpy as np
from src.ll4ma_movement_primitives.promps import Waypoint, ProMPConfig
# from ll4ma_movement_primitives.msg import ProMPConfigROS, WaypointROS, Row


def python_to_ros_config(promp_config):
    ros_config = ProMPConfigROS()
    ros_config.state_types = promp_config.state_types
    ros_config.num_bfs = promp_config.num_bfs
    ros_config.w_keys = promp_config.w_keys
    ros_config.inv_wish_factor = promp_config.inv_wish_factor
    ros_config.dist_threshold = promp_config.dist_threshold
    ros_config.num_demos = promp_config.num_demos
    ros_config.init_joint_state = promp_config.init_joint_state
    ros_config.fixed_mahalanobis_threshold = promp_config.fixed_mahalanobis_threshold
    if promp_config.mu_w_mle is not None:
        ros_config.mu_w_mle = promp_config.mu_w_mle.flatten().tolist()
    # Pack dimension lists
    for dim_list in promp_config.dimensions:
        row = Row()
        row.elements = dim_list
        ros_config.dimensions.rows.append(row)
    # Pack covariance MLE and prior
    for i in range(promp_config.num_dims * promp_config.num_bfs):
        # MLE
        if np.any(promp_config.sigma_w_mle):
            row = Row()
            row.elements = promp_config.sigma_w_mle[i, :].flatten().tolist()
            ros_config.sigma_w_mle.rows.append(row)
        # Prior
        if np.any(promp_config.sigma_w_prior):
            row = Row()
            row.elements = promp_config.sigma_w_prior[i, :].flatten().tolist()
            ros_config.sigma_w_prior.rows.append(row)
    return ros_config


def ros_to_python_config(ros_config):
    promp_config = ProMPConfig()
    promp_config.state_types = ros_config.state_types
    promp_config.num_bfs = ros_config.num_bfs
    promp_config.dt = ros_config.dt
    promp_config.mu_w_mle = np.array(ros_config.mu_w_mle)
    promp_config.w_keys = ros_config.w_keys
    promp_config.inv_wish_factor = ros_config.inv_wish_factor
    promp_config.dist_threshold = ros_config.dist_threshold
    promp_config.num_demos = ros_config.num_demos
    promp_config.init_joint_state = ros_config.init_joint_state
    promp_config.fixed_mahalanobis_threshold = ros_config.fixed_mahalanobis_threshold
    # Unpack covariance MLE
    sigma_w_mle = None
    for row in ros_config.sigma_w_mle.rows:
        r = np.array(row.elements)
        sigma_w_mle = r if sigma_w_mle is None else np.vstack((sigma_w_mle, r))
    promp_config.sigma_w_mle = sigma_w_mle
    # Unpack prior covariance
    sigma_w_prior = None
    for row in ros_config.sigma_w_prior.rows:
        r = np.array(row.elements)
        sigma_w_prior = r if sigma_w_prior is None else np.vstack(
            (sigma_w_prior, r))
    # Unpack dimension lists
    for row in ros_config.dimensions.rows:
        promp_config.dimensions.append(row.elements)
    return promp_config


def ros_to_python_waypoint(ros_wpt):
    python_wpt = Waypoint()
    python_wpt.values = ros_wpt.values
    python_wpt.time = ros_wpt.time
    python_wpt.phase_val = ros_wpt.phase_val
    python_wpt.condition_keys = ros_wpt.condition_keys
    # Unpack observation covariance
    python_wpt.sigma = None
    for row in ros_wpt.sigma.rows:
        r = np.array(row.elements)
        python_wpt.sigma = r if python_wpt.sigma is None else np.vstack(
            (python_wpt.sigma, r))
    return python_wpt


def python_to_ros_waypoint(python_wpt):
    ros_wpt = WaypointROS()
    ros_wpt.time = python_wpt.time
    ros_wpt.phase_val = python_wpt.phase_val
    ros_wpt.values = python_wpt.values
    ros_wpt.condition_keys = python_wpt.condition_keys
    # Pack observation covariance
    for i in range(python_wpt.sigma.shape[0]):
        row = Row()
        row.elements = python_wpt.sigma[i, :].flatten().tolist()
        ros_wpt.sigma.rows.append(row)
    return ros_wpt


def damped_pinv(A, rho=0.017):
    AA_T = np.dot(A, A.T)
    damping = np.eye(A.shape[0]) * rho**2
    inv = np.linalg.inv(AA_T + damping)
    d_pinv = np.dot(A.T, inv)
    return d_pinv


def mahalanobis_distance(y, mu, sigma):
    diff = y - mu
    inv = np.linalg.pinv(sigma)
    return np.sqrt(np.dot(diff.T, np.dot(inv, diff))).flatten()


def reject_outliers(data, m=3.0):
    """
    From this stackoverflow post: https://stackoverflow.com/questions/11686720
    3.5 is a common value, but for our purposes it's okay to throw out more outliers,
    gives a more reliable and conservative confidence threshold.
    """
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    return data[s < m]


def dist_to_singular(sigma):
    return np.linalg.norm(sigma, ord='fro') / np.linalg.cond(sigma, p='fro')


def smallest_sing_val(sigma):
    U, S, V = np.linalg.svd(sigma)
    return np.min(S)
