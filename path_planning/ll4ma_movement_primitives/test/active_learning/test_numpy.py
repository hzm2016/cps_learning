import os
import rospy
import numpy as np
import cPickle as pickle
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal
from sklearn.mixture import BayesianGaussianMixture as DPGMM
from tf import transformations as tf
from ll4ma_movement_primitives.promps import ProMP, ros_to_python_config
from rospy_service_helper import get_task_trajectories, learn_promps, learn_weights

_OBJ_POSE_DOF = 2
_POSE_3D_DOF = 7
_WEIGHT_DOF = 10


def gaussian_pdf(x, mu, sigma, zero_lim=1e-4):
    try:
        mvn = multivariate_normal(mean=mu, cov=sigma)
        return max(mvn.pdf(x), zero_lim)
    except np.linalg.LinAlgError:
        return zero_lim


def quaternion_rotation_matrix(q):
    """
    See "Visualizing Quaternions" by Andrew Hanson, pg. 63
    """
    q1, q2, q3, q0 = q
    Q = np.array([[q0, -q1, -q2, -q3], [q1, q0, -q3, q2], [q2, q3, q0, -q1],
                  [q3, -q2, q1, q0]])
    return Q


def TF_mu(tf_pose, mu):
    tf_pos = tf_pose[:3]
    tf_quat = tf_pose[3:]
    T = tf.quaternion_matrix(tf_quat)
    for i in range(3):
        T[i, 3] = tf_pos[i]
    orig_pos = mu[:3]
    orig_quat = mu[3:]
    orig_pos = np.insert(orig_pos, orig_pos.size, 1)  # For homogenous multiply
    new_pos = np.dot(T, orig_pos)[:3]
    Q = quaternion_rotation_matrix(tf_quat)
    new_quat = np.dot(Q, orig_quat)
    return np.hstack((new_pos, new_quat))


def TF_sigma(x, sigma):
    """
    Applies rotation based on quaternions. 7x7 block-diagonal matrix, where 
    first block is 3x3 position rotation (standard rotation matrix derived from 
    quaternion) and the second block is 4x4 quaternion rotation (interpreting 
    quaternion as a real-valued 4x4 matrix by appropriately setting the 
    components essentially as composition of unit quaternions).

    See "Loose-limbed People: Estimating 3D Human Pose and Motion Using 
    Non-parametric Belief Propagation", Sigal et al. 2012 Appendix A.
    """
    pos = x[:3]
    quat = x[3:]
    T = tf.quaternion_matrix(quat)
    R = T[:3, :][:, :3]
    Q = quaternion_rotation_matrix(quat)
    M = block_diag(R, Q)
    return np.dot(M, np.dot(sigma, M.T))


def pose_from_planar_position(x, y):
    return np.array([x, y, 0, 0, 0, 0, 1])


def objective_function(
        obj_pose,
        psi,
        alpha_ks,
        mu_ks,
        sigma_ks,  # GMM negative instances
        pi_js,
        mu_js,
        sigma_js,  # ProMP dists
        beta_rs,
        mu_rs,
        sigma_rs):  # GMM ee in obj frame
    """
    Minimize negative entropy.
    """

    # GMM over negative instances
    gmm_obj_prob = 0
    # for alpha_k, mu_k, sigma_k in zip(alpha_ks, mu_ks, sigma_ks):
    #     gmm_k = alpha_k * gaussian_pdf(obj_pose, mu_k, sigma_k)
    #     gmm_obj_prob += gmm_k

    # ProMP
    promp_prob = 0
    pose = pose_from_planar_position(obj_pose[0], obj_pose[1])
    for pi_j, mu_j, sigma_j in zip(pi_js, mu_js, sigma_js):
        for beta_r, mu_r, sigma_r in zip(beta_rs, mu_rs, sigma_rs):
            # Condition ProMP distributions on transformed point
            L_j = np.dot(
                np.dot(sigma_j, psi),
                np.linalg.inv(
                    TF_sigma(pose, sigma_r) +
                    np.dot(psi.T, np.dot(sigma_j, psi))))

            # mu_new_j = mu_j + np.dot(L_j,
            #                          TF_mu(pose, mu_r) - np.dot(psi.T, mu_j))

            # sigma_new_j = sigma_j - np.dot(L_j, np.dot(psi.T, sigma_j))

            # prob_j = pi_j * beta_r * gaussian_pdf(
            #     TF_mu(pose, mu_r), np.dot(psi.T, mu_new_j),
            #     np.dot(psi.T, np.dot(sigma_new_j, psi)))

            prob_j = pi_j * beta_r * gaussian_pdf(
                TF_mu(pose, mu_r), np.dot(psi.T, mu_j),
                np.dot(psi.T, np.dot(sigma_j, psi)))

            print "PROB J", prob_j

            promp_prob += prob_j

    # Want to find maximum entropy point, so we minimize its negation
    neg_entropy = gmm_obj_prob * np.log(gmm_obj_prob) + promp_prob * np.log(
        promp_prob)

    return neg_entropy


rospy.init_node('test_numpy')

# Read in the active learner
data_dir = "/media/adam/data_haro/test_optimization/demos"
learner_filename = "backup/active_learner_047.pkl"
with open(os.path.join(data_dir, learner_filename), 'r') as f:
    learner = pickle.load(f)

# Get the object GMM parameters
alpha_ks = learner.gmm.weights
mu_ks = learner.gmm.mus
sigma_ks = learner.gmm.sigmas

# TODO for now just learning a new one directly from demos
trajectory_names = ['trajectory_00%d' % idx for idx in range(1, 10)]
ee_trajs = get_task_trajectories("/pandas/get_task_trajectories",
                                 "end_effector_pose_base_frame",
                                 trajectory_names)

w, ros_config = learn_weights(
    "/policy_learner/learn_weights", ee_trajs=ee_trajs[:1])
promp_config = ros_to_python_config(ros_config)
promp = ProMP(config=promp_config)
promp.add_demo(w)
for ee_traj in ee_trajs[1:]:
    w, _ = learn_weights("/policy_learner/learn_weights", ee_trajs=[ee_traj])
    promp.add_demo(w)

# Get the ProMP parameters
pi_js = [1.0]
mu_js = [promp.get_mu_w()]
sigma_js = [promp.get_sigma_w()]

# promp_names = learner.promp_library.get_promp_names()

# #promp_names = ['promp_2']

# for promp_name in promp_names:
#     pi_js.append(1.0 / len(promp_names))
#     promp = learner.promp_library.get_promp(promp_name)
#     mu_js.append(promp.get_mu_w())
#     sigma_w = promp.get_sigma_w()
#     sigma_js.append(sigma_w)

#     print "RANK", np.linalg.matrix_rank(sigma_w)
#     print "COND", np.linalg.cond(sigma_w)
#     print "DIM", sigma_w.shape[0]

# Load the trajectory data to get the end-effector pose in the object frame
ee_poses = None
for filename in os.listdir(data_dir):
    if not filename.endswith(".pkl"):
        continue

    with open(os.path.join(data_dir, filename), 'r') as f:
        df = pickle.load(f)
    prefix = "robot_state_end_effector_pose_object_frame"
    df = df.filter(like=prefix)
    pos_cols = ["{}_position_{}".format(prefix, d) for d in ['x', 'y', 'z']]
    rot_cols = [
        "{}_orientation_{}".format(prefix, d) for d in ['x', 'y', 'z', 'w']
    ]
    cols = pos_cols + rot_cols
    pose = df[cols].values.T[:, -1]
    ee_poses = pose if ee_poses is None else np.vstack(
        (ee_poses, pose))  # (n_instances, 7)

# Learn parameters for ee in obj frame GMM
obj_gmm = DPGMM()
obj_gmm.fit(ee_poses)
beta_rs = obj_gmm.weights_
mu_rs = obj_gmm.means_
sigma_rs = obj_gmm.covariances_

psi, _ = promp.get_block_phi(1.0)

obj_pose = np.array([-0.5, -0.3])

neg_entropy = objective_function(obj_pose, psi.T, alpha_ks, mu_ks, sigma_ks,
                                 pi_js, mu_js, sigma_js, beta_rs, mu_rs,
                                 sigma_rs)

# with open(os.path.join(data_dir, "trajectory_027.pkl"), 'r') as f:
#     df = pickle.load(f)
# prefix = "robot_state_object_pose"
# df = df.filter(like=prefix)
# pos_cols = ["{}_position_{}".format(prefix, d) for d in ['x', 'y', 'z']]
# rot_cols = ["{}_orientation_{}".format(prefix, d) for d in ['x', 'y', 'z', 'w']]
# cols = pos_cols + rot_cols
# pose = df[cols].values.T[:,-1]
# print "POSE", pose
