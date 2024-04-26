import rospy
import os
import casadi as ca
import numpy as np
import rospkg
import yaml
import cPickle as pickle
from geometry_msgs.msg import PoseStamped
from casadi import SX, mtimes, Function, nlpsol
from scipy.linalg import block_diag
from sklearn.mixture import BayesianGaussianMixture as DPGMM
from tf import transformations as tf
from ll4ma_movement_primitives.promps import ProMP, ros_to_python_config, Optimizer
from ll4ma_movement_primitives.util import casadi_util as ca_util
from rospy_service_helper import get_task_trajectories, learn_promps, learn_weights

_OBJ_POSE_DOF = 2

rospy.init_node('test_optimization')

# Read in the active learner
data_dir = "/media/adam/data_haro/test_optimization/demos"
# data_dir = "/home/adam/.rosbags/demos"
learner_filename = "backup/active_learner_030.pkl"
with open(os.path.join(data_dir, learner_filename), 'r') as f:
    learner = pickle.load(f)

# Get the object GMM parameters
gmm_obj = {}
gmm_obj['weights'] = learner.gmm.weights
gmm_obj['means'] = learner.gmm.mus
gmm_obj['covs'] = learner.gmm.sigmas

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
promps = {}
promps['pi'] = [1.0]
promps['mu'] = [promp.get_mu_w()]
promps['sigma'] = [promp.get_sigma_w()]

# # Get the ProMP parameters
# pi_js = []
# mu_js = []
# sigma_js = []
# promp_names = learner.promp_library.get_promp_names(
# )[:1]  # TODO just trying one for now
# for promp_name in promp_names:
#     pi_js.append(1.0 / len(promp_names))
#     promp = learner.promp_library.get_promp(promp_name)
#     mu_js.append(promp.get_mu_w())
#     sigma_js.append(promp.get_sigma_w())

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
gmm_ee = {}
gmm_ee['beta'] = obj_gmm.weights_
gmm_ee['mu'] = obj_gmm.means_
gmm_ee['sigma'] = obj_gmm.covariances_

psi, _ = promp.get_block_phi(1.0)
table_pos = [-0.77, 0, 0]
table_quat = [0, 0, -1, 0]

obj_pose = SX.sym('obj_pose', _OBJ_POSE_DOF)

optimizer = Optimizer()

neg_entropy = optimizer.max_entropy_objective(obj_pose, table_pos, table_quat,
                                              psi.T, gmm_obj, promps, gmm_ee)

test = Function('f', [obj_pose[0], obj_pose[1]], [neg_entropy],
                ['obj_x', 'obj_y'], ['neg_entropy'])

lbx = [-0.35, -0.5]
ubx = [0.35, 0.5]
x_guess = [-0.35, 0.5]

problem = {'x': obj_pose, 'f': neg_entropy}
solver = nlpsol("nlp", "ipopt", problem)
soln = solver(x0=x_guess, lbx=lbx, ubx=ubx)['x'].full().flatten()
val = test(obj_x=soln[0], obj_y=soln[1])['neg_entropy'].full().flatten()[0]
print "\n\n    WINNER: {}: {}\n".format(soln, val)

# ==============================================================================


def test_tf():
    x = 0.25
    y = 0.25
    z = 0.25
    theta = 0.0

    # Get transform table to base
    rospack = rospkg.RosPack()
    path = rospack.get_path("robot_aruco_calibration")
    filename = os.path.join(path, "config/robot_camera_calibration.yaml")
    with open(filename, 'r') as f:
        poses = yaml.load(f)

    parent_link = "lbr4_base_link"
    child_link = "table_center"
    name = "{}__to__{}".format(parent_link, child_link)
    table_pos = []
    table_quat = []
    table_pos.append(poses[name]["position"]["x"])
    table_pos.append(poses[name]["position"]["y"])
    table_pos.append(poses[name]["position"]["z"])
    table_quat.append(poses[name]["orientation"]["x"])
    table_quat.append(poses[name]["orientation"]["y"])
    table_quat.append(poses[name]["orientation"]["z"])
    table_quat.append(poses[name]["orientation"]["w"])

    obj_x = SX.sym('obj_x')
    obj_y = SX.sym('obj_y')
    obj_z = SX.sym('obj_z')
    obj_theta = SX.sym('obj_theta')
    table_position = SX.sym('table_p', 3)
    table_quaternion = SX.sym('table_q', 4)
    base_TF_object = ca_util.table_planar_to_base_tf(
        (obj_x, obj_y, obj_z), obj_theta, table_position, table_quaternion)
    q = ca_util.tf_to_quaternion(base_TF_object)
    p = base_TF_object[:3, 3]
    F = Function(
        'F',
        [obj_x, obj_y, obj_z, obj_theta, table_position, table_quaternion],
        [p, q], ['obj_x', 'obj_y', 'obj_z', 'obj_theta', 'table_p', 'table_q'],
        ['p', 'q'])
    p, q = F(x, y, z, theta, table_pos, table_quat)

    pose_stmp = PoseStamped()
    pose_stmp.header.frame_id = "lbr4_base_link"
    pose_stmp.pose.position.x = p[0]
    pose_stmp.pose.position.y = p[1]
    pose_stmp.pose.position.z = p[2]
    pose_stmp.pose.orientation.x = q[1]
    pose_stmp.pose.orientation.y = q[2]
    pose_stmp.pose.orientation.z = q[3]
    pose_stmp.pose.orientation.w = q[0]

    pub = rospy.Publisher("/test_obj_pose", PoseStamped, queue_size=1)
    rate = rospy.Rate(100)
    print "PUBLISHING POSE"
    while not rospy.is_shutdown():
        pose_stmp.header.stamp = rospy.Time.now()
        pub.publish(pose_stmp)
        rate.sleep()


# test_tf()
