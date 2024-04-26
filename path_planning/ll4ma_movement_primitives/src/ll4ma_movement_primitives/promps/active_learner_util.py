import os
import errno
import yaml
import rospy
import numpy as np
import tf2_ros as tf
from tf import transformations
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import colors
from std_srvs.srv import Trigger
from ll4ma_movement_primitives.srv import VisualizeRectangle, VisualizeRectangleRequest
from ll4ma_movement_primitives.msg import Array, Mixture


def savefig(learner, fig, idx, fig_type):
    """
    Save pyplot figure according to plot type with the specified index.
    """
    if fig_type == "entropy":
        filename = os.path.join(learner.img_path, "entropy_%03d.png" % idx)
    elif fig_type == "pos_prob":
        filename = os.path.join(learner.img_path, "pos_prob_%03d.png" % idx)
    elif fig_type == "neg_prob":
        filename = os.path.join(learner.img_path, "neg_prob_%03d.png" % idx)
    elif fig_type == "gmm":
        filename = os.path.join(learner.img_path, "gmm_%03d.png" % idx)
    elif fig_type == "scatter":
        filename = os.path.join(learner.img_path, "scatter_%03d.png" % idx)
    else:
        rospy.logwarn("Unknown figure type: %s" % fig_type)
        return None
    rospy.loginfo("Saving image to: %s" % filename)
    plt.savefig(filename, transparent=True)


def render_rviz_img(learner, img_type):
    """
    Projects the most recent instance of the image type in rViz by publishing
    image to a topic and using an external rViz plugin to render the projection.
    """
    try:
        if img_type == "entropy":
            render = rospy.ServiceProxy("/display_entropy_image", Trigger)
        elif img_type == "pos_prob":
            render = rospy.ServiceProxy("/display_pos_prob_image", Trigger)
        elif img_type == "neg_prob":
            render = rospy.ServiceProxy("/display_neg_prob_image", Trigger)
        elif img_type == "gmm":
            render = rospy.ServiceProxy("/display_gmm_image", Trigger)
        elif img_type == "scatter":
            render = rospy.ServiceProxy("/display_scatter_image", Trigger)
        else:
            rospy.logwarn("Unknown image type: %s" % img_type)
            return False
        render()
    except rospy.ServiceException as e:
        rospy.logwarn("Could not render rViz image: %s" % e)


def plot_scatter_data(instances, ax, cname, bounds):
    """
    Visualize scatter plot data points.
    """
    # Flipped to render correctly in rViz projection:
    ax.set_xlim(bounds[1], bounds[0])
    ax.set_ylim(bounds[2], bounds[3])
    converter = colors.ColorConverter()
    color = converter.to_rgba(colors.cnames[cname])
    for instance in instances:
        p = instance.object_planar_pose
        ax.scatter(p[0], p[1], color=color, s=900.0)


def plot_covariance(ax, params, cname):
    """
    Plot covariance of mixture model centered at mean. Either GMM or ProMP library.
    """
    color = colors.cnames[cname]
    means = params['means']
    covs = params['covs']
    for mean, cov in zip(means, covs):
        if not np.any(mean) or not np.any(cov):
            continue
        mean = mean[:2]
        cov = cov[:2, :][:, :2]
        v, w = np.linalg.eigh(cov)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(1.0)
        ax.add_artist(ell)


def plot_gmm_promp(learner, ax, bounds):
    """
    Visualize GMM and ProMP library mixture components.
    """
    ax.set_xlim(bounds[1], bounds[0])
    ax.set_ylim(bounds[2], bounds[3])
    gmm_params = learner.get_gmm_params()
    promp_params = learner.get_promp_traj_params()
    plot_covariance(ax, gmm_params, "firebrick")
    plot_covariance(ax, promp_params, "cornflowerblue")


def visualize_value(learner, idx, value_type):
    """
    Visualize the specified value type in ['entropy', 'pos_prob', 'neg_prob']. 
    Saves figure to file and sends the image to a topic for projection in rViz.
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(learner.config["num_xs"], learner.config["num_ys"])
    fig.patch.set_visible(False)
    ax.axis('off')
    grid = 0.5 * np.ones((learner.config["num_xs"], learner.config["num_ys"]))
    vmin = 0.0
    vmax = 1.0
    for instance in learner.task_candidates:
        point = instance.grid_coords
        if value_type == "entropy":
            vmin = 0.0
            vmax = 0.7
            grid[point[0], point[1]] = instance.entropy
        elif value_type == "pos_prob":
            grid[point[0], point[1]] = instance.pos_prob
        elif value_type == "neg_prob":
            grid[point[0], point[1]] = instance.neg_prob
        else:
            rospy.logwarn("Unknown value type: %s" % value_type)
            return None
    plt.imshow(grid.T, vmin=vmin, vmax=vmax, cmap="hot")
    plt.tight_layout()
    savefig(learner, fig, idx, value_type)
    render_rviz_img(learner, value_type)
    plt.close(fig)


def visualize_gmm(learner, idx):
    """
    Visualize GMM and ProMP mixture components saved to image and projected into rViz.
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(learner.config["num_xs"], learner.config["num_ys"])
    fig.patch.set_visible(False)
    ax.axis('off')

    x_min = learner.config["x_min"]
    x_max = learner.config["x_max"]
    y_min = learner.config["y_min"]
    y_max = learner.config["y_max"]

    plot_gmm_promp(learner, ax, (x_min, x_max, y_min, y_max))

    plt.tight_layout()
    savefig(learner, fig, idx, "gmm")
    render_rviz_img(learner, "gmm")
    plt.close(fig)


def visualize_scatter(learner, idx):
    """
    Visualize scatter plot data (task samples) saved to image and projected into rViz.
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(learner.config["num_xs"], learner.config["num_ys"])
    fig.patch.set_visible(False)
    ax.axis('off')

    pos_samples = [i for i in learner.selected_tasks if i.label == 1]
    neg_samples = [i for i in learner.selected_tasks if i.label == 0]

    x_min = learner.config["x_min"]
    x_max = learner.config["x_max"]
    y_min = learner.config["y_min"]
    y_max = learner.config["y_max"]

    plot_scatter_data(pos_samples, ax, "cornflowerblue",
                      (x_min, x_max, y_min, y_max))
    plot_scatter_data(neg_samples, ax, "firebrick",
                      (x_min, x_max, y_min, y_max))

    plt.tight_layout()
    savefig(learner, fig, idx, "scatter")
    render_rviz_img(learner, "scatter")
    plt.close(fig)


def visualize_region(learner):
    """
    Visualize the region from which samples are being taken. For now just 
    showing rectangular region projected into rViz.
    """
    x_min = learner.config["x_min"]
    x_max = learner.config["x_max"]
    y_min = learner.config["y_min"]
    y_max = learner.config["y_max"]
    req = VisualizeRectangleRequest()
    req.max_x = x_max
    req.min_x = x_min
    req.max_y = y_max
    req.min_y = y_min
    req.pose.position.x = req.max_x - (req.max_x - req.min_x) / 2.0
    req.pose.position.y = req.max_y - (req.max_y - req.min_y) / 2.0
    # Assuming points in x-y plane:
    req.pose.position.z = learner.region_corners[0][2]
    try:
        viz = rospy.ServiceProxy("/visualization/visualize_rect_region",
                                 VisualizeRectangle)
        resp = viz(req)
    except rospy.ServiceException as e:
        rospy.logwarn("Service call to visualize rectangle failed: %s" % e)


def load_config(config_path, config_filename):
    """
    Load configuration file specifying desired region of generalization.
    """
    config_file = os.path.join(config_path, config_filename)
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    return config


def setup_img_path(session_path):
    """
    Create directories for storing imgs over learning iterations.
    """
    img_path = os.path.expanduser(os.path.join(session_path, "imgs"))
    try:
        os.makedirs(img_path)
        with open(os.path.join(img_path, "img.index"), 'w+') as f:
            f.write('1')
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return img_path


def increment_file_index(img_path, filename):
    """
    Increment the index stored in the file. Assumed the first line has only 
    integer index. Returns the previous index before increment.
    """
    with open(os.path.join(img_path, filename), 'r') as f:
        idx = int(f.readline())
    with open(os.path.join(img_path, filename), 'w+') as f:
        f.write(str(idx + 1))
    return idx


def get_tf(reference_frame, query_frame):
    """
    Lookup TF transformation between two frames. Times out after 10 seconds if 
    lookup cannot succeed.
    """
    trans = rot = None
    tf_buffer = tf.Buffer()
    tf_listener = tf.TransformListener(tf_buffer)
    rospy.loginfo("Waiting for transform between '%s' and '%s'..." %
                  (reference_frame, query_frame))
    try:
        tf_stmp = tf_buffer.lookup_transform(reference_frame, query_frame,
                                             rospy.Time.now(),
                                             rospy.Duration(10.0))
        rospy.loginfo("Transform between '{}' and '{}' found!".format(
            reference_frame, query_frame))
    except (tf.LookupException, tf.ConnectivityException,
            tf.ExtrapolationException):
        rospy.logwarn("Could not get transform from '%s' to '%s'." %
                      (reference_frame, query_frame))
        return None

    t = tf_stmp.transform.translation
    r = tf_stmp.transform.rotation
    trans = [t.x, t.y, t.z]
    rot = [r.x, r.y, r.z, r.w]
    frame = transformations.quaternion_matrix(rot)
    frame[0, 3] = trans[0]
    frame[1, 3] = trans[1]
    frame[2, 3] = trans[2]
    return frame


def tf_to_pose(reference_T_query):
    """
    Convert homogeneous transformation matrix (4x4) to its corresponding 
    position and quaternion.
    """
    pos = reference_T_query[:3, 3]
    quat = transformations.quaternion_from_matrix(reference_T_query)
    return pos, quat


def tf_from_pose(pose):
    """
    Converting from ROS Pose
    """
    tf = quaternion_to_tf([
        pose.orientation.x, pose.orientation.y, pose.orientation.z,
        pose.orientation.w
    ])
    tf[0, 3] = pose.position.x
    tf[1, 3] = pose.position.y
    tf[2, 3] = pose.position.z
    return tf


def euler_yaw_to_tf(yaw):
    tf = np.zeros((4, 4))
    tf[0, 0] = np.cos(yaw)
    tf[0, 1] = -np.sin(yaw)
    tf[1, 0] = np.sin(yaw)
    tf[1, 1] = np.cos(yaw)
    tf[2, 2] = 1
    tf[3, 3] = 1
    return tf


def quaternion_to_tf(q):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q0 = q[3]
    tf = np.zeros((4, 4))
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


def table_planar_to_base_tf(obj_x, obj_y, obj_z, obj_theta, table_position,
                            table_quaternion):
    base_TF_table = quaternion_to_tf(table_quaternion)
    base_TF_table[0, 3] = table_position[0]
    base_TF_table[1, 3] = table_position[1]
    base_TF_table[2, 3] = table_position[2]
    table_TF_object = table_planar_to_table_tf(obj_x, obj_y, obj_z, obj_theta)
    base_TF_object = np.dot(base_TF_table, table_TF_object)
    return base_TF_object


def table_planar_to_table_tf(obj_x, obj_y, obj_z, obj_theta):
    tf = euler_yaw_to_tf(obj_theta)
    tf[0, 3] = obj_x
    tf[1, 3] = obj_y
    tf[2, 3] = obj_z
    return tf


def tf_to_quaternion(tf):
    r11 = tf[0, 0]
    r12 = tf[0, 1]
    r13 = tf[0, 2]
    r21 = tf[1, 0]
    r22 = tf[1, 1]
    r23 = tf[1, 2]
    r31 = tf[2, 0]
    r32 = tf[2, 1]
    r33 = tf[2, 2]

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

    choose_q0 = q0q0 >= q1q1 and q0q0 >= q2q2 and q0q0 >= q3q3
    choose_q1 = q1q1 >= q0q0 and q1q1 >= q2q2 and q1q1 >= q3q3
    choose_q2 = q2q2 >= q0q0 and q2q2 >= q1q1 and q2q2 >= q3q3
    choose_q3 = q3q3 >= q0q0 and q3q3 >= q1q1 and q3q3 >= q2q2

    if choose_q0:
        q0 = np.sqrt(q0q0)
        q1 = q0q1 / q0
        q2 = q0q2 / q0
        q3 = q0q3 / q0
    elif choose_q1:
        q1 = np.sqrt(q1q1)
        q0 = q0q1 / q1
        q2 = q1q2 / q1
        q3 = q1q3 / q1
    elif choose_q2:
        q2 = np.sqrt(q2q2)
        q0 = q0q2 / q2
        q1 = q1q2 / q2
        q3 = q2q3 / q2
    elif choose_q3:
        q3 = np.sqrt(q3q3)
        q0 = q0q3 / q3
        q1 = q1q3 / q3
        q2 = q2q3 / q3
    else:
        rospy.logerr("OOPS")
        return None

    return q1, q2, q3, q0


def tf_to_position(tf):
    return tf[:3, 3]


def TF_mu(base_TF_object, mu):
    base_pose_ee = np.zeros(7)
    object_q_ee = mu[3:]
    object_q_ee /= np.linalg.norm(object_q_ee)
    object_TF_ee = quaternion_to_tf(object_q_ee)
    object_TF_ee[0, 3] = mu[0]
    object_TF_ee[1, 3] = mu[1]
    object_TF_ee[2, 3] = mu[2]
    base_TF_ee = np.dot(base_TF_object, object_TF_ee)
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
    M = np.zeros((7, 7))
    base_q_object = tf_to_quaternion(base_TF_object)
    T = quaternion_to_tf(base_q_object)
    M[:3, :3] = T[:3, :3]
    M[3:, 3:] = quaternion_rotation_matrix(base_q_object)
    return np.dot(M, np.dot(sigma, M.T))


def serialize_array(array):
    a = Array()
    if array.ndim == 1:
        a.num_rows = len(array)
        a.num_cols = 1
    else:
        a.num_rows, a.num_cols = array.shape
    a.data = array.flatten()
    return a


def deserialize_array(array):
    return np.array(array.data).reshape((array.num_rows, array.num_cols))


def serialize_mixture(mixture):
    m = Mixture()
    if 'weights' in mixture.keys():
        m.weights = mixture['weights']
        for mean in mixture['means']:
            m.means.append(serialize_array(mean))
        for cov in mixture['covs']:
            m.covariances.append(serialize_array(cov))
    return m


def deserialize_mixture(mixture):
    result = {}
    if mixture.weights:
        result['weights'] = mixture.weights
        result['means'] = [deserialize_array(a) for a in mixture.means]
        result['covs'] = [deserialize_array(a) for a in mixture.covariances]
    return result
