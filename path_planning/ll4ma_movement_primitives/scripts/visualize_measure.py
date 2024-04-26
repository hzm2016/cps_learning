#!/usr/bin/env python
import os
import sys
import numpy as np
import rospy
import cPickle as pickle
import matplotlib.pyplot as plt
from copy import copy
from ll4ma_movement_primitives.promps import (
    ProMP, ros_to_python_config, python_to_ros_config, python_to_ros_waypoint,
    TaskInstance)
from rospy_service_helper import (
    learn_promps, learn_weights, get_task_trajectories, visualize_promps,
    generate_joint_trajectory, visualize_joint_trajectory, get_smooth_traj,
    visualize_poses)
from ll4ma_trajectory_msgs.msg import TaskTrajectory, TaskTrajectoryPoint
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rospy_service_helper import SRV_NAMES
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from ll4ma_movement_primitives.promps import active_learner_util as al_util
from ll4ma_policy_learning.msg import PlanarPose
from ll4ma_movement_primitives.promps import Optimizer


def parse_args(args):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', dest='active_learner', type=int)
    parser.add_argument(
        '--abs_path',
        dest='abs_path',
        type=str,
        default="/media/adam/data_haro/rss_2019")
    # default="/media/adam/URLG_HDD/adam")

    parser.add_argument('-s', dest="sampling_type", type=str, default="random")
    parsed_args = parser.parse_args(sys.argv[1:])
    return parsed_args


if __name__ == '__main__':
    # rospy.init_node("visualize_measure")

    args = parse_args(sys.argv[1:])

    rel_path = "experiment_2__{}__trial_1".format(args.sampling_type)

    print "\nSAMPLING TYPE:", args.sampling_type

    # learner_path = os.path.join(args.abs_path, rel_path, "backup")
    learner_path = os.path.join(args.abs_path, rel_path, "learner_copies")

    def make_dir(new_dir):
        import errno
        try:
            os.makedirs(new_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    viz_path = os.path.join(args.abs_path, rel_path, "entropy_viz")
    make_dir(viz_path)

    if args.active_learner:
        learner_filename = os.path.join(
            learner_path, "active_learner_%03d__.pkl" % args.active_learner)

        with open(learner_filename, 'r') as f:
            learner = pickle.load(f)

        table_depth = 10
        table_width = 15
        num_orient = 16

        fig, axes = plt.subplots(4, 4)
        fig.set_size_inches(14, 6)
        # fig.patch.set_visible(False)
        # ax.axis('off')
        entropy_grids = [
            -1.0 * np.ones((table_depth, table_width))
            for _ in range(num_orient)
        ]
        value_grids = [
            -1.0 * np.ones((table_depth, table_width))
            for _ in range(num_orient)
        ]
        pos_grids = [
            -1.0 * np.ones((table_depth, table_width))
            for _ in range(num_orient)
        ]
        neg_grids = [
            -1.0 * np.ones((table_depth, table_width))
            for _ in range(num_orient)
        ]

        x_min = learner.config["x_min"]
        x_max = learner.config["x_max"]
        y_min = learner.config["y_min"]
        y_max = learner.config["y_max"]
        theta_min = learner.config["theta_min"]
        theta_max = learner.config["theta_max"]
        z = learner.config["z"]

        xs = np.linspace(x_min, x_max, learner.config["num_xs"])
        ys = np.linspace(y_min, y_max, learner.config["num_ys"])
        thetas = np.linspace(theta_min, theta_max, num_orient)

        print "NUM PROMPS", learner.promp_library.get_num_promps()

        obj_gmm = learner.obj_gmm.get_params()
        ee_gmm = learner.ee_gmm.get_params()
        # Normalize quaternion
        for i in range(len(ee_gmm['means'])):
            ee_gmm['means'][i][3:] /= np.linalg.norm(ee_gmm['means'][i][3:])
        promps = learner.promp_library.get_params()
        phi = learner.promp_library.get_phi().T

        table_pos = [
            learner.table_position[0], learner.table_position[1],
            learner.table_position[2]
        ]
        table_quat = [
            learner.table_quaternion[0], learner.table_quaternion[1],
            learner.table_quaternion[2], learner.table_quaternion[3]
        ]
        lbx = [
            learner.config["x_min"], learner.config["y_min"],
            learner.config["z"], learner.config["theta_min"]
        ]
        ubx = [
            learner.config["x_max"], learner.config["y_max"],
            learner.config["z"], learner.config["theta_max"]
        ]

        optimizer = Optimizer(table_pos, table_quat, lbx, ubx)

        max_value = 0.0
        for k, theta in enumerate(thetas):
            # if k != 0:
            #     continue
            print "Orientation = {}, Theta = {}".format(k, theta)
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    object_planar_pose = PlanarPose(x=x, y=y, z=z, theta=theta)

                    entropy, probs = learner._evaluate_entropy(
                        object_planar_pose)

                    value = learner._evaluate_mahalanobis(object_planar_pose)

                    final_value = probs[0] * value

                    # object_planar_pose = [x, y, z, theta]
                    # mahal = optimizer.evaluate_mahalanobis(
                    #     object_planar_pose, phi, promps, ee_gmm)

                    if value > max_value:
                        max_value = value
                    entropy_grids[k][i, j] = value
                    value_grids[k][i, j] = final_value
                    pos_grids[k][i, j] = probs[0]
                    neg_grids[k][i, j] = probs[1]

        fig, axes = plt.subplots(4, 4)
        fig.set_size_inches(14, 6)
        for k in range(num_orient):
            idx = int(k > 3) + int(k > 7) + int(k > 11)
            ax = axes[idx, k % 4]
            ax.imshow(entropy_grids[k], vmin=0.0, vmax=max_value, cmap="hot")
        save_filename = os.path.join(viz_path, "entropy.png")
        plt.tight_layout()
        plt.savefig(save_filename)
        plt.close(fig)

        fig, axes = plt.subplots(4, 4)
        fig.set_size_inches(14, 6)
        for k in range(num_orient):
            idx = int(k > 3) + int(k > 7) + int(k > 11)
            ax = axes[idx, k % 4]
            ax.imshow(pos_grids[k], vmin=0.0, vmax=1.0, cmap="hot")
        save_filename = os.path.join(viz_path, "pos.png")
        plt.tight_layout()
        plt.savefig(save_filename)
        plt.close(fig)

        fig, axes = plt.subplots(4, 4)
        fig.set_size_inches(14, 6)
        for k in range(num_orient):
            idx = int(k > 3) + int(k > 7) + int(k > 11)
            ax = axes[idx, k % 4]
            ax.imshow(neg_grids[k], vmin=0.0, vmax=1.0, cmap="hot")
        save_filename = os.path.join(viz_path, "neg.png")
        plt.tight_layout()
        plt.savefig(save_filename)
        plt.close(fig)

        fig, axes = plt.subplots(4, 4)
        fig.set_size_inches(14, 6)
        for k in range(num_orient):
            idx = int(k > 3) + int(k > 7) + int(k > 11)
            ax = axes[idx, k % 4]
            ax.imshow(value_grids[k], vmin=0.0, vmax=max_value, cmap="hot")
        save_filename = os.path.join(viz_path, "final_value.png")
        plt.tight_layout()
        plt.show()
        plt.savefig(save_filename)
        plt.close(fig)

    else:
        for learner_idx, learner_filename in enumerate(
                sorted(os.listdir(learner_path))):
            if not learner_filename.endswith(".pkl"):
                continue
            if learner_filename in ["active_learner_001.pkl"]:
                continue

            print "LEARNER:", learner_filename

            learner_filename = os.path.join(learner_path, learner_filename)

            with open(learner_filename, 'r') as f:
                learner = pickle.load(f)

            table_depth = 10
            table_width = 15
            num_orient = 16

            fig, axes = plt.subplots(4, 4)
            fig.set_size_inches(14, 6)
            # fig.patch.set_visible(False)
            # ax.axis('off')
            grids = [
                100.0 * np.ones((table_depth, table_width))
                for _ in range(num_orient)
            ]

            x_min = learner.config["x_min"]
            x_max = learner.config["x_max"]
            y_min = learner.config["y_min"]
            y_max = learner.config["y_max"]
            theta_min = learner.config["theta_min"]
            theta_max = learner.config["theta_max"]
            z = learner.config["z"]

            xs = np.linspace(x_min, x_max, learner.config["num_xs"])
            ys = np.linspace(y_min, y_max, learner.config["num_ys"])
            thetas = np.linspace(theta_min, theta_max, num_orient)

            max_value = 0.0
            for k, theta in enumerate(thetas):
                print "Orientation = {}".format(k)
                for i, x in enumerate(xs):
                    for j, y in enumerate(ys):
                        object_planar_pose = PlanarPose(
                            x=x, y=y, z=z, theta=theta)
                        entropy = learner._evaluate_entropy(object_planar_pose)
                        if entropy > max_value:
                            max_value = copy(entropy)
                        grids[k][i, j] = entropy
            for j in range(num_orient):
                idx = int(j > 3) + int(j > 7) + int(j > 11)
                ax = axes[idx, j % 4]
                ax.imshow(grids[j], vmin=0.0, vmax=max_value, cmap="hot")
            print "MAX VALUE", max_value
            # ax.set_title("Theta = {}".format(instance.object_planar_pose.theta))
            plt.tight_layout()
            save_filename = os.path.join(viz_path,
                                         "entropy_%03d.png" % learner_idx)
            plt.savefig(save_filename)
            plt.close(fig)
