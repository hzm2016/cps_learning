#!/usr/bin/env python
import os
import sys
import numpy as np
import rospy
import cPickle as pickle
from ll4ma_movement_primitives.promps import (
    ProMP, ros_to_python_config, python_to_ros_config, python_to_ros_waypoint)
from rospy_service_helper import (
    learn_promps, learn_weights, get_task_trajectories, visualize_promps,
    generate_joint_trajectory, visualize_joint_trajectory, get_smooth_traj,
    visualize_poses)
from ll4ma_trajectory_msgs.msg import TaskTrajectory, TaskTrajectoryPoint
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rospy_service_helper import SRV_NAMES
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, PoseStamped
from ll4ma_movement_primitives.promps import Waypoint, TaskInstance
from ll4ma_movement_primitives.promps import active_learner_util as al_util
from ll4ma_policy_learning.msg import PlanarPose


def parse_args(args):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a',
        '--active_learner',
        dest='active_learner',
        type=int,
        required=True)
    parser.add_argument(
        '-s',
        '--sampling_method',
        dest='sampling_method',
        type=str,
        required=True)
    parser.add_argument(
        '-e',
        '--experiment',
        dest='experiment',
        type=str,
        default="experiment_1")
    parser.add_argument('-t', '--trial', dest='trial', type=int, default=0)
    parser.add_argument(
        '-p',
        '--path',
        dest='path',
        type=str,
        default="/media/adam/data_haro/rss_2019")
    parsed_args = parser.parse_args(sys.argv[1:])
    return parsed_args


if __name__ == '__main__':

    args = parse_args(sys.argv[1:])

    learner_path = os.path.join(args.path, args.experiment,
                                args.sampling_method, "trial_{}".format(
                                    args.trial), "learners")
    learner_filename = os.path.join(
        learner_path, "active_learner_%03d.pkl" % args.active_learner)
    with open(learner_filename, 'r') as f:
        learner = pickle.load(f)

    num_promps = learner.promp_library.get_num_promps()
    # overlaps = -1 * np.ones((num_promps, num_promps))

    for i in range(num_promps):
        for j in range(i + 1, num_promps):
            in_common = 0
            names1 = learner._metadata.promp_data_names["promp_{}".format(i)]
            names2 = learner._metadata.promp_data_names["promp_{}".format(j)]
            for name in names1:
                if name in names2:
                    in_common += 1
            # Compute overlap w.r.t. smaller one
            overlap = float(in_common) / min(len(names1), len(names2))
            if overlap > 0.0:
                print "promp_{} -- promp_{}: {}".format(i, j, overlap)

    print learner._metadata

    # np.set_printoptions(linewidth=300)
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    # print overlaps