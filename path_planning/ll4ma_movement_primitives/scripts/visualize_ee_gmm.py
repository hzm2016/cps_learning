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
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from ll4ma_movement_primitives.promps import active_learner_util as al_util


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

    parsed_args = parser.parse_args(sys.argv[1:])
    return parsed_args


if __name__ == '__main__':
    rospy.init_node("visualize_promp_session")

    args = parse_args(sys.argv[1:])

    rel_path = "experiment_1__grid_data"

    learner_path = os.path.join(args.abs_path, rel_path, "learners")
    # learner_filename = os.path.join(
    #     learner_path, "active_learner_%03d.pkl" % args.active_learner)
    learner_filename = os.path.join(learner_path, "final_active_learner.pkl")
    with open(learner_filename, 'r') as f:
        learner = pickle.load(f)

    task_instance = learner.selected_tasks[-1]

    base_TF_object = al_util.tf_from_pose(task_instance.object_pose)
    mean_ee_pose_array = PoseArray()
    mean_ee_pose_array.header.frame_id = "lbr4_base_link"

    for object_mu_ee in learner.ee_gmm.get_means():
        base_mu_ee = al_util.TF_mu(base_TF_object, object_mu_ee)
        pose = Pose()
        pose.position.x = base_mu_ee[0]
        pose.position.y = base_mu_ee[1]
        pose.position.z = base_mu_ee[2]
        pose.orientation.x = base_mu_ee[3]
        pose.orientation.y = base_mu_ee[4]
        pose.orientation.z = base_mu_ee[5]
        pose.orientation.w = base_mu_ee[6]
        mean_ee_pose_array.poses.append(pose)

    ee_pose_array = learner.get_ee_gmm_instances_in_base(
        task_instance.object_pose)
    ee_pose_array.header.frame_id = "lbr4_base_link"

    mesh_marker = Marker()
    mesh_marker.id = 0
    mesh_marker.header.frame_id = "lbr4_base_link"
    mesh_marker.type = Marker.MESH_RESOURCE
    mesh_marker.action = Marker.ADD
    mesh_marker.color.r = 248 / 256.0
    mesh_marker.color.g = 148 / 256.0
    mesh_marker.color.b = 6 / 256.0
    mesh_marker.color.a = 0.9
    mesh_marker.scale.x = 1
    mesh_marker.scale.y = 1
    mesh_marker.scale.z = 1
    mesh_marker.pose = task_instance.object_pose
    mesh_marker.mesh_resource = "package://ll4ma_robots_description/meshes/environment/drill.obj"

    mesh_pub = rospy.Publisher("/test_object_mesh", Marker, queue_size=1)
    mean_ee_pose_pub = rospy.Publisher(
        "/ee_gmm_means", PoseArray, queue_size=1)
    ee_pose_pub = rospy.Publisher("/ee_gmm_instances", PoseArray, queue_size=1)

    rate = rospy.Rate(100)

    rospy.loginfo("Visualization active.")
    while not rospy.is_shutdown():
        mesh_marker.header.stamp = rospy.Time.now()
        mean_ee_pose_array.header.stamp = rospy.Time.now()
        ee_pose_array.header.stamp = rospy.Time.now()
        mesh_pub.publish(mesh_marker)
        mean_ee_pose_pub.publish(mean_ee_pose_array)
        ee_pose_pub.publish(ee_pose_array)
        rate.sleep()
