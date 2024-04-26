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

srvs = {
    "learn_weights": SRV_NAMES["learn_weights"],
    "viz_promps": SRV_NAMES["viz_promps"],
    "gen_traj": SRV_NAMES["gen_traj"],
    "get_smooth_traj": SRV_NAMES["get_smooth_traj"],
    "viz_traj": SRV_NAMES["viz_traj"],
    "viz_poses": SRV_NAMES["viz_poses"]
}


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
        required=True,
        choices=[
            "random", "mahal_cond", "mahal_no", "entropy", "kl_cond_ee",
            "kl_ee_cond", "min_prob_no", "min_prob_cond"
        ])
    parser.add_argument(
        '-e',
        '--experiment',
        dest='experiment',
        type=str,
        default="experiment_1")
    parser.add_argument('-t', '--trial', dest='trial', type=int, default=0)
    parser.add_argument(
        '--show_robot', dest='show_robot', action="store_true", default=False)
    parser.add_argument(
        '-p',
        '--path',
        dest='path',
        type=str,
        default="/media/adam/data_haro/rss_2019")
    parsed_args = parser.parse_args(sys.argv[1:])
    return parsed_args


if __name__ == '__main__':
    rospy.init_node("validate_experiments")

    args = parse_args(sys.argv[1:])

    rospy.loginfo("Waiting for services...")
    for key in srvs.keys():
        rospy.loginfo("{:4}{}".format("", srvs[key]))
        rospy.wait_for_service(srvs[key])
    rospy.loginfo("Services are up!")

    learner_path = os.path.join(args.path, args.experiment,
                                args.sampling_method, "trial_{}".format(
                                    args.trial), "learners")
    learner_filename = os.path.join(
        learner_path, "active_learner_%03d.pkl" % args.active_learner)
    with open(learner_filename, 'r') as f:
        learner = pickle.load(f)

    x_min = learner.config["x_min"]
    x_max = learner.config["x_max"]
    y_min = learner.config["y_min"]
    y_max = learner.config["y_max"]
    theta_min = learner.config["theta_min"]
    theta_max = learner.config["theta_max"]
    z = learner.config["z"]
    learner.finite_goal_set = []
    xs = np.linspace(x_min, x_max, learner.config["num_xs"])
    ys = np.linspace(y_min, y_max, learner.config["num_ys"])
    thetas = np.linspace(theta_min, theta_max, learner.config["num_thetas"])

    # np.random.seed(args.trial)
    np.random.seed(100)

    task_instances = []
    for _ in range(100):
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        theta = np.random.uniform(theta_min, theta_max)
        task_instance = TaskInstance()
        task_instance.object_planar_pose = PlanarPose(
            x=x, y=y, z=z, theta=theta)
        task_instances.append(task_instance)

    for task_idx, task_instance in enumerate(task_instances):
        replay = True
        while replay:
            rospy.loginfo("Showing instance {}".format(task_idx + 1))
            if rospy.is_shutdown():
                sys.exit(1)
            planar = task_instance.object_planar_pose
            base_TF_object = al_util.table_planar_to_base_tf(
                planar.x, planar.y, planar.z, planar.theta,
                learner.table_position, learner.table_quaternion)
            pos, quat = al_util.tf_to_pose(base_TF_object)
            task_instance.object_pose = Pose()
            task_instance.object_pose.position.x = pos[0]
            task_instance.object_pose.position.y = pos[1]
            task_instance.object_pose.position.z = pos[2]
            task_instance.object_pose.orientation.x = quat[0]
            task_instance.object_pose.orientation.y = quat[1]
            task_instance.object_pose.orientation.z = quat[2]
            task_instance.object_pose.orientation.w = quat[3]

            ee_gmm_params = learner.get_ee_gmm_params()
            query_waypoints = []
            for object_mu_ee, object_sigma_ee in zip(ee_gmm_params['means'],
                                                     ee_gmm_params['covs']):
                base_mu_ee = al_util.TF_mu(base_TF_object, object_mu_ee)
                base_sigma_ee = al_util.TF_sigma(base_TF_object,
                                                 object_sigma_ee)
                waypoint = Waypoint()
                waypoint.phase_val = 1.0
                waypoint.condition_keys = ["x.{}".format(i) for i in range(7)]
                waypoint.values = base_mu_ee
                waypoint.sigma = base_sigma_ee
                query_waypoints.append(waypoint)

            promp_key, waypoint = learner.get_most_likely_promp(
                query_waypoints)
            if waypoint is None:
                rospy.logwarn("Could not find valid ProMP/Waypoint")
                continue
            promp = learner.get_promp(promp_key)
            pose = Pose()
            pose.position.x = waypoint.values[0]
            pose.position.y = waypoint.values[1]
            pose.position.z = waypoint.values[2]
            pose.orientation.x = waypoint.values[3]
            pose.orientation.y = waypoint.values[4]
            pose.orientation.z = waypoint.values[5]
            pose.orientation.w = waypoint.values[6]
            visualize_poses(srvs["viz_poses"], [pose])
            waypoints = [waypoint]
            promp_config = promp.get_config()

            # Visualize generated samples as the end-effector trace for each sample
            ros_config = python_to_ros_config(promp_config)
            ros_waypoints = [
                python_to_ros_waypoint(waypoint) for waypoint in waypoints
            ]
            visualize_promps(
                srvs["viz_promps"],
                ros_config,
                waypoints=ros_waypoints,
                num_samples=30,
                clear=True)

            # Visualize the object mesh
            mesh_marker = Marker()
            mesh_marker.id = 0
            mesh_marker.header.frame_id = "lbr4_base_link"
            mesh_marker.type = Marker.MESH_RESOURCE
            mesh_marker.action = Marker.ADD
            mesh_marker.color.r = 0
            mesh_marker.color.g = 1
            mesh_marker.color.b = 0
            mesh_marker.color.a = 1
            mesh_marker.scale.x = 1
            mesh_marker.scale.y = 1
            mesh_marker.scale.z = 1
            mesh_marker.pose = task_instance.object_pose
            mesh_marker.mesh_resource = "package://ll4ma_robots_description/meshes/environment/drill.obj"
            mesh_pub = rospy.Publisher(
                "/test_object_mesh", Marker, queue_size=1)
            rate = rospy.Rate(100)
            pose_pub = rospy.Publisher(
                "/test_end_effector_pose", PoseStamped, queue_size=1)
            for _ in range(300):
                mesh_marker.header.stamp = rospy.Time.now()
                mesh_pub.publish(mesh_marker)
                rate.sleep()

            reflex_marker = Marker()
            reflex_marker.id = 1
            reflex_marker.header.frame_id = "lbr4_base_link"
            reflex_marker.type = Marker.MESH_RESOURCE
            reflex_marker.action = Marker.ADD
            reflex_marker.color.r = 0
            reflex_marker.color.g = 0
            reflex_marker.color.b = 1
            reflex_marker.color.a = 1
            reflex_marker.scale.x = 1
            reflex_marker.scale.y = 1
            reflex_marker.scale.z = 1
            reflex_marker.mesh_resource = "package://ll4ma_robots_description/meshes/reflex/pad.STL"

            # Generate trajectory to show for rViz (and store for execution)
            traj = promp.generate_trajectory(
                dt=0.01, duration=10.0, waypoints=waypoints)
            task_trajectory = TaskTrajectory()
            for j in range(len(traj['x'][0])):
                t_point = TaskTrajectoryPoint()
                t_point.pose.position.x = traj['x'][0][j]
                t_point.pose.position.y = traj['x'][1][j]
                t_point.pose.position.z = traj['x'][2][j]
                # Need to normalize quaternion
                q = np.array([
                    traj['x'][3][j], traj['x'][4][j], traj['x'][5][j],
                    traj['x'][6][j]
                ])
                q /= np.linalg.norm(q)
                t_point.pose.orientation.x = q[0]
                t_point.pose.orientation.y = q[1]
                t_point.pose.orientation.z = q[2]
                t_point.pose.orientation.w = q[3]
                task_trajectory.points.append(t_point)

            if args.show_robot:
                # Generate the corresponding joint trajectory
                joint_trajectory = generate_joint_trajectory(
                    srvs["gen_traj"], promp.init_joint_state, task_trajectory)
                # Subsample the points (otherwise smoothing takes too long)
                joint_trajectory.points = joint_trajectory.points[::5]
                # Get a smooth trajectory
                joint_trajectory = get_smooth_traj(srvs["get_smooth_traj"],
                                                   joint_trajectory)
                if joint_trajectory is not None:
                    success = visualize_joint_trajectory(
                        srvs["viz_traj"], joint_trajectory)

            reflex_pub = rospy.Publisher(
                "/test_reflex_mesh", Marker, queue_size=1)

            for t_point in task_trajectory.points:
                pose_stmp = PoseStamped()
                pose_stmp.pose = t_point.pose
                pose_stmp.header.frame_id = "lbr4_base_link"
                pose_stmp.header.stamp = rospy.Time.now()
                pose_pub.publish(pose_stmp)
                reflex_marker.pose = t_point.pose
                reflex_pub.publish(reflex_marker)
                rate.sleep()
            rospy.loginfo("Visualization active.")

            action = raw_input("\n  Action (r)eplay, (c)ontinue: ")
            replay = (action == 'r')