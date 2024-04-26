#!/usr/bin/env python
import os
import errno
import sys
import rospy
import rosbag
import pandas as pd
import numpy as np
from tqdm import tqdm
from ll4ma_trajectory_msgs.msg import TaskTrajectory, TaskTrajectoryPoint
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from ll4ma_logger.srv import FileAction, FileActionResponse
from ll4ma_trajectory_util.srv import (
    GetJointTrajectories, GetJointTrajectoriesResponse, GetTaskTrajectories,
    GetTaskTrajectoriesResponse, GetPose, GetPoseResponse)

# TODO this is a legacy class from the ProMP active learning experiments.
# This should be replaced by a more general class that handles rosbag to
# pandas conversion and a node that offers services for managing the files.


class PandasInterface:
    def __init__(self):
        abs_path = os.path.expanduser(rospy.get_param("~data_abs_path"))
        rel_path = rospy.get_param("~data_rel_path")
        self.path = os.path.join(abs_path, rel_path)
        self._make_dir(self.path)
        rospy.loginfo("Loading pickled dataframes...")
        rospy.loginfo("Path: %s" % self.path)
        self.dfs = {}
        num_files = 0

        # If a bag file doesn't have a pickle, create it
        for fn in sorted(os.listdir(self.path)):
            if fn.endswith(".bag"):
                bag_fn = os.path.join(self.path, fn)
                pkl_fn = os.path.join(self.path, fn.split('.')[0] + ".pkl")
                if not os.path.isfile(pkl_fn):
                    rospy.loginfo("Creating pickle for '%s'" % fn)
                    self._rosbag_to_pickle(bag_fn, pkl_fn)

        # # Process pickle files
        # rospy.loginfo("Loading pickled dataframes...\n")
        # for fn in tqdm(sorted(os.listdir(self.path))):
        #     if not fn.endswith(".pkl"):
        #         continue
        #     abs_fn = os.path.join(self.path, fn)
        #     traj_name = fn.split(".")[0]
        #     self.dfs[traj_name] = pd.read_pickle(abs_fn)
        #     num_files += 1

        rospy.loginfo("Data loading complete. Loaded %d files." % num_files)
        self.rate = rospy.Rate(100)

        # Services being offered
        self.get_jnt_traj_srv = rospy.Service("/pandas/get_joint_trajectories",
                                              GetJointTrajectories,
                                              self.get_joint_trajectories)
        self.get_task_traj_srv = rospy.Service("/pandas/get_task_trajectories",
                                               GetTaskTrajectories,
                                               self.get_task_trajectories)
        self.get_pose_srv = rospy.Service("/pandas/get_final_pose", GetPose,
                                          self.get_final_pose)
        self.get_latest_srv = rospy.Service(
            "/pandas/get_latest_trajectory_name", FileAction,
            self.get_latest_trajectory_name)
        self.register_fn_srv = rospy.Service(
            "/pandas/register_filename", FileAction, self.register_filename)

    def run(self):
        rospy.loginfo("Data services are available.")
        while not rospy.is_shutdown():
            self.rate.sleep()
        rospy.loginfo("Exiting.")

    # === BEGIN Service functions being offered ====================================================

    def get_joint_trajectories(self, req):
        resp = GetJointTrajectoriesResponse()
        resp.joint_trajectories = self._get_joint_trajectories(
            req.entity_name, req.trajectory_names, req.subsample)
        resp.success = True
        return resp

    def get_task_trajectories(self, req):
        resp = GetTaskTrajectoriesResponse()
        resp.task_trajectories = self._get_task_trajectories(
            req.entity_name, req.trajectory_names, req.subsample)
        resp.success = True
        return resp

    def get_final_pose(self, req):
        resp = GetPoseResponse()
        resp.pose = self._get_final_pose(req.trajectory_name, req.type)
        resp.success = resp.pose is not None
        return resp

    def get_latest_trajectory_name(self, req):
        resp = FileActionResponse()
        idx_fn = os.path.join(self.path, "rosbags.index")
        with open(idx_fn, 'r') as f:
            idx = int(f.readline())
        idx -= 1
        resp.filename = "trajectory_%03d" % idx
        resp.success = True
        return resp

    def register_filename(self, req):
        resp = FileActionResponse()
        pkl_fn = os.path.join(self.path, req.filename + ".pkl")
        # If only the rosbag exists, then we need to load into a dataframe and pickle it
        if not os.path.isfile(pkl_fn):
            bag_fn = os.path.join(self.path, req.filename + ".bag")
            if not os.path.isfile(bag_fn):
                rospy.logerr(
                    "Bag file cannot be found for pickle conversion: %s" %
                    bag_fn)
                resp.success = False
                return resp
            else:
                self._rosbag_to_pickle(bag_fn, pkl_fn)
        self.dfs[req.filename] = pd.read_pickle(pkl_fn)
        resp.success = True
        return resp

    # === END Service functions being offered ======================================================

    def _get_joint_trajectories(self,
                                entity_name,
                                trajectory_names=[],
                                subsample=0.9):
        if not trajectory_names:
            trajectory_names = self.dfs.keys()
        joint_trajs = []
        for trajectory_name in trajectory_names:
            joint_traj = self._get_joint_trajectory(entity_name,
                                                    trajectory_name, subsample)
            joint_trajs.append(joint_traj)
        return joint_trajs

    def _get_joint_trajectory(self,
                              entity_name,
                              trajectory_name,
                              subsample=0.9):
        # # Retrieve all the data
        # df = self.dfs[trajectory_name]

        abs_fn = os.path.join(self.path, "{}.pkl".format(trajectory_name))
        df = pd.read_pickle(abs_fn)

        joint_positions = self._get_jnt_data(df, entity_name, trajectory_name,
                                             "position", subsample)
        joint_velocities = self._get_jnt_data(df, entity_name, trajectory_name,
                                              "velocity", subsample)

        # Create the joint trajectory
        joint_traj = JointTrajectory()
        # joint_traj.trajectory_name = trajectory_name
        num_pts = joint_positions.shape[1]
        for j in range(num_pts):
            j_point = JointTrajectoryPoint()
            j_point.positions = joint_positions[:, j]
            j_point.velocities = joint_velocities[:, j]
            joint_traj.points.append(j_point)
        rospy.loginfo("Retrieved joint trajectory of size %d for '%s'" %
                      (num_pts, trajectory_name))
        return joint_traj

    def _get_jnt_data(self,
                      df,
                      entity_name,
                      trajectory_name,
                      state_type="",
                      subsample=0.9):
        # Retrieve data corresponding to joint
        data = df.filter(like="%s_joint_state_%s_" % (entity_name, state_type))
        # Drop any rows that don't have any data for this attribute
        data = data.dropna(how="all")
        # Sort the columns
        data = data[sorted(data.columns)]
        # Subsample so it doesn't take all day
        data = data.drop(
            np.linspace(
                1,
                data.shape[0] - 1,
                int(subsample * data.shape[0]),
                dtype=int))
        # Make an array
        data = data.values.T
        return data

    def _get_task_trajectories(self,
                               entity_name,
                               trajectory_names=[],
                               subsample=0.9):
        if not trajectory_names:
            trajectory_names = self.dfs.keys()
        task_trajs = []
        for trajectory_name in trajectory_names:
            task_traj = self._get_task_trajectory(entity_name, trajectory_name,
                                                  subsample)
            task_trajs.append(task_traj)
        return task_trajs

    def _get_task_trajectory(self, entity_name, trajectory_name,
                             subsample=0.9):
        # Retrieve all the data
        abs_fn = os.path.join(self.path, "{}.pkl".format(trajectory_name))
        df = pd.read_pickle(abs_fn)
        # df = self.dfs[trajectory_name]

        # Process any nan values, they were showing up in robot-generated twists
        df = df.fillna(0)
        poses = self._get_pose_data(df, entity_name, trajectory_name,
                                    subsample)
        twists = self._get_twist_data(df, trajectory_name, subsample)

        # Create the task trajectory
        task_traj = TaskTrajectory()
        task_traj.trajectory_name = trajectory_name
        num_pts = poses.shape[1]
        for j in range(num_pts):
            t_point = TaskTrajectoryPoint()
            t_point.pose.position.x = poses[0, j]
            t_point.pose.position.y = poses[1, j]
            t_point.pose.position.z = poses[2, j]
            t_point.pose.orientation.x = poses[3, j]
            t_point.pose.orientation.y = poses[4, j]
            t_point.pose.orientation.z = poses[5, j]
            t_point.pose.orientation.w = poses[6, j]
            t_point.twist.linear.x = twists[0, j]
            t_point.twist.linear.y = twists[1, j]
            t_point.twist.linear.z = twists[2, j]
            t_point.twist.angular.x = twists[3, j]
            t_point.twist.angular.y = twists[4, j]
            t_point.twist.angular.z = twists[5, j]
            task_traj.points.append(t_point)
        rospy.loginfo("Retrieved task trajectory of size %d for '%s'" %
                      (num_pts, trajectory_name))
        return task_traj

    def _get_pose_data(self, df, entity_name, trajectory_name, subsample=0.9):
        # Retrieve data corresponding to task space
        data = df.filter(like=entity_name)
        # Drop any rows that don't have any data for this attribute
        data = data.dropna(how="all")
        # Sort the columns, have to hack a little since there is no good OOB ordering
        pos_cols = [
            "robot_state_%s_position_%s" % (entity_name, d)
            for d in ['x', 'y', 'z']
        ]
        rot_cols = [
            "robot_state_%s_orientation_%s" % (entity_name, d)
            for d in ['x', 'y', 'z', 'w']
        ]
        cols = pos_cols + rot_cols
        data = data[cols]
        # Subsample so it doesn't take all day
        data = data.drop(
            np.linspace(
                1,
                data.shape[0] - 1,
                int(subsample * data.shape[0]),
                dtype=int))
        # Make an array
        data = data.values.T
        return data

    def _get_twist_data(self, df, trajectory_name, subsample=0.9):
        # Retrieve data corresponding to task space
        data = df.filter(like="lbr4_twist")
        # Drop any rows that don't have any data for this attribute
        data = data.dropna(how="all")
        # Sort the columns, have to hack a little since there is no good OOB ordering
        lin_cols = [
            "robot_state_lbr4_twist_linear_%s" % d for d in ['x', 'y', 'z']
        ]
        ang_cols = [
            "robot_state_lbr4_twist_angular_%s" % d for d in ['x', 'y', 'z']
        ]
        cols = lin_cols + ang_cols
        data = data[cols]
        # Subsample so it doesn't take all day
        data = data.drop(
            np.linspace(
                1,
                data.shape[0] - 1,
                int(subsample * data.shape[0]),
                dtype=int))
        # Make an array
        data = data.values.T
        return data

    def _get_final_pose(self, trajectory_name, pose_type):
        if pose_type in ["object", "end_effector"]:
            task_trajectory = self._get_task_trajectory(
                pose_type, trajectory_name)
            last_point = task_trajectory.points[-1]
            return last_point.pose
        else:
            rospy.logwarn("Unknown pose type: %s" % pose_type)
            return None

    def _rosbag_to_pickle(self, bag_fn, pkl_fn):
        data = {"timestamp": []}

        for topic, msg, t in rosbag.Bag(bag_fn).read_messages():
            # Skipping TF messages for now
            if "robot_state" not in topic:
                continue

            data["timestamp"].append(t)

            # LBR4 joint state
            for i in range(len(msg.lbr4.joint_state.position)):
                prefix = "robot_state_lbr4_joint_state_"
                name = prefix + "position_" + str(i)
                if name not in data.keys():
                    data[name] = []
                data[name].append(msg.lbr4.joint_state.position[i])

                name = prefix + "velocity_" + str(i)
                if name not in data.keys():
                    data[name] = []
                data[name].append(msg.lbr4.joint_state.velocity[i])

                name = prefix + "effort_" + str(i)
                if name not in data.keys():
                    data[name] = []
                data[name].append(msg.lbr4.joint_state.effort[i])

            # LBR4 pose
            prefix = "robot_state_lbr4_pose_"
            name = prefix + "position_x"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.lbr4.pose.position.x)

            name = prefix + "position_y"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.lbr4.pose.position.y)

            name = prefix + "position_z"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.lbr4.pose.position.z)

            name = prefix + "orientation_x"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.lbr4.pose.orientation.x)

            name = prefix + "orientation_y"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.lbr4.pose.orientation.y)

            name = prefix + "orientation_z"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.lbr4.pose.orientation.z)

            name = prefix + "orientation_w"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.lbr4.pose.orientation.w)

            # LBR4 twist
            prefix = "robot_state_lbr4_twist_"
            name = prefix + "linear_x"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.lbr4.twist.linear.x)

            name = prefix + "linear_y"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.lbr4.twist.linear.y)

            name = prefix + "linear_z"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.lbr4.twist.linear.z)

            name = prefix + "angular_x"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.lbr4.twist.angular.x)

            name = prefix + "angular_y"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.lbr4.twist.angular.y)

            name = prefix + "angular_z"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.lbr4.twist.angular.z)

            # LBR4 wrench
            prefix = "robot_state_lbr4_wrench_"
            name = prefix + "force_x"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.lbr4.wrench.force.x)

            name = prefix + "force_y"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.lbr4.wrench.force.y)

            name = prefix + "force_z"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.lbr4.wrench.force.z)

            name = prefix + "torque_x"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.lbr4.wrench.torque.x)

            name = prefix + "torque_y"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.lbr4.wrench.torque.y)

            name = prefix + "torque_z"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.lbr4.wrench.torque.z)

            # Reflex finger state
            for i, finger in enumerate(msg.reflex.finger):
                prefix = "robot_state_reflex_finger_"
                name = prefix + str(i) + "_proximal"
                if name not in data.keys():
                    data[name] = []
                data[name].append(finger.proximal)

                name = prefix + str(i) + "_distal_approx"
                if name not in data.keys():
                    data[name] = []
                data[name].append(finger.distal_approx)

                for j in range(9):
                    name = prefix + str(i) + "_contact_" + str(j)
                    if name not in data.keys():
                        data[name] = []
                    data[name].append(finger.contact[j])

                    name = prefix + str(i) + "_pressure_" + str(j)
                    if name not in data.keys():
                        data[name] = []
                    data[name].append(finger.pressure[j])

            # Reflex motor state
            for i, motor in enumerate(msg.reflex.motor):
                prefix = "robot_state_reflex_motor_"
                name = prefix + str(i) + "_joint_angle"
                if name not in data.keys():
                    data[name] = []
                data[name].append(motor.joint_angle)

                name = prefix + str(i) + "_velocity"
                if name not in data.keys():
                    data[name] = []
                data[name].append(motor.velocity)

            # End-effector pose in base frame
            prefix = "robot_state_end_effector_pose_base_frame_"
            name = prefix + "position_x"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.end_effector_pose_base.position.x)

            name = prefix + "position_y"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.end_effector_pose_base.position.y)

            name = prefix + "position_z"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.end_effector_pose_base.position.z)

            name = prefix + "orientation_x"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.end_effector_pose_base.orientation.x)

            name = prefix + "orientation_y"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.end_effector_pose_base.orientation.y)

            name = prefix + "orientation_z"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.end_effector_pose_base.orientation.z)

            name = prefix + "orientation_w"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.end_effector_pose_base.orientation.w)

            # End-effector pose in object frame
            prefix = "robot_state_end_effector_pose_object_frame_"
            name = prefix + "position_x"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.end_effector_pose_obj.position.x)

            name = prefix + "position_y"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.end_effector_pose_obj.position.y)

            name = prefix + "position_z"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.end_effector_pose_obj.position.z)

            name = prefix + "orientation_x"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.end_effector_pose_obj.orientation.x)

            name = prefix + "orientation_y"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.end_effector_pose_obj.orientation.y)

            name = prefix + "orientation_z"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.end_effector_pose_obj.orientation.z)

            name = prefix + "orientation_w"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.end_effector_pose_obj.orientation.w)

            # Object pose
            prefix = "robot_state_object_pose_"
            name = prefix + "position_x"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.object_pose.position.x)

            name = prefix + "position_y"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.object_pose.position.y)

            name = prefix + "position_z"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.object_pose.position.z)

            name = prefix + "orientation_x"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.object_pose.orientation.x)

            name = prefix + "orientation_y"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.object_pose.orientation.y)

            name = prefix + "orientation_z"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.object_pose.orientation.z)

            name = prefix + "orientation_w"
            if name not in data.keys():
                data[name] = []
            data[name].append(msg.object_pose.orientation.w)

        df = pd.DataFrame.from_dict(data, orient='index').transpose()
        df.to_pickle(pkl_fn)

    def _make_dir(self, new_dir):
        try:
            os.makedirs(new_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise


if __name__ == '__main__':
    rospy.init_node("pandas_interface")
    interface = PandasInterface()
    interface.run()
