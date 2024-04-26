#!/usr/bin/env python
import os
import errno
import rospy
import numpy as np
import random
from copy import deepcopy
import cPickle as pickle
from scipy import interpolate
from actionlib import SimpleActionServer
from trajectory_action_lib import JointTrajectoryActionClient
from ll4ma_rosbag_utils.srv import RosbagAction, RosbagActionRequest
from pyquaternion import Quaternion as pyQuaternion
from ll4ma_movement_primitives.util import quaternion
from ll4ma_movement_primitives.phase_variables import LinearPV
from ll4ma_movement_primitives.basis_functions import GaussianLinearBFS
from ll4ma_movement_primitives.promps import (
    Waypoint, ActiveLearner, python_to_ros_config, ros_to_python_config,
    python_to_ros_waypoint)
from ll4ma_movement_primitives.promps import active_learner_util as al_util
from ll4ma_movement_primitives.promps import _VALIDATION
from ll4ma_movement_primitives.msg import Row, TwoDArray
from ll4ma_movement_primitives.msg import ProMPPolicyAction, ProMPPolicyFeedback, ProMPPolicyResult
from ll4ma_movement_primitives.srv import AddProMPDemo, AddProMPDemoResponse
from ll4ma_movement_primitives.srv import SetTaskLabel, SetTaskLabelResponse
from trajectory_msgs.msg import JointTrajectoryPoint
from ll4ma_logger_msgs.msg import RobotState
from ll4ma_trajectory_msgs.msg import TaskTrajectory, TaskTrajectoryPoint
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse
from geometry_msgs.msg import Pose, PoseStamped
from rospy_service_helper import (
    zero_reflex_tactile, set_controller_mode, visualize_promps,
    visualize_poses, visualize_joint_trajectory, generate_joint_trajectory,
    get_smooth_traj, set_rosbag_status, command_reflex_hand, SRV_NAMES)
try:
    from ll4ma_planner.srv import EETrajPlan, EETrajPlanRequest
except:
    pass


class TaskProMPActionServer:
    """
    Action server controlling an active learning session.

    This class relies heavily on private parameters from the ROS parameter
    server. As such, you should really only use this from a launch file.
    """
    
    def __init__(self):
        # Read parameters from ROS parameter server
        self.num_bfs = rospy.get_param("~num_bfs")
        self.config_path = rospy.get_param("~config_path")
        self.config_filename = rospy.get_param("~config_filename")
        self.demo_path = rospy.get_param("~demo_path")
        self.rate_val = rospy.get_param("~rate")
        self.joint_tolerance = rospy.get_param("~joint_tolerance")
        self.timeout = rospy.get_param("~timeout")
        self.base_frame = rospy.get_param("~base_frame")
        self.use_reflex = rospy.get_param("/use_reflex", False)
        self.offline_mode = rospy.get_param("/offline_mode", True)
        self.experiment_type = rospy.get_param("/experiment_type", None)
        self.active_learning_type = rospy.get_param("/active_learning_type", None)
        self.object_type = rospy.get_param("/object_type", None)
        self.random_seed = rospy.get_param("~random_seed")
        if not self.offline_mode:
            self.joint_traj_client = JointTrajectoryActionClient(rospy_init=False)

        self.ns = "/task_promp_action_server"
        self.server = SimpleActionServer(self.ns, ProMPPolicyAction, self._goal_cb, False)
        self.feedback = ProMPPolicyFeedback()
        self.result = ProMPPolicyResult()

        self.rate = rospy.Rate(self.rate_val)
        self.robot_state = None
        self.review_status = False

        self.backup_path = os.path.expanduser(os.path.join(self.demo_path, "backup"))
        self.backup_idx_file = os.path.join(self.backup_path, "al_backup.index")

        self._set_active_learner()

        # Topics this node needs access to
        self.topics = {
            "task_cmd": "/lbr4/task_cmd",
            "robot_state": "/lbr4/robot_state",
            "viz_pose": "/lbr4/visualize_pose"
        }

        # Services that this node will call
        self.srvs = {
            "viz_promps": SRV_NAMES["viz_promps"],
            "viz_poses": SRV_NAMES["viz_poses"],
            "viz_traj": SRV_NAMES["viz_traj"],
            "gen_traj": SRV_NAMES["gen_traj"],
            "get_smooth_traj": SRV_NAMES["get_smooth_traj"],
            "del_latest_rosbag": SRV_NAMES["delete_latest_rosbag"]
        }
        if self.use_reflex:
            self.srvs["grasp_reflex"] = SRV_NAMES["grasp_reflex"]
        if not self.offline_mode:
            self.srvs["set_cmd_mode"] = SRV_NAMES["set_cmd_mode"]
            self.srvs["set_krc_mode"] = SRV_NAMES["set_krc_mode"]
            self.srvs["set_recording"] = SRV_NAMES["set_recording"]
            self.srvs["ik_planner"] = "/ll4ma_planner/get_ik_ee_traj"
            if self.use_reflex:
                self.srvs["zero_tactile"] = SRV_NAMES["zero_reflex_tactile"]

        # Publishers
        self.viz_pose_pub = rospy.Publisher(
            self.topics["viz_pose"], PoseStamped, queue_size=1)
        self.task_cmd_pub = rospy.Publisher(
            self.topics["task_cmd"], PoseStamped, queue_size=1)

        # Subscribers
        rospy.Subscriber(self.topics["robot_state"], RobotState,
                         self._robot_state_cb)

    def start_server(self):
        rospy.loginfo("Initializing services I offer...")
        prev_promp = rospy.Service("{}/preview_promp".format(self.ns), Trigger,
                                   self.review_recent_task)
        clear_all = rospy.Service("{}/clear_all".format(self.ns), Trigger,
                                  self.clear_all)
        gen_task = rospy.Service("{}/generate_task_instance".format(self.ns),
                                 Trigger, self.generate_next_task_instance)
        add_demo = rospy.Service("{}/add_demonstration".format(self.ns),
                                 AddProMPDemo, self.add_demo)
        lbl_task = rospy.Service("{}/label_task_instance".format(self.ns),
                                 SetTaskLabel, self.label_task_instance)
        discard_task = rospy.Service("{}/discard_task_instance".format(self.ns),
                                     Trigger, self.discard_task_instance)
        add_neg_inst = rospy.Service("{}/add_negative_instance".format(self.ns),
                                     Trigger, self.add_negative_instance)
        rev_status = rospy.Service("{}/set_review_status".format(self.ns),
                                   SetBool, self.set_review_status)
        rec_pose = rospy.Service("{}/record_ee_pose_in_obj".format(self.ns),
                                 Trigger, self.record_ee_pose_in_obj_frame)
        lift = rospy.Service("{}/lift_object".format(self.ns), Trigger,
                             self.lift_object)
        set_pose = rospy.Service("{}/set_current_object_pose".format(self.ns),
                                 Trigger, self.set_current_object_pose)

        # Make sure services that will be called are ready to receive requests
        rospy.loginfo("Waiting for services I need...")
        for srv in self.srvs.keys():
            rospy.loginfo("    %s" % self.srvs[srv])
            rospy.wait_for_service(self.srvs[srv])
        rospy.loginfo("Services are up!")

        # self.active_learner.visualize_region()

        rospy.loginfo("Task ProMP Server is running.")
        self.server.start()
        while not rospy.is_shutdown():
            self.rate.sleep()

    def stop(self):
        rospy.loginfo("Task ProMP Server shutdown. Exiting.")

    # === BEGIN service functions being offered ===============================

    def review_recent_task(self, req):
        if not self.active_learner.has_unlabeled_instance():
            return TriggerResponse(success=False, message="There is no unlabeled instance.")
        if not self.active_learner.promp_library.is_ready():
            return TriggerResponse(success=False, message="ProMP library is not ready for execution.")
        if ll4ma_planner not in sys.modules:
            return TriggerResponse(success=False,
                                   message=("LL4MA Planner is not loaded. "
                                            "Cannot generate joint trajectory."))
        ee_gmm_params = self.active_learner.get_ee_gmm_params()
        if not ee_gmm_params:
            return TriggerResponse(success=False,
                                   message="GMM of End-Effector pose in object frame not learned yet.")

        # Object should be placed where instance is shown, but use the actual
        # object state in case the user is inaccurate
        obj_pose = self.robot_state.object_pose
        base_TF_object = al_util.tf_from_pose(obj_pose)

        query_waypoints = []
        for object_mu_ee, object_sigma_ee in zip(ee_gmm_params['means'], ee_gmm_params['covs']):
            base_mu_ee = al_util.TF_mu(base_TF_object, object_mu_ee)
            base_sigma_ee = al_util.TF_sigma(base_TF_object, object_sigma_ee)
            waypoint = Waypoint()
            waypoint.phase_val = 1.0
            waypoint.condition_keys = ["x.{}".format(i) for i in range(7)]
            waypoint.values = base_mu_ee
            waypoint.sigma = base_sigma_ee
            query_waypoints.append(waypoint)

        # Select the most likely ProMP to achieve the task
        promp_name, waypoint = self.active_learner.get_most_likely_promp(query_waypoints)
        promp = self.active_learner.get_promp(promp_name)

        promp_config = promp.get_config()
        waypoints = [waypoint]

        # Visualize generated samples as the end-effector trace for each sample
        ros_config = python_to_ros_config(promp_config)
        ros_waypoints = [python_to_ros_waypoint(waypoint) for waypoint in waypoints]
        visualize_promps(self.srvs["viz_promps"], ros_config, waypoints=ros_waypoints)

        # Generate trajectory to show for rViz (and store for execution)
        traj = promp.generate_trajectory(dt=0.5, duration=10.0, waypoints=waypoints)
        self.task_trajectory = TaskTrajectory()
        for j in range(len(traj['x'][0])):
            t_point = TaskTrajectoryPoint()
            t_point.pose.position.x = traj['x'][0][j]
            t_point.pose.position.y = traj['x'][1][j]
            t_point.pose.position.z = traj['x'][2][j]
            # Need to normalize quaternion
            q = np.array([traj['x'][3][j], traj['x'][4][j], traj['x'][5][j], traj['x'][6][j]])
            q /= np.linalg.norm(q)
            t_point.pose.orientation.x = q[0]
            t_point.pose.orientation.y = q[1]
            t_point.pose.orientation.z = q[2]
            t_point.pose.orientation.w = q[3]
            self.task_trajectory.points.append(t_point)

        # Use planner to get joint trajectory from task space trajectory
        poses = [point.pose for point in self.task_trajectory.points]
        req = EETrajPlanRequest()
        req.arm_jstate = self.robot_state.lbr4.joint_state
        req.t_steps = len(poses)
        req.duration = 10.0
        req.des_ee_poses = poses
        try:
            planner = rospy.ServiceProxy(self.srvs["ik_planner"], EETrajPlan)
            resp = planner(req)
        except rospy.ServiceException as e:
            rospy.logerr("Service call to IK planner failed: {}".format(e))
            return TriggerResponse(success=False)

        # Get a smooth trajectory
        self.joint_trajectory = get_smooth_traj(self.srvs["get_smooth_traj"],
                                                resp.arm_joint_traj)

        if self.joint_trajectory is not None:
            success = visualize_joint_trajectory(self.srvs["viz_traj"],
                                                 self.joint_trajectory)
            return TriggerResponse(success=success, message=promp_name)
        else:
            return TriggerResponse(
                success=False, message="Failed to generate joint trajectory.")

    def clear_all(self, req):
        self._reset()
        rospy.loginfo("Action server successfully reset.")
        return TriggerResponse(success=True)

    def generate_next_task_instance(self, req):
        resp = TriggerResponse()
        if self.active_learner.has_unlabeled_instance():
            return TriggerResponse(
                success=False,
                message="There exists an unlabeled instance, need to label it."
            )

        rospy.loginfo("Generating next task instance...")
        task_instance = self.active_learner.generate_task_instance()
        if task_instance is None:
            rospy.logwarn("No instance could be generated.")
            return TriggerResponse(success=False)

        # Visualize the selected instance
        pose = self._get_pose_from_planar(task_instance.object_planar_pose)
        visualize_poses(self.srvs["viz_poses"], [pose])
        rospy.loginfo("Task instance is being displayed.")
        resp.success = True
        return resp

    def discard_task_instance(self, req):
        resp = TriggerResponse()
        resp.success = self.active_learner.discard_task_instance()
        return resp

    def label_task_instance(self, req):
        resp = SetTaskLabelResponse()
        resp.success = self.active_learner.label_task_instance(req.label)
        return resp

    def add_negative_instance(self, req):
        rospy.loginfo("Adding most recent task instance as negative to GMM")
        resp = TriggerResponse()
        resp.success = self.active_learner.add_instance_to_obj_gmm()
        self._backup()
        return resp

    def add_demo(self, req):
        if req.w:
            rospy.loginfo("Adding new demonstration to ProMP library")
            w = np.array(req.w)
            config = ros_to_python_config(req.config)
            self.active_learner.add_promp_demo(w, config, req.data_name)
        self.active_learner.set_instance_trajectory_name(req.data_name)
        self._backup()
        return AddProMPDemoResponse(success=True)

    def set_review_status(self, req):
        self.review_status = True
        return SetBoolResponse(success=True)

    def record_ee_pose_in_obj_frame(self, req):
        pose = deepcopy(self.robot_state.end_effector_pose_obj)
        self.active_learner.set_instance_ee_pose_in_obj(pose)
        p = pose.position
        q = pose.orientation
        if q.x + q.y + q.z + q.w == 0:
            rospy.logwarn("End-effector pose in object frame is not known.")
            return TriggerResponse(success=False)
        else:
            instance = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
            self.active_learner.add_instance_to_ee_gmm(instance)
            return TriggerResponse(success=True)

    def lift_object(self, req):
        self._grasp()
        self._lift()
        return TriggerResponse(success=True)

    def set_current_object_pose(self, req):
        obj_pose = self.robot_state.object_pose
        self.active_learner.set_instance_object_pose(obj_pose)
        return TriggerResponse(success=True)

    # === END service functions being offered ==================================

    def _goal_cb(self, action_goal):
        if self.offline_mode:
            rospy.logwarn("ProMP action server is in OFFLINE MODE. Cannot execute.")
            return False

        rospy.loginfo("New ProMP action goal received.")
        self.result.success = False

        if not self.review_status:
            rospy.logerr("No ProMP has been reviewed.")
            self.server.set_aborted()
            return False

        if self.robot_state is None or not self.robot_state.lbr4.joint_state.position:
            rospy.logwarn("Robot state is unknown. Waiting for 10 seconds...")
            i = 0
            while i < 100 and self.robot_state is None:
                rospy.sleep(0.1)
                i += 1
        # Don't try to run if you don't know where the robot is
        if self.robot_state is None:
            self.server.set_aborted()
            rospy.logerr("GOAL ABORTED. Current robot state unknown.")
            return False

        # Try to run policy
        self._execute_policy(action_goal)

        # Report results to console
        if self.result.success:
            rospy.loginfo("GOAL REACHED.")
            self.server.set_succeeded(self.result)
        else:
            rospy.logwarn("FAILURE. Goal was not reached.")
            self.server.set_aborted()

    def _execute_policy(self, goal):
        """
        Execute the generated policy. Note that this is executing in joint space
        for now until a better task space controller is up and running. Therefore
        all the error checking here is in joint space.
        """
        # Do some error checking
        if not self.joint_trajectory:
            rospy.logerr("No trajectory has been set. Aborting.")
            return False

        # Move to start position if current position differs too much
        start_error = self._joint_error(self.robot_state.lbr4.joint_state.position,
                                        self.joint_trajectory.points[0].positions)
        if start_error > self.joint_tolerance:
            rospy.loginfo("Moving to policy start position...")
            self.joint_traj_client.send_goal(position=self.joint_trajectory.points[0].positions,
                                             wait_for_result=True)

        # If for some reason we are still not at the start, just bail
        start_error = self._joint_error(self.robot_state.lbr4.joint_state.position,
                                        self.joint_trajectory.points[0].positions)

        if start_error > self.joint_tolerance:
            rospy.logerr("GOAL ABORTED. Too far from start:\n"
                         "    Current error: {}\n"
                         "        Tolerance: {}\n".format(
                             start_error, self.joint_tolerance))
            self.result.success = False
            return False

        rospy.loginfo("Executing the policy...")
        rospy.sleep(3.0)  # Just give a little time from MTS

        # Zero out the ReFlex tactile sensors
        if self.use_reflex:
            zero_reflex_tactile(self.srvs["zero_tactile"])

        # Start rosbag recorder
        set_rosbag_status(self.srvs["set_recording"], self.demo_path, recording=True)

        # Send trajectory to action server for execution
        success = self.joint_traj_client.send_goal(self.joint_trajectory, wait_for_result=True)

        if not success:
            # Stop rosbag recorder and discard rosbag
            rospy.sleep(2.0)
            set_rosbag_status(self.srvs["set_recording"], self.demo_path, recording=False)
            rospy.sleep(2.0)
            try:
                discard = rospy.ServiceProxy(self.srvs["del_latest_rosbag"], RosbagAction)
                discard(RosbagActionRequest(path=self.demo_path))
            except rospy.ServiceException as e:
                rospy.logwarn("Service call to delete rosbag failed %s" % e)
        else:
            if self.use_reflex:
                self._grasp()
                self._lift()
            # Stop rosbag recorder
            set_rosbag_status(self.srvs["set_recording"], self.demo_path, recording=False)
            rospy.loginfo("ProMP execution complete.")

        self.result.success = True
        self._reset()
        return True

    def _grasp(self):
        rospy.loginfo("Closing hand...")
        command_reflex_hand(self.srvs["grasp_reflex"])
        rospy.loginfo("Object grasped.")

    def _lift(self):
        rospy.loginfo("Lifting...")
        start_point = TaskTrajectoryPoint()
        end_point = TaskTrajectoryPoint()
        current_pose = self.robot_state.end_effector_pose_base
        start_point.pose = deepcopy(current_pose)
        end_point.pose = deepcopy(current_pose)
        end_point.pose.position.z += 0.2  # TODO change this
        task_trajectory = TaskTrajectory()
        task_trajectory.points = self._interpolate_task_points(start_point, end_point)

        # Generate the corresponding joint trajectory
        joint_trajectory = generate_joint_trajectory(self.srvs["gen_traj"],
                                                     self.robot_state.lbr4.joint_state.position,
                                                     task_trajectory)

        # Subsample the points (otherwise smoothing takes too long)
        # joint_trajectory.points = joint_trajectory.points[::5]

        # Get a smooth trajectory
        smooth_trajectory = get_smooth_traj(self.srvs["get_smooth_traj"], joint_trajectory)

        # Send trajectory to action server for execution
        self.joint_traj_client.send_goal(smooth_trajectory, wait_for_result=True)
        rospy.loginfo("Object lifted.")

    def _cartesian_error(self, actual, desired, orientation=True, norm=True):
        p1 = actual.position
        p2 = desired.position
        p_err = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
        if orientation:
            oa = actual.orientation
            od = desired.orientation
            q1 = np.array([oa.x, oa.y, oa.z, oa.w])
            q2 = np.array([od.x, od.y, od.z, od.w])
            q_err = np.squeeze(quaternion.err(q1, q2))
            pose_error = np.hstack((p_err, q_err))
        else:
            pose_error = np.hstack((p_err, np.zeros(3)))
        if norm:
            return np.linalg.norm(pose_error)
        else:
            return pose_error

    def _joint_error(self, actual, desired, norm=True):
        err = np.array(desired) - np.array(actual)
        if norm:
            err = np.linalg.norm(err)
        return err

    def _robot_state_cb(self, robot_state):
        if self.robot_state is None:
            # Initialize the start point so we can directly utilize move
            # to start functionality
            self.start_pose = robot_state.end_effector_pose_base
            # Set goal as starting point until it is changed with a trajectory set
            self.goal_pose = robot_state.end_effector_pose_base
            rospy.loginfo("Robot state received!")
        self.robot_state = robot_state

    def _reset(self):
        self.task_trajectory = None
        self.joint_trajectory = None
        self.review_status = False

    def _backup(self):
        self.active_learner.write_metadata()
        # Get index (or create new index file if it doesn't exist)
        if not os.path.isfile(self.backup_idx_file):
            with open(self.backup_idx_file, 'w+') as f:
                f.write('2')
            idx = 1
        else:
            with open(self.backup_idx_file, 'r') as f:
                idx = int(f.readline())
        # Pickle the data
        with open(os.path.join(self.backup_path, "active_learner_%03d.pkl" % idx), 'w+') as f:
            pickle.dump(self.active_learner, f)
        # Increment index
        with open(self.backup_idx_file, 'w+') as f:
            f.write(str(idx + 1))

    def _set_active_learner(self):
        if self.active_learning_type == _VALIDATION:
            # TODO testing
            learner_filename = os.path.join(self.demo_path, "test_learner.pkl")
            with open(learner_filename, 'r') as f:
                self.active_learner = pickle.load(f)
            rospy.logwarn("Using validation active learner")
        else:
            try:
                os.makedirs(self.backup_path)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise

            if os.path.isfile(self.backup_idx_file):
                rospy.logwarn("Loading a cached active learner.")
                with open(self.backup_idx_file, 'r') as f:
                    idx = int(f.readline()) - 1  # Index of most recently saved
                with open(
                        os.path.join(self.backup_path,
                                     "active_learner_%03d.pkl" % idx),
                        'r') as f:
                    self.active_learner = pickle.load(f)
                # self.active_learner._generate_visualizations()
            else:
                rospy.logwarn("Creating a new active learner.")
                self.active_learner = ActiveLearner(self.config_path,
                                                    self.config_filename,
                                                    self.demo_path,
                                                    self.active_learning_type,
                                                    self.experiment_type,
                                                    self.object_type,
                                                    self.random_seed)

    def _get_tf_from_planar(self, planar):
        table_position = self.active_learner.table_position
        table_quaternion = self.active_learner.table_quaternion
        base_TF_object = al_util.table_planar_to_base_tf(planar.x, planar.y, planar.z, planar.theta,
                                                         table_position, table_quaternion)
        return base_TF_object

    def _get_pose_from_planar(self, planar):
        """
        Convert planar pose in table frame to a full pose (position and quaternion)
        with respect to the base frame.
        
        Args:
            planar (PlanarPose): Planar pose of object on table

        Returns:
            pose (Pose): Pose of object in base frame
        """
        base_TF_object = self._get_tf_from_planar(planar)
        obj_quaternion = al_util.tf_to_quaternion(base_TF_object)
        pose = Pose()
        pose.position.x = base_TF_object[0, 3]
        pose.position.y = base_TF_object[1, 3]
        pose.position.z = base_TF_object[2, 3]
        pose.orientation.x = obj_quaternion[0]
        pose.orientation.y = obj_quaternion[1]
        pose.orientation.z = obj_quaternion[2]
        pose.orientation.w = obj_quaternion[3]
        return pose

    def _interpolate_task_points(self, start_point, end_point, duration=5.0):
        # Interpolate position
        s = start_point.pose.position
        e = end_point.pose.position
        xs_f = interpolate.interp1d([0, duration], [s.x, e.x])
        ys_f = interpolate.interp1d([0, duration], [s.y, e.y])
        zs_f = interpolate.interp1d([0, duration], [s.z, e.z])
        ts = np.linspace(0.0, duration, int(duration * self.rate_val))
        xs = xs_f(ts)
        ys = ys_f(ts)
        zs = zs_f(ts)

        # Interpolate orientation
        wp1 = start_point.pose.orientation
        wp2 = end_point.pose.orientation
        # We don't want to interpolate if they're already equal, will cause
        # divide by zero error in pyQuaternion
        wp1_arr = np.array([wp1.w, wp1.x, wp1.y, wp1.z])
        wp2_arr = np.array([wp2.w, wp2.x, wp2.y, wp2.z])
        if np.allclose(wp1_arr, wp2_arr):
            qs = [wp1_arr for _ in range(len(ts))]
        else:
            q1 = pyQuaternion(x=wp1.x, y=wp1.y, z=wp1.z, w=wp1.w)
            q2 = pyQuaternion(x=wp2.x, y=wp2.y, z=wp2.z, w=wp2.w)
            qs = pyQuaternion.intermediates(
                q1, q2, len(ts), include_endpoints=True)
            qs = [q.elements for q in qs]  # getting list form generator

        # Convert back to trajectory points
        interp_points = []
        for i in range(len(ts)):
            t_point = TaskTrajectoryPoint()
            t_point.pose.position.x = xs[i]
            t_point.pose.position.y = ys[i]
            t_point.pose.position.z = zs[i]
            t_point.pose.orientation.x = qs[i][1]
            t_point.pose.orientation.y = qs[i][2]
            t_point.pose.orientation.z = qs[i][3]
            t_point.pose.orientation.w = qs[i][0]
            interp_points.append(t_point)
        return interp_points


if __name__ == '__main__':
    rospy.init_node("task_promp_action_server")
    import sys
    argv = rospy.myargv(argv=sys.argv)
    try:
        server = TaskProMPActionServer(
            *argv[1:]) if len(argv) > 1 else TaskProMPActionServer()
        rospy.on_shutdown(server.stop)
        server.start_server()
    except rospy.ROSInterruptException:
        pass
