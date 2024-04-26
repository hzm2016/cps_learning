#!/usr/bin/env python
import sys
import rospy
import actionlib
import numpy as np
from scipy.linalg import block_diag
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Wrench, PoseStamped
from ll4ma_movement_primitives.util import quaternion
from ll4ma_movement_primitives.phase_variables import LinearPV
from ll4ma_movement_primitives.basis_functions import GaussianLinearBFS
from ll4ma_movement_primitives.promps import ProMP, ProMPConfig, ProMPLibrary, Waypoint
from ll4ma_policy_learning.msg import (
    ProMPPolicyAction,
    ProMPPolicyFeedback,
    ProMPPolicyResult
)
from ll4ma_trajectory_msgs.msg import TaskTrajectoryPoint
from ll4ma_robot_control_msgs.msg import RobotState
from ll4ma_logger_msgs.srv import SaveTrajectoryExecution, SaveTrajectoryExecutionRequest
from ll4ma_logger.srv import DatabaseConfiguration, DatabaseConfigurationRequest
from std_srvs.srv import SetBool, SetBoolRequest, Empty, Trigger


class ProMPActionServer:

    def __init__(self, robot_name="robot", env_namespace="simulator", pose_tolerance = 0.01):
        self.robot_name = robot_name
        self.env_namespace = env_namespace
        self.pose_tolerance = pose_tolerance
        rospy.init_node("%s_promp_action_server" % self.robot_name)
        self.rate = rospy.Rate(rospy.get_param("/%s/cmd_rate" % self.robot_name))
        self.ns = "/%s/promp/execute_policy" % self.robot_name
        self.server = actionlib.SimpleActionServer(self.ns, ProMPPolicyAction, self.goal_cb, False)
        self.feedback = ProMPPolicyFeedback()
        self.result = ProMPPolicyResult()
        self.robot_state = None
        self.jnt_state = None
        self.goal_pose = None
        self.jnt_cmd = JointState()
        self.task_cmd = TaskTrajectoryPoint()

        self.jnt_des_cmd_topic = rospy.get_param("/%s/jnt_des_topic" % self.robot_name)
        self.task_cmd_topic = rospy.get_param("/%s/task_des_topic" % self.robot_name)
        self.robot_state_topic = rospy.get_param("/%s/robot_state_topic" % self.robot_name)

        self.jnt_des_cmd_pub = rospy.Publisher(self.jnt_des_cmd_topic, JointState, queue_size=1)
        self.task_cmd_pub = rospy.Publisher(self.task_cmd_topic, TaskTrajectoryPoint, queue_size=1)
        rospy.Subscriber(self.robot_state_topic, RobotState, self._robot_state_cb)

    def goal_cb(self, goal):
        rospy.loginfo("[TrajectoryActionServer] New goal received.")
        self.result.success = False

        if self.robot_state is None:
            rospy.logwarn("Robot state is unknown. Waiting for 10 seconds...")
            i = 0
            while i < 100 and self.robot_state is None:
                rospy.sleep(0.1)
                i += 1
        # don't try to run trajectory if you don't know where the robot is
        if self.robot_state is None:
            self.server.set_aborted()
            rospy.logerr("GOAL ABORTED. Current robot state unknown.")
            self.result.success = False
        else:
            self._execute_policy(goal)
            
        # report results to console
        if self.result.success:
            rospy.loginfo("GOAL REACHED.")
            self.server.set_succeeded(self.result)
        else:
            rospy.logwarn("FAILURE. Goal was not reached.")
            self.server.set_aborted()
            
    def start(self):
        rospy.loginfo("Server is running. Waiting for a goal...")
        self.server.start()
        while not rospy.is_shutdown():
            self.rate.sleep()
        rospy.loginfo("Server shutdown. Exiting.")

    def _execute_policy(self, goal, timeout=6.0):
        self.is_joint_space = goal.is_joint_space
        # TODO hack
        duration = 5.0
        
        phase_var = LinearPV()
        phase_var.max_time = duration
        phase_var.reset()
        bfs = GaussianLinearBFS() # TODO set num_bfs
        bfs.max_time = duration
        bfs.reset()

        promp_configs = self._ros_to_python_configs(goal.promp_configs)
        rospy.loginfo("Creating ProMP library from configurations.")
        library = ProMPLibrary(phase_var, bfs)
        promps = [ProMP(config=promp_config) for promp_config in promp_configs]
        for promp in promps:
            promp.phase = phase_var
            promp.bfs = bfs
            library.add_primitive(promp.name, promp)

        for i in range(goal.num_executions):
            rospy.loginfo("Executing run %d of %d." % (i+1, goal.num_executions))

            # reset the environment
            # TODO can generalize this for when rviz simulator is not being used
            try:
                reset_robot = rospy.ServiceProxy("/robot_commander/reset_robot_state", Empty)
                reset_robot()
            except rospy.ServiceException, e:
                pass
            
            phase_var.reset()
            start_time = rospy.get_time()
            timed_out = False
            prev_time = 0.0
            self.rate.sleep()

            # TODO TEMPORARY
            first_time = True
            act1 = 0.0
                
            while not timed_out:
                # update system state
                current_time = rospy.get_time() - start_time
                dt = current_time - prev_time

                phase = phase_var.get_value(dt)
                block_phi, cmd_keys = library.get_block_phi(phase)                
                
                
                # TODO TEMPORARY testing online conditioning
                if current_time > 2.0:
                    if first_time:
                        first_time = False
                        library.add_primitive("new", library.get_copy("default"))
                        library.reset_w("new")
                        waypoint.x = phase
                        if self.is_joint_space:
                            waypoint.value = np.ones(15) 
                        else:
                            waypoint.value = np.ones(14) 
                        library.update_w("new", [waypoint])
                        library.sample_w("new")
                    # command = np.dot(block_phi, library.get_w("new")) # instantaneous transition

                    act1 += 0.001
                    print act1
                    act1 = min(1.0, act1)
                    act2 = 1.0 - act1
                    cmd = library.get_coactivation_point(block_phi, ["new", "default"],
                                                             [act1, act2])
                else:
                    cmd = np.dot(block_phi, library.get_w("default"))

                # cmd = np.dot(block_phi, library.get_w("default"))
                    

                self._set_cmd(cmd, cmd_keys)
                            
                if goal.is_joint_space:
                    self.jnt_cmd.header.stamp = rospy.Time.now()
                    self.jnt_des_cmd_pub.publish(self.jnt_cmd)
                else:
                    # normalize since right now there's no guarantee ProMP gives valid quaternion
                    norm = np.linalg.norm(np.array([self.task_cmd.pose.orientation.x,
                                                    self.task_cmd.pose.orientation.y,
                                                    self.task_cmd.pose.orientation.z,
                                                    self.task_cmd.pose.orientation.w]))
                    self.task_cmd.pose.orientation.x /= norm
                    self.task_cmd.pose.orientation.y /= norm
                    self.task_cmd.pose.orientation.z /= norm
                    self.task_cmd.pose.orientation.w /= norm

                    self.task_cmd.header.stamp = rospy.Time.now()
                    self.task_cmd_pub.publish(self.task_cmd)
                    
                prev_time = current_time
                timed_out = current_time > timeout
                self.rate.sleep()

            if timed_out:
                pass

            # stop logging
            try:
                set_record = rospy.ServiceProxy("%s/set_logging" % self.log_namespace, SetBool)
                req = SetBoolRequest()
                req.data = False;
                resp = set_record(req)
                success = resp.success
            except rospy.ServiceException, e:
                rospy.logwarn("Stop recording request failed: %s" % e)

            # see if the execution was a success
            trajectory_success = False
            try:
                get_eval = rospy.ServiceProxy("/check_trajectory_success", Trigger)
                resp = get_eval()
                trajectory_success = resp.success
            except rospy.ServiceException, e:
                rospy.logwarn("Could not get trajectory evaluation.")
                
            # save data
            try:
                save_data = rospy.ServiceProxy("%s/save_data" % self.log_namespace,
                                               SaveTrajectoryExecution)
                req = SaveTrajectoryExecutionRequest()
                req.execution_success = trajectory_success
                resp = save_data(req)
            except rospy.ServiceException, e:
                rospy.logwarn("Could not save data.")

        # TODO right now it's not checking convergence so only failure is timeout, which at this point
        # does not necessarily mean it failed. Change this return once robust success checks are made.
        return True

    def _set_cmd(self, cmd, keys):
        self.jnt_cmd.position = []
        for c, k in zip(cmd, keys):
            state_type, dim = k.split(".")
            if self.is_joint_space:
                if state_type == 'q':
                    # TODO assuming for now that joints are only being populated in order from same
                    # promp; this assumption will break if joints are split across different ProMPs
                    # to reflect different coupling between different sets of joints.
                    self.jnt_cmd.position.append(c)
            else:
                if state_type == 'x':
                    self._set_task_cmd(c, int(dim))
                        
    def _set_task_cmd(self, cmd, dim):
        if dim == 0:
            self.task_cmd.pose.position.x = cmd
        elif dim == 1:
            self.task_cmd.pose.position.y = cmd
        elif dim == 2:
            self.task_cmd.pose.position.z = cmd
        elif dim == 3:
            self.task_cmd.pose.orientation.x = cmd
        elif dim == 4:
            self.task_cmd.pose.orientation.y = cmd
        elif dim == 5:
            self.task_cmd.pose.orientation.z = cmd
        elif dim == 6:
            self.task_cmd.pose.orientation.w = cmd
        else:
            rospy.logwarn("Unknown task dimension: %s" % str(dim))

    def _pose_error(self, orientation=True, dim=None):
        if dim is not None:
            return abs(self.pose_error[dim])
        elif orientation:
            return np.linalg.norm(self.pose_error)
        else:
            return np.linalg.norm(self.pose_error[:3])

    def _goal_pose_error(self, orientation=True, dim=None):
        if dim is not None:
            if dim == 0:
                return np.linalg.norm(self.robot_state.pose.position.x - self.goal_pose.position.x)
            elif dim == 1:
                return np.linalg.norm(self.robot_state.pose.position.y - self.goal_pose.position.y)
            elif dim == 2:
                return np.linalg.norm(self.robot_state.pose.position.z - self.goal_pose.position.z)
        else:
            return self._cartesian_error(self.robot_state.pose, self.goal_pose, orientation)
        
    def _cartesian_error(self, actual, desired, orientation=True):
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
        return np.linalg.norm(pose_error)
        
    def _joint_error(self, actual, desired, norm=False):
        a = np.array(actual)
        d = np.array(desired)
        err = d - a
        if norm:
            err = np.linalg.norm(err)
        return err

    def _robot_state_cb(self, robot_state):        
        if self.robot_state is None:
            rospy.loginfo("Robot state received.")
            self.num_jnts = len(robot_state.joint_state.position)
            for i in range(self.num_jnts):
                self.jnt_cmd.position.append(0.0)            
        self.robot_state = robot_state

    def _condition_goal_pose(self, promps, goal_poses, phase_val):
        if len(goal_poses) > 1:
            goal_pose = np.random.choice(goal_poses)
        else:
            goal_pose = goal_poses[0]
            
        for promp in promps:
            if promp.state_type == 'x':
                waypoint = Waypoint(x=phase_val)
                if int(promp.dimension) == 0:
                    waypoint.value = goal_pose.position.x
                elif int(promp.dimension) == 1:
                    waypoint.value = goal_pose.position.y
                elif int(promp.dimension) == 2:
                    waypoint.value = goal_pose.position.z
                elif int(promp.dimension) == 3:
                    waypoint.value = goal_pose.orientation.x
                elif int(promp.dimension) == 4:
                    waypoint.value = goal_pose.orientation.y
                elif int(promp.dimension) == 5:
                    waypoint.value = goal_pose.orientation.z
                elif int(promp.dimension) == 6:
                    waypoint.value = goal_pose.orientation.w
                else:
                    rospy.loginfo("Unknown dimension for pose: %s" % promp.dimension)
                promp.sample_w([waypoint])
    

    def _ros_to_python_configs(self, ros_configs):
        promp_configs = []
        for ros_config in ros_configs:
            promp_config = ProMPConfig()
            promp_config.state_types = ros_config.state_types
            # unpack dimension lists
            for row in ros_config.dimensions.rows:
                promp_config.dimensions.append(row.elements)
            promp_config.num_bfs    = ros_config.num_bfs
            promp_config.dt         = ros_config.dt
            promp_config.init       = ros_config.init
            promp_config.goal       = ros_config.goal
            promp_config.mu_w       = ros_config.mu_w
            # unpack sigma_w
            sigma_w = None
            for row in ros_config.sigma_w.rows:
                elements = np.array(row.elements)
                sigma_w = elements if sigma_w is None else np.vstack((sigma_w, elements))
            promp_config.sigma_w = sigma_w
            promp_configs.append(promp_config)
        return promp_configs

    
if __name__ == '__main__':
    argv = rospy.myargv(argv=sys.argv)
    server = ProMPActionServer(*argv[1:]) if len(argv) > 1 else ProMPActionServer()
    try:
        server.start()
    except rospy.ROSInterruptException:
        pass
        
