#!/usr/bin/env python
import sys
import rospy
import actionlib
import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Wrench, PoseStamped
from ll4ma_movement_primitives.util import quaternion
from ll4ma_movement_primitives.phase_variables import ExponentialPV
from ll4ma_movement_primitives.dmps import DMPConfig, DMP
from ll4ma_movement_primitives.msg import DMPPolicyAction, DMPPolicyFeedback, DMPPolicyResult
from ll4ma_trajectory_msgs.msg import TaskTrajectoryPoint
from ll4ma_robot_control_msgs.msg import RobotState


class DMPActionServer:

    def __init__(self, robot_name="robot", pose_tolerance = 0.01):
        self.robot_name = robot_name
        self.pose_tolerance = pose_tolerance
        rospy.init_node("%s_dmp_action_server" % self.robot_name)
        self.rate = rospy.Rate(rospy.get_param("/cmd_rate", 100))
        self.ns = "/%s/dmp/execute_policy" % self.robot_name
        self.server = actionlib.SimpleActionServer(self.ns, DMPPolicyAction, self.goal_cb, False)
        self.feedback = DMPPolicyFeedback()
        self.result = DMPPolicyResult()
        self.robot_state = None
        self.jnt_state = None
        self.goal_pose = None
        self.is_joint_space = True
        self.jnt_cmd = JointState()
        self.task_cmd = TaskTrajectoryPoint()

        self.jnt_des_cmd_topic = rospy.get_param("/%s/jnt_des_topic" % self.robot_name)
        self.task_cmd_topic = rospy.get_param("/%s/task_des_topic" % self.robot_name)
        self.robot_state_topic = rospy.get_param("/%s/robot_state_topic" % self.robot_name)

        self.jnt_des_cmd_pub = rospy.Publisher(self.jnt_des_cmd_topic, JointState, queue_size=1)
        self.task_cmd_pub = rospy.Publisher(self.task_cmd_topic, TaskTrajectoryPoint, queue_size=1)
        rospy.Subscriber(self.robot_state_topic, RobotState, self._robot_state_cb)

    def goal_cb(self, goal):
        rospy.loginfo("New goal received.")
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

    def _execute_policy(self, goal, timeout=10.0):
        rospy.loginfo("Executing policy...")
        phase_var = ExponentialPV()
        dmp_configs = self._ros_to_python_configs(goal.dmp_configs)
        dmps = [DMP(dmp_config) for dmp_config in dmp_configs]
                            
        start_time = rospy.get_time()
        timed_out = False
        prev_time = 0.0
        self.rate.sleep()
        
        # initialize DMPs with current robot state
        # self._set_dmp_init(dmps)
        phase_var.reset()
        
        while not timed_out:
            # update system state
            current_time = rospy.get_time() - start_time
            dt = current_time - prev_time
            phase = phase_var.get_value(dt)
            for dmp in dmps:
                # TODO for now assuming it's either task space or joint space, though it's possible
                # to also set desired joints for null space resolution in task space.
                if dmp.state_type == 'q':
                    self.is_joint_space = True
                    q, _, _ = dmp.get_values(phase, dt)
                    self.jnt_cmd.position[int(dmp.dimension)] = q
                elif dmp.state_type == 'x':
                    self.is_joint_space = False
                    x, _, _ = dmp.get_values(phase, dt)
                    self._set_task_cmd(x, int(dmp.dimension))
                else:
                    rospy.logerr("Unknown element type: %s" % dmp.state_type)

            if self.is_joint_space:
                self.jnt_cmd.header.stamp = rospy.Time.now()
                self.jnt_des_cmd_pub.publish(self.jnt_cmd)
            else:
                self.task_cmd.header.stamp = rospy.Time.now()
                self.task_cmd_pub.publish(self.task_cmd)
            
            # if self._goal_pose_error() < self.pose_tolerance:
            #     rospy.loginfo("Desired pose achieved.")
            #     return True
            
            prev_time = current_time
            timed_out = current_time > timeout
            self.rate.sleep()

        if timed_out:
            rospy.logwarn("Timed out executing DMP policy!")
        return not timed_out
                        
    def _set_task_cmd(self, x, dim):
        if dim == 0:
            self.task_cmd.pose.position.x = x
        elif dim == 1:
            self.task_cmd.pose.position.y = x
        elif dim == 2:
            self.task_cmd.pose.position.z = x
        elif dim == 3:
            self.task_cmd.pose.orientation.x = x
        elif dim == 4:
            self.task_cmd.pose.orientation.y = x
        elif dim == 5:
            self.task_cmd.pose.orientation.z = x
        elif dim == 6:
            self.task_cmd.pose.orientation.w = x

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
    
    def _set_dmp_init(self, dmps):
        for dmp in dmps:
            if dmp.state_type == 'x':
                if dmp.dimension == '0':
                    dmp.set_init(self.robot_state.pose.position.x)
                elif dmp.dimension == '1':
                    dmp.set_init(self.robot_state.pose.position.y)
                elif dmp.dimension == '2':
                    dmp.set_init(self.robot_state.pose.position.z)
                elif dmp.dimension == 'rot':
                    dmp.set_init(np.array([self.robot_state.pose.orientation.x,
                                           self.robot_state.pose.orientation.y,
                                           self.robot_state.pose.orientation.z,
                                           self.robot_state.pose.orientation.w]))
                else:
                    rospy.logerr("Unknown dim type: %s" % str(dim))
            elif dmp.state_type == 'q':
                # TODO
                pass
            else:
                rospy.logerr("Unknown state type: %s" % dmp.state_type)
            dmp.ts.reset() # so it registers new init

    def _get_weights(self, config_weights):
        weights = None
        for weight_vector in config_weights:
            if weights is None:
                weights = np.array(weight_vector.weights)
            else:
                weights = np.vstack((weights, np.array(weight_vector.weights)))
        return weights

    def _get_init(self, config_init):
        if len(config_init) > 1:
            init = np.array(config_init)
        else:
            init = config_init[0]
        return init

    def _get_goal(self, config_goal):
        if len(config_goal) > 1:
            goal = np.array(config_goal)
        else:
            goal = config_goal[0]
        return goal

    def _ros_to_python_configs(self, ros_configs):
        dmp_configs = []
        for ros_config in ros_configs:
            dmp_config = DMPConfig()
            dmp_config.state_type = ros_config.state_type
            dmp_config.dimension  = ros_config.dimension
            dmp_config.num_bfs    = ros_config.num_bfs
            dmp_config.dt         = ros_config.dt
            dmp_config.tau        = ros_config.tau
            dmp_config.alpha      = ros_config.alpha
            dmp_config.beta       = ros_config.beta
            dmp_config.gamma      = ros_config.gamma
            dmp_config.w          = ros_config.w
            dmp_config.init       = ros_config.init
            dmp_config.goal       = ros_config.goal
            dmp_configs.append(dmp_config)
        return dmp_configs
                
                
if __name__ == '__main__':
    argv = rospy.myargv(argv=sys.argv)
    server = DMPActionServer(*argv[1:]) if len(argv) > 1 else DMPActionServer()
    try:
        server.start()
    except rospy.ROSInterruptException:
        pass
        
