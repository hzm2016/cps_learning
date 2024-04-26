#!/usr/bin/env python
import rospy
import actionlib
from ll4ma_policy_learning.msg import (
    Row,
    ProMPConfigROS,
    ProMPPolicyAction,
    ProMPPolicyGoal
)


class ProMPActionClient:

    def __init__(self, robot_name="robot", rospy_init=True):
        self.robot_name = robot_name
        if rospy_init:
            rospy.init_node("%s_promp_action_client" % self.robot_name)
        self.ns = "/%s/promp/execute_policy" % self.robot_name
        self.client = actionlib.SimpleActionClient(self.ns, ProMPPolicyAction)
        rospy.loginfo("Trying to connect with action server...")
        server_running = self.client.wait_for_server(timeout=rospy.Duration(10.0))
        if server_running:
            rospy.loginfo("Connected!")
        else:
            rospy.logwarn("You're probably connected to ProMP action server.")
            # TODO make a better check if this is important, for some reason when this is launched
            # with Gazebo it says it times out waiting, but does so immediately without actually
            # waiting, even though it is in fact connected. I think ROS is making more of a check
            # than just connection, so look into ROS source code for action server to check.
        
    def send_goal(self, promp_configs, num_executions=1, is_joint_space=False, timeout=30.0):
        rospy.loginfo("Sending goal to %s action server..." % self.ns)
        configs = self._python_to_ros_configs(promp_configs)
        goal = ProMPPolicyGoal()
        goal.promp_configs = configs
        goal.num_executions = num_executions
        goal.is_joint_space = is_joint_space
        self.client.send_goal(goal)
        rospy.loginfo("Goal sent successfully.")
        self.wait_for_result(timeout)

    def wait_for_result(self, timeout=30.0):
        rospy.loginfo("Waiting for result...")
        success = self.client.wait_for_result(timeout=rospy.Duration(timeout))
        if success:
            pass
        else:
            rospy.logwarn("Timed out waiting for result.")

    def _python_to_ros_configs(self, promp_configs):
        ros_configs = []
        for promp_config in promp_configs:
            ros_config = ProMPConfigROS()
            ros_config.state_types = promp_config.state_types
            # pack dimension lists
            for dim_list in promp_config.dimensions:
                row = Row()
                row.elements = dim_list
                ros_config.dimensions.rows.append(row)
            ros_config.num_bfs    = promp_config.num_bfs
            ros_config.dt         = promp_config.dt
            ros_config.init       = promp_config.init
            ros_config.goal       = promp_config.goal
            ros_config.mu_w       = promp_config.mu_w.flatten().tolist()
            # pack covariance
            for i in range(promp_config.sigma_w.shape[0]):
                row = Row()
                row.elements = promp_config.sigma_w[i,:].flatten().tolist()
                ros_config.sigma_w.rows.append(row)
            ros_configs.append(ros_config)
        return ros_configs
