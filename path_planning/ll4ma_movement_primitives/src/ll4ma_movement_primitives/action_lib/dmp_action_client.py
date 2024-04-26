#!/usr/bin/env python
import rospy
import actionlib
from ll4ma_movement_primitives.msg import DMPConfigROS, DMPPolicyAction, DMPPolicyGoal


class DMPActionClient:

    def __init__(self, robot_name="robot", rospy_init=True):
        self.robot_name = robot_name
        if rospy_init:
            rospy.init_node("%s_dmp_action_client" % self.robot_name)
        self.ns = "/%s/dmp/execute_policy" % self.robot_name
        self.client = actionlib.SimpleActionClient(self.ns, DMPPolicyAction)
        rospy.loginfo("Trying to connect with action server...")
        server_running = self.client.wait_for_server(timeout=rospy.Duration(10.0))
        if server_running:
            rospy.loginfo("Connected!")
        else:
            rospy.logwarn("You're probably connected to DMP action server.")
            # TODO make a better check if this is important, for some reason when this is launched
            # with Gazebo it says it times out waiting, but does so immediately without actually
            # waiting, even though it is in fact connected. I think ROS is making more of a check
            # than just connection, so look into ROS source code for action server to check.
        
    def send_goal(self, dmp_configs, timeout=30.0):
        rospy.loginfo("Sending goal to %s action server..." % self.ns)
        configs = self._python_to_ros_configs(dmp_configs)
        goal = DMPPolicyGoal()
        goal.dmp_configs = configs
        self.client.send_goal(goal)
        rospy.loginfo("Goal sent successfully.")
        self.wait_for_result(timeout)

    def wait_for_result(self, timeout=10.0):
        rospy.loginfo("Waiting for result...")
        success = self.client.wait_for_result(timeout=rospy.Duration(timeout))
        
    def _python_to_ros_configs(self, dmp_configs):
        ros_configs = []
        for dmp_config in dmp_configs:
            ros_config = DMPConfigROS()
            ros_config.state_type = dmp_config.state_type
            ros_config.dimension  = dmp_config.dimension
            ros_config.num_bfs    = dmp_config.num_bfs
            ros_config.dt         = dmp_config.dt
            ros_config.tau        = dmp_config.tau
            ros_config.alpha      = dmp_config.alpha
            ros_config.beta       = dmp_config.beta
            ros_config.gamma      = dmp_config.gamma
            ros_config.w          = dmp_config.w
            ros_config.init       = dmp_config.init
            ros_config.goal       = dmp_config.goal
            ros_configs.append(ros_config)
        return ros_configs
