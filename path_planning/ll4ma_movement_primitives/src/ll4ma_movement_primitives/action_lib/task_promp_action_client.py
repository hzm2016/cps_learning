#!/usr/bin/env python
import rospy
import actionlib
from ll4ma_movement_primitives.msg import ProMPPolicyAction, ProMPPolicyGoal


class TaskProMPActionClient:
    """
    Client for controlling an active learning session.
    """
    
    def __init__(self, rospy_init=True):
        if rospy_init:
            rospy.init_node("task_promp_action_client")
        self.ns = "/task_promp_action_server"
        self.client = actionlib.SimpleActionClient(self.ns, ProMPPolicyAction)
        rospy.loginfo("Trying to connect with action server...")
        server_running = self.client.wait_for_server(
            timeout=rospy.Duration(10.0))
        if server_running:
            rospy.loginfo("Connected!")
        else:
            rospy.logwarn("You're probably connected to ProMP action server.")
            # TODO make a better check if this is important, for some reason when this is launched
            # with Gazebo it says it times out waiting, but does so immediately without actually
            # waiting, even though it is in fact connected. I think ROS is making more of a check
            # than just connection, so look into ROS source code for action server to check.

    def send_goal(self, wait_for_result=False, timeout=60.0):
        rospy.loginfo("Sending goal to ProMP action server...")
        goal = ProMPPolicyGoal()
        self.client.send_goal(goal)
        rospy.loginfo("Goal sent successfully.")
        if wait_for_result:
            rospy.loginfo("Waiting for result...")
            success = self.client.wait_for_result(
                timeout=rospy.Duration(timeout))
            if not success:
                rospy.logwarn("Timed out waiting for result.")
        result = self.client.get_result()
        return result and result.success
