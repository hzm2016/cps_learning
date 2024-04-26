#!/usr/bin/env python
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped, Twist, Wrench
from std_srvs.srv import Empty, EmptyResponse
from reflex_msgs.msg import Hand
from ll4ma_logger_msgs.msg import RobotState


class RobotStateRelay:

    def __init__(self):
        self.robot_state = RobotState()
        self.rate = rospy.Rate(500) # TODO can read from param

        self.use_reflex = rospy.get_param("use_reflex", False)
        self.use_object = rospy.get_param("use_object", False)

        self.lbr4_jnt_state_topic = "/lbr4/joint_states"
        self.lbr4_pose_topic      = "/lbr4/pose"
        self.lbr4_twist_topic     = "/lbr4/twist"
        self.lbr4_wrench_topic    = "/lbr4/wrench"
        self.reflex_state_topic   = "/reflex_takktile/hand_state"
        self.ee_pose_base_topic   = "/end_effector/pose/base_frame"
        self.ee_pose_obj_topic    = "/end_effector/pose/object_frame"
        self.obj_pose_topic       = "/object/pose"
        self.robot_state_topic    = "/lbr4/robot_state"

        rospy.Subscriber(self.lbr4_jnt_state_topic, JointState, self._lbr4_jnt_state_cb, queue_size=1)
        rospy.Subscriber(self.lbr4_pose_topic, Pose, self._lbr4_pose_cb, queue_size=1)
        rospy.Subscriber(self.lbr4_twist_topic, Twist, self._lbr4_twist_cb, queue_size=1)
        rospy.Subscriber(self.lbr4_wrench_topic, Wrench, self._lbr4_wrench_cb, queue_size=1)
        rospy.Subscriber(self.ee_pose_base_topic, PoseStamped, self._ee_pose_base_cb, queue_size=1)
        if self.use_reflex:
            rospy.Subscriber(self.reflex_state_topic, Hand, self._reflex_state_cb, queue_size=1)
        if self.use_object:
            rospy.Subscriber(self.obj_pose_topic, PoseStamped, self._obj_pose_cb, queue_size=1)
            rospy.Subscriber(self.ee_pose_obj_topic, PoseStamped, self._ee_pose_obj_cb, queue_size=1)

        self.reset_srv = rospy.Service("/robot_state_relay/reset", Empty, self.reset)
        
        self.robot_state_pub = rospy.Publisher(self.robot_state_topic, RobotState, queue_size=1)

        self.lbr4_jnt_state_received = False
        self.lbr4_pose_received      = False
        self.lbr4_twist_received     = False
        self.lbr4_wrench_received    = False
        self.reflex_state_received   = False
        self.ee_pose_base_received   = False
        self.ee_pose_obj_received    = False
        self.obj_pose_received       = False
        self.should_publish          = False

    def run(self):
        rospy.loginfo("Waiting for LBR4 state...")
        while not rospy.is_shutdown() and not self.lbr4_jnt_state_received:
            self.rate.sleep()
        while not rospy.is_shutdown() and not self.lbr4_pose_received:
            self.rate.sleep()
        while not rospy.is_shutdown() and not self.lbr4_twist_received:
            self.rate.sleep()
        while not rospy.is_shutdown() and not self.lbr4_wrench_received:
            self.rate.sleep()        
        rospy.loginfo("LBR4 state received!")

        if self.use_reflex:
            rospy.loginfo("Waiting for ReFlex state...")
            while not rospy.is_shutdown() and not self.reflex_state_received:
                self.rate.sleep()
            rospy.loginfo("ReFlex state received!")

        if self.use_object:
            rospy.loginfo("Waiting for object pose...")
            while not rospy.is_shutdown() and not self.obj_pose_received:
                self.rate.sleep()
            rospy.loginfo("Object pose received!")
            rospy.loginfo("Waiting for end-effector pose in object frame...")
            while not rospy.is_shutdown() and not self.ee_pose_obj_received:
                self.rate.sleep()
            rospy.loginfo("End-effector pose received!")
        
        rospy.loginfo("Waiting for end-effector pose in base frame...")
        while not rospy.is_shutdown() and not self.ee_pose_base_received:
            self.rate.sleep()
        rospy.loginfo("End-effector pose received!")
        
        rospy.loginfo("Aggregating robot state to topic: %s" % self.robot_state_topic)
        while not rospy.is_shutdown():
            if self.should_publish:
                self.robot_state_pub.publish(self.robot_state)
                self.should_publish = False
            self.rate.sleep()

    def shutdown(self):
        rospy.loginfo("Exiting.")

    # Service function
    def reset(self, req):
        rospy.loginfo("Reset robot state")
        self.robot_state = RobotState()
        self.should_publish = False
        return EmptyResponse()
 
    def _lbr4_jnt_state_cb(self, lbr4_jnt_state):
        self.robot_state.lbr4.joint_state = lbr4_jnt_state
        self.lbr4_jnt_state_received = True
        self.should_publish = True

    def _lbr4_pose_cb(self, lbr4_pose):
        self.robot_state.lbr4.pose = lbr4_pose
        self.lbr4_pose_received = True
        self.should_publish = True

    def _lbr4_twist_cb(self, lbr4_twist):
        self.robot_state.lbr4.twist = lbr4_twist
        self.lbr4_twist_received = True
        self.should_publish = True

    def _lbr4_wrench_cb(self, lbr4_wrench):
        self.robot_state.lbr4.wrench = lbr4_wrench
        self.lbr4_wrench_received = True
        self.should_publish = True

    def _reflex_state_cb(self, reflex_state):
        self.robot_state.reflex = reflex_state
        self.reflex_state_received = True
        self.should_publish = True

    def _ee_pose_base_cb(self, pose_stmp):
        self.robot_state.end_effector_pose_base = pose_stmp.pose
        self.ee_pose_base_received = True
        self.should_publish = True

    def _ee_pose_obj_cb(self, pose_stmp):
        self.robot_state.end_effector_pose_obj = pose_stmp.pose
        self.ee_pose_obj_received = True
        self.should_publish = True

    def _obj_pose_cb(self, pose_stmp):
        self.robot_state.object_pose = pose_stmp.pose
        self.obj_pose_received = True
        self.should_publish = True


if __name__ == '__main__':
    rospy.init_node("robot_state_relay")
    relay = RobotStateRelay()
    rospy.on_shutdown(relay.shutdown)
    relay.run()
