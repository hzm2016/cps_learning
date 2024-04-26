#!/usr/bin/env python
import rospy
import tf
from geometry_msgs.msg import PoseStamped


class TFRelay:

    def __init__(self):
        self.parent_frame = rospy.get_param("~parent_frame")
        self.child_frame = rospy.get_param("~child_frame")
        self.pose_topic = rospy.get_param("~pose_topic")
        self.tf_listener = tf.TransformListener()
        self.rate = rospy.Rate(500)
        self.pose_pub = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=1)
        self.pose_received = False
        self.pose_stmp = PoseStamped()
        self.pose_stmp.header.frame_id = self.parent_frame

        
    def run(self):
        rospy.loginfo("Relaying pose transforms from '{}' to '{}'...".format(self.parent_frame,
                                                                             self.child_frame))
        while not rospy.is_shutdown():
            self.set_pose()
            if self.pose_received:
                self.pose_pub.publish(self.pose_stmp)
            self.rate.sleep()

    def shutdown(self):
        rospy.loginfo("Exiting.")
            
    def set_pose(self):
        trans = rot = None
        try:
            trans, rot = self.tf_listener.lookupTransform(self.parent_frame, self.child_frame,
                                                          rospy.Time())
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass
        if trans and rot:
            self.pose_stmp.pose.position.x    = trans[0]
            self.pose_stmp.pose.position.y    = trans[1]
            self.pose_stmp.pose.position.z    = trans[2]
            self.pose_stmp.pose.orientation.x = rot[0]
            self.pose_stmp.pose.orientation.y = rot[1]
            self.pose_stmp.pose.orientation.z = rot[2]
            self.pose_stmp.pose.orientation.w = rot[3]
            self.pose_stmp.header.stamp       = rospy.Time.now()
            self.pose_received = True

            
if __name__ == '__main__':
    rospy.init_node("tf_relay")
    relay = TFRelay()
    rospy.on_shutdown(relay.shutdown)
    relay.run()
