#!/usr/bin/env python
import rospy
import tf
from dbot_ros_msgs.msg import ObjectState


class ObjectTFBroadcaster:

    def __init__(self):
        self.object_frame = rospy.get_param("~object_frame")
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.rate = rospy.Rate(500)
        self._mesh_pose = None
        rospy.Subscriber("/particle_tracker/object_state", ObjectState, self._set_mesh_pose)

    def run(self):
        rospy.loginfo("Relaying object mesh pose if detected...")
        while not rospy.is_shutdown():
            if self._mesh_pose:
                self._publish_mesh_tf()
            self.rate.sleep()

    def shutdown(self):
        rospy.loginfo("Exiting.")

    def _set_mesh_pose(self, obj_state):
        self._mesh_pose = obj_state.pose.pose
                
    def _publish_mesh_tf(self):
        pos  = [self._mesh_pose.position.x,
                self._mesh_pose.position.y,
                self._mesh_pose.position.z]
        quat = [self._mesh_pose.orientation.x,
                self._mesh_pose.orientation.y,
                self._mesh_pose.orientation.z,
                self._mesh_pose.orientation.w]
        self.tf_broadcaster.sendTransform(pos, quat, rospy.Time.now(), self.object_frame,
                                          "camera_rgb_optical_frame")
            
            
if __name__ == '__main__':
    rospy.init_node("obj_tf_broadcaster")
    broadcaster = ObjectTFBroadcaster()
    rospy.on_shutdown(broadcaster.shutdown)
    broadcaster.run()
