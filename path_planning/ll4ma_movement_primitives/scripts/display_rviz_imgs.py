#!/usr/bin/env python
import os
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, TriggerResponse


class RVizImageDisplay:
    def __init__(self):
        self.img_type = rospy.get_param("~image_type", "")
        self.img_path = rospy.get_param("~image_path", "~/.rosbags/demos/imgs")
        self.img_path = os.path.expanduser(self.img_path)
        self.img_file = rospy.get_param("~image_file", "")
        self.idx_file = os.path.join(self.img_path, "img.index")
        self.rate = rospy.Rate(1)
        self.img_msg = None
        self.image_pub = rospy.Publisher(
            "/{}_image".format(self.img_type), Image, queue_size=10)
        self.img_srv = rospy.Service("/display_{}_image".format(self.img_type),
                                     Trigger, self._set_img)

    def run(self):
        rospy.loginfo("Waiting for image to be set.")
        while self.img_msg is None and not rospy.is_shutdown():
            self.rate.sleep()
        rospy.loginfo("Image for '{}' is set!".format(self.img_type))

        while not rospy.is_shutdown():
            self.image_pub.publish(self.img_msg)
            self.rate.sleep()

    def show_img(self, img_name):
        filename = os.path.join(self.img_path, img_name)
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        self.img_msg = CvBridge().cv2_to_imgmsg(img, "bgr8")

    def _set_img(self, req):
        with open(self.idx_file, 'r') as f:
            img_name = "%s_%03d.png" % (self.img_type, int(f.readline()) - 1)
        self.show_img(img_name)
        return TriggerResponse(success=True)


if __name__ == '__main__':
    rospy.init_node("rviz_image_display")
    display = RVizImageDisplay()
    # Display image from file if provided, otherwise wait for service call
    if display.img_file:
        display.show_img(display.img_file)
        display.run()
    else:
        try:
            display.run()
        except rospy.ROSInterruptException:
            pass
