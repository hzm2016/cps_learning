#!/usr/bin/env python
import sys
import rospy
from matplotlib import colors
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from ll4ma_movement_primitives.promps import active_learner_util as al_util

if __name__ == '__main__':
    rospy.init_node("table_visualizer")
    config_path = rospy.get_param("~config_path")
    config_filename = rospy.get_param("~config_filename")
    experiment_type = rospy.get_param("~experiment_type")
    config = al_util.load_config(config_path, config_filename)

    if experiment_type == "experiment_1":
        table_pose = config["lbr4_base_link__to__table_1_center"]
    elif experiment_type == "experiment_2":
        table_pose = config["lbr4_base_link__to__table_2_center"]
    else:
        rospy.logerr("Unknown experiment type: {}".format(experiment_type))
        sys.exit(1)

    pub = rospy.Publisher("/visualization/table_mesh", Marker, queue_size=1)
    rate = rospy.Rate(100)

    def get_color(color_name):
        """
        color can be any name from this page:
        http://matplotlib.org/mpl_examples/color/named_colors.hires.png
        """
        converter = colors.ColorConverter()
        c = converter.to_rgba(colors.cnames[color_name])
        return ColorRGBA(*c)

    marker = Marker()
    marker.header.frame_id = "lbr4_base_link"
    marker.id = 123
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.color = get_color("mediumaquamarine")
    marker.color.a = 0.9
    marker.scale.x = config["table_x"]
    marker.scale.y = config["table_y"]
    marker.scale.z = config["table_z"]
    marker.pose.position.x = table_pose["position"]["x"]
    marker.pose.position.y = table_pose["position"]["y"]
    marker.pose.position.z = table_pose["position"]["z"]
    marker.pose.position.z -= config["table_z"] / 2.0
    marker.pose.orientation.x = table_pose["orientation"]["x"]
    marker.pose.orientation.y = table_pose["orientation"]["y"]
    marker.pose.orientation.z = table_pose["orientation"]["z"]
    marker.pose.orientation.w = table_pose["orientation"]["w"]

    rospy.loginfo("Table visualization is active.")
    while not rospy.is_shutdown():
        pub.publish(marker)
        marker.header.stamp = rospy.Time.now()
        rate.sleep()