#!/usr/bin/env python
import os
import sys
import yaml
import rospy
import rospkg

if __name__ == '__main__':
    """
    Script to update the table to base calibrations in local configs.
    """
    rospy.init_node("update_config_calibration")

    base_link = rospy.get_param("~base_link", "lbr4_base_link")
    experiment_type = rospy.get_param("~experiment_type", "experiment_1")
    table_1_link = rospy.get_param("~table_1_link", "table_1_center")
    table_2_link = rospy.get_param("~table_2_link", "table_2_center")
    table_3_link = rospy.get_param("~table_3_link", "table_3_center")

    table_1_key = "{}__to__{}".format(base_link, table_1_link)
    table_2_key = "{}__to__{}".format(base_link, table_2_link)
    table_3_key = "{}__to__{}".format(base_link, table_3_link)

    rospack = rospkg.RosPack()
    path = rospack.get_path("robot_aruco_calibration")
    filename = os.path.join(path, "config/robot_camera_calibration.yaml")
    with open(filename, 'r') as f:
        calib_data = yaml.load(f)

    path = rospack.get_path("ll4ma_policy_learning")
    if experiment_type == "experiment_1":
        filename = os.path.join(path, "config/experiment_1.yaml")
    elif experiment_type == "experiment_2":
        filename = os.path.join(path, "config/experiment_2.yaml")
    elif experiment_type == "experiment_3":
        filename = os.path.join(path, "config/experiment_3.yaml")
    else:
        rospy.logwarn("Unknown experiment type: {}".format(experiment_type))
        sys.exit(1)

    with open(filename, 'r') as f:
        experiment_data = yaml.load(f)

    if table_1_key in calib_data.keys():
        experiment_data[table_1_key] = calib_data[table_1_key]
    if table_2_key in calib_data.keys():
        experiment_data[table_2_key] = calib_data[table_2_key]
    if table_3_key in calib_data.keys():
        experiment_data[table_3_key] = calib_data[table_3_key]

    with open(filename, 'w') as f:
        yaml.dump(experiment_data, f, default_flow_style=False)
