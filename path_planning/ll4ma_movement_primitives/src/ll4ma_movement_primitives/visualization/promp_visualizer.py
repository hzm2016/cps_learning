#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import ColorRGBA
from matplotlib import colors
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped, PoseArray
from sensor_msgs.msg import JointState
from policy_interface import PolicyLearner
from ll4ma_movement_primitives.promps import (
    ProMP, ProMPConfig, Waypoint, ros_to_python_config, ros_to_python_waypoint)
from ll4ma_policy_learning.srv import VisualizeProMPs, VisualizeProMPsResponse
from ll4ma_trajectory_util.srv import VisualizePoses, VisualizePosesResponse
from std_srvs.srv import Trigger, TriggerResponse


class ProMPVisualizer:
    """
    Visualizes ProMP samples using rviz markers.

    Provides services for generating rviz visualizations of ProMP samples.
    The service calls consume ProMP configurations that specify the policy
    parameters and generate ProMP trajectory samples that are visualized
    in rviz using marker arrays.
    """
    
    def __init__(self):
        self.base_frame = rospy.get_param("~base_frame")
        self.marker_topic = rospy.get_param("~marker_topic", "promp_viz")
        self.promp_marker_array = MarkerArray()
        self.rate = rospy.Rate(1)
        self.promp_marker_pub = rospy.Publisher(
            self.marker_topic, MarkerArray, queue_size=1)
        self.pose_pub = rospy.Publisher(
            "/object_poses", PoseArray, queue_size=1)
        self.pose_array = PoseArray()
        self.pose_array.header.frame_id = self.base_frame
        self.idx = 0

        # Services this node offers
        self.viz_promp_srv = rospy.Service("/visualization/visualize_promps",
                                           VisualizeProMPs,
                                           self.visualize_promps)
        self.viz_pose_srv = rospy.Service("/visualization/visualize_poses",
                                          VisualizePoses, self.visualize_poses)
        self.clear_pose_srv = rospy.Service("/visualization/clear_poses",
                                            Trigger, self.clear_poses)
        self.clear_samples_srv = rospy.Service(
            "/visualization/clear_promp_samples", Trigger, self.clear_samples)

    def run(self):
        rospy.loginfo("ProMP visualization services are available.")
        while not rospy.is_shutdown():
            if self.promp_marker_array.markers:
                self.promp_marker_pub.publish(self.promp_marker_array)
            self.rate.sleep()

    # === BEGIN Service functions being offered ===============================

    def visualize_promps(self, req):
        rospy.loginfo("Generating ProMP visualization...")
        resp = VisualizeProMPsResponse()
        resp.success = self._visualize_samples(
            req.promp_config, req.waypoints, req.num_samples, req.duration,
            req.dt, req.color_name, req.clear)
        rospy.loginfo("Visualization complete.")
        return resp

    def visualize_poses(self, req):
        resp = VisualizePosesResponse()
        resp.success = self._visualize_poses(req.poses, req.base_link)
        return resp

    def clear_poses(self, req):
        self.pose_array.poses = []
        resp = TriggerResponse()
        resp.success = not self.pose_array.poses
        return resp

    def clear_samples(self, req):
        self.promp_marker_array = MarkerArray()
        resp = TriggerResponse()
        resp.success = True
        return resp

    # === END Service functions being offered ==================================

    def _visualize_samples(self,
                           ros_config,
                           ros_waypoints=[],
                           num_samples=10,
                           duration=10.0,
                           dt=0.1,
                           color_name=None,
                           clear=False):
        if not color_name:
            color_name = "dodgerblue"
        if clear:
            self.promp_marker_array = MarkerArray()
            self.idx = 0

        # Learn ProMPs
        promp_config = ros_to_python_config(ros_config)
        promp = ProMP(config=promp_config)
        waypoints = [ros_to_python_waypoint(w) for w in ros_waypoints]

        # Create the markers
        for _ in range(num_samples):
            traj = promp.generate_trajectory(
                duration=duration, dt=dt, waypoints=waypoints)
            marker = self._get_marker(traj, color_name, 0.004, self.idx)
            self.promp_marker_array.markers.append(marker)
            self.idx += 1
        # Get one for the mean
        traj = promp.generate_trajectory(
            duration=duration, dt=dt, mean=True, waypoints=waypoints)
        color_name = "navy"
        marker = self._get_marker(traj, color_name, 0.02, self.idx)
        self.promp_marker_array.markers.append(marker)

        rospy.loginfo("ProMP visualization is active.")
        return True

    def _visualize_poses(self, poses, base_link):
        self.pose_array.poses = []
        self.pose_array.header.frame_id = base_link
        for pose in poses:
            self.pose_array.poses.append(pose)
            self.pose_array.header.stamp = rospy.Time()
        # Publish multiple times to make sure it registers
        self.pose_pub.publish(self.pose_array)
        rospy.sleep(0.5)
        self.pose_pub.publish(self.pose_array)
        return True

    def _get_marker(self, traj, color_name, size=0.004, idx=0):
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = rospy.Time.now()
        m.id = idx
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = size
        m.color = self._get_color(color_name)
        for j in range(len(traj['x'][0])):
            m.points.append(
                Point(traj['x'][0][j], traj['x'][1][j], traj['x'][2][j]))
        return m

    def _get_color(self, color_name):
        """
        color can be any name from this page:
        http://matplotlib.org/mpl_examples/color/named_colors.hires.png
        """
        converter = colors.ColorConverter()
        c = converter.to_rgba(colors.cnames[color_name])
        return ColorRGBA(*c)


if __name__ == '__main__':
    rospy.init_node("promp_sample_visualizer")
    visualizer = ProMPVisualizer()
    visualizer.run()
