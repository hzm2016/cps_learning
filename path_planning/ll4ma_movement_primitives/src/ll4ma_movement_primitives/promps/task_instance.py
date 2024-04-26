import rospy
from geometry_msgs.msg import Pose
from ll4ma_movement_primitives.msg import PlanarPose


class TaskInstance(object):
    """
    This class represents a task instance which characterizes the essential 
    components and attributes relevant to a task to be performed. Used primarily 
    for ProMP conditioning and sampling in active learning.

    object_planar_pose and init_pose_guess are PlanarPose instances
    """

    def __init__(self, **kwargs):
        self.object_planar_pose = kwargs.get("object_planar_pose",
                                             PlanarPose())
        self.table_pose = kwargs.get("table_pose", Pose())
        self.ee_pose_in_obj = kwargs.get("ee_pose_in_obj", Pose())
        self.grid_coords = kwargs.get("grid_coords", (-1, -1, -1))
        self.label = kwargs.get("label", -100)
        self.trajectory_name = kwargs.get("trajectory_name", "UNASSIGNED")

    def has_label(self):
        return self.label != -100

    def __str__(self):
        output = "\n\n{s:{c}^{n}}\n\n".format(s=" Task Instance ", c='=', n=90)
        output += "{:4}{s:<24}{t}\n".format(
            "", s="Trajectory Name:", t=self.trajectory_name)
        output += "{:4}{s:<24}{t}\n".format("", s="Label:", t=self.label)
        output += "{:4}{s:<24}\n".format("", s="Planar Pose:")
        output += "{:8}{s:<24}\n".format(
            "", s="x: {x}".format(x=self.object_planar_pose.x))
        output += "{:8}{s:<24}\n".format(
            "", s="y: {y}".format(y=self.object_planar_pose.y))
        output += "{:8}{s:<24}\n".format(
            "", s="z: {z}".format(z=self.object_planar_pose.z))
        output += "{:8}{s:<24}\n".format(
            "", s="theta: {theta}".format(theta=self.object_planar_pose.theta))
        output += "{:4}{s:<24}\n".format("", s="Table Pose:")
        output += "{:8}{s:<24}\n".format("", s="Position:")
        output += "{:12}{s:<24}\n".format(
            "", s="x: {x}".format(x=self.table_pose.position.x))
        output += "{:12}{s:<24}\n".format(
            "", s="y: {y}".format(y=self.table_pose.position.y))
        output += "{:12}{s:<24}\n".format(
            "", s="z: {z}".format(z=self.table_pose.position.z))
        output += "{:8}{s:<24}\n".format("", s="Orientation:")
        output += "{:12}{s:<24}\n".format(
            "", s="x: {x}".format(x=self.table_pose.orientation.x))
        output += "{:12}{s:<24}\n".format(
            "", s="y: {y}".format(y=self.table_pose.orientation.y))
        output += "{:12}{s:<24}\n".format(
            "", s="z: {z}".format(z=self.table_pose.orientation.z))
        output += "{:12}{s:<24}\n".format(
            "", s="w: {w}".format(w=self.table_pose.orientation.w))
        output += "{:4}{s:<24}\n".format(
            "", s="End-Effectory Pose (Object Frame):")
        output += "{:8}{s:<24}\n".format("", s="Position:")
        output += "{:12}{s:<24}\n".format(
            "", s="x: {x}".format(x=self.ee_pose_in_obj.position.x))
        output += "{:12}{s:<24}\n".format(
            "", s="y: {y}".format(y=self.ee_pose_in_obj.position.y))
        output += "{:12}{s:<24}\n".format(
            "", s="z: {z}".format(z=self.ee_pose_in_obj.position.z))
        output += "{:8}{s:<24}\n".format("", s="Orientation:")
        output += "{:12}{s:<24}\n".format(
            "", s="x: {x}".format(x=self.ee_pose_in_obj.orientation.x))
        output += "{:12}{s:<24}\n".format(
            "", s="y: {y}".format(y=self.ee_pose_in_obj.orientation.y))
        output += "{:12}{s:<24}\n".format(
            "", s="z: {z}".format(z=self.ee_pose_in_obj.orientation.z))
        output += "{:12}{s:<24}\n".format(
            "", s="w: {w}".format(w=self.ee_pose_in_obj.orientation.w))
        output += "{:4}{s:<24}({i}, {j}, {k})\n".format(
            "",
            s="Grid Coordinates:",
            i=self.grid_coords[0],
            j=self.grid_coords[1],
            k=self.grid_coords[2])
        return output


if __name__ == '__main__':
    instance = TaskInstance()
    print instance
