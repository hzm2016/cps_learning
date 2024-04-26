import tkFont
import Tkinter as tk
import rospy
from std_srvs.srv import Trigger
from ll4ma_rosbag_utils.srv import  RosbagAction, RosbagActionRequest
from ll4ma_movement_primitives.srv import AddProMPDemo, AddProMPDemoRequest
from ll4ma_movement_primitives.srv import SetTaskLabel, SetTaskLabelRequest
from rospy_service_helper import (SRV_NAMES, get_joint_trajectories,
                                  get_task_trajectories, learn_weights)


class DemonstrationDialog:
    def __init__(self, parent, path, num_bfs, learn=True):
        self.srvs = {
            "set_label": SRV_NAMES["set_label"],
            "get_traj_name": SRV_NAMES["get_traj_name"],
            "register_filename": SRV_NAMES["register_filename"],
            "get_task_trajs": SRV_NAMES["get_task_trajs"],
            "get_joint_trajs": SRV_NAMES["get_joint_trajs"],
            "add_demo": SRV_NAMES["add_demo"],
            "del_latest_rosbag": SRV_NAMES["delete_latest_rosbag"],
            "add_negative_inst": SRV_NAMES["add_negative_inst"],
            "learn_weights": SRV_NAMES["learn_weights"]
        }
        self.path = path
        self.num_bfs = num_bfs
        self.learn = learn
        top = self.top = tk.Toplevel(parent, bg="blue", padx=10, pady=10)
        top.title("Demonstration")
        btn_font = tkFont.Font(family='Helvetica', size=20, weight='bold')
        add = tk.Button(top, text="Add Demo", command=self.add_demo)
        add['font'] = btn_font
        add.pack(side=tk.LEFT, pady=5, padx=5)
        discard = tk.Button(top, text="Discard Demo", command=self.discard_demo)
        discard['font'] = btn_font
        discard.pack(side=tk.LEFT, pady=5, padx=5)
        infeasible = tk.Button(top, text="Mark Infeasible", command=self.mark_infeasible)
        infeasible['font'] = btn_font
        infeasible.pack(side=tk.LEFT, pady=5, padx=5)

    def add_demo(self):
        # Set the success label on the action server for the recently generated pose
        try:
            set_label = rospy.ServiceProxy(self.srvs["set_label"], SetTaskLabel)
            set_label(SetTaskLabelRequest(label=1))
        except rospy.ServiceException as e:
            rospy.logwarn("Service call to set task label failed %s" % e)
            self.top.destroy()
            return False

        # Get latest trajectory name (the one that was just recorded)
        try:
            get_name = rospy.ServiceProxy(self.srvs["get_traj_name"], RosbagAction)
            resp = get_name(RosbagActionRequest())
            traj_name = resp.filename
        except rospy.ServiceException as e:
            rospy.logwarn("Service call to get latest trajectory name failed: %s" % e)
            self.top.destroy()
            return False

        # Register the filename with data server so the data can be requested
        try:
            register = rospy.ServiceProxy(self.srvs["register_filename"], RosbagAction)
            req = RosbagActionRequest()
            req.filename = traj_name
            resp = register(req)
        except rospy.ServiceException as e:
            rospy.logwarn("Service call to register filename failed: %s" % e)
            self.top.destroy()
            return False

        if self.learn:
            # Get trajectory data
            ee_trajs = get_task_trajectories(self.srvs["get_task_trajs"],
                                             "end_effector_pose_base_frame", [traj_name])
            j_traj = get_joint_trajectories(self.srvs["get_joint_trajs"], "lbr4", [traj_name])[0]

            # Learn ProMP weights
            w, config = learn_weights(self.srvs["learn_weights"], ee_trajs=ee_trajs,
                                      num_bfs=self.num_bfs)
            config.init_joint_state = j_traj.points[0].positions

            # Create request to Incorporate this new demo into the ProMP library
            req = AddProMPDemoRequest()
            req.w = w
            req.config = config
            req.data_name = traj_name

            # Incorporate the new demo into the ProMP library
            try:
                add_demo = rospy.ServiceProxy(self.srvs["add_demo"], AddProMPDemo)
                add_demo(req)
            except rospy.ServiceException as e:
                rospy.logwarn("Service call to add ProMP demo failed: %s" % e)
            self.top.destroy()
            return True
        else:
            # Create request that will just save appropriate data regarding the
            # demonstration (i.e. doesn't actually try to learn a ProMP)
            req = AddProMPDemoRequest()
            req.data_name = traj_name
            try:
                save = rospy.ServiceProxy(self.srvs["add_demo"], AddProMPDemo)
                save(req)
            except rospy.ServiceException as e:
                rospy.logwarn("Service call to add demo failed: %s" % e)
            self.top.destroy()
            return True

    def discard_demo(self):
        # Get rid of the rosbag that was just recorded
        try:
            discard = rospy.ServiceProxy(self.srvs["del_latest_rosbag"], RosbagAction)
            discard(RosbagActionRequest(path=self.path))
        except rospy.ServiceException as e:
            rospy.logwarn("Service call to delete rosbag failed %s" % e)
        self.top.destroy()

    def mark_infeasible(self):
        # Set the label as infeasible on the action server for the recently generated pose
        try:
            set_label = rospy.ServiceProxy(self.srvs["set_label"], SetTaskLabel)
            set_label(SetTaskLabelRequest(label=0))
        except rospy.ServiceException as e:
            rospy.logwarn("Service call to set task label failed: %s" % e)

        # Add the instance as a negative instance to the GMM
        try:
            add_gmm_inst = rospy.ServiceProxy(self.srvs["add_negative_inst"], Trigger)
            add_gmm_inst()
        except rospy.ServiceException as e:
            rospy.logwarn("Service call to add negative instance to GMM failed: %s" % e)
        self.top.destroy()

        # Discard the recorded demo
        self.discard_demo()


if __name__ == '__main__':
    root = tk.Tk()
    dialog = DemonstrationDialog(root, "", 0)
    root.mainloop()
