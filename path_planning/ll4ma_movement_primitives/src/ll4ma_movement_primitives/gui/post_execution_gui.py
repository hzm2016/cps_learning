import rospy
import Tkinter as tk
import tkFont
from std_srvs.srv import Trigger
from ll4ma_rosbag_utils.srv import RosbagAction, RosbagActionRequest
from ll4ma_movement_primitives.srv import SetTaskLabel, SetTaskLabelRequest
from rospy_service_helper import SRV_NAMES)


class PostExecutionDialog:
    """
    Dialog popup that requests user's feedback on what to do with the most
    recently executed task instance (mark success, mark infeasible, or discard)

    The dialog is intended to be created from a main window like the
    PolicyExecutionGUI. You can view it by running this file as a script:

        $ python post_execution_gui.py
    """
    
    def __init__(self, parent, path, executed_promp_name):
        self.srvs = {
            "set_label": SRV_NAMES["set_label"],
            "del_latest_rosbag": SRV_NAMES["delete_latest_rosbag"],
            "add_negative_inst": SRV_NAMES["add_negative_inst"]
        }
        self.path = path
        self.executed_promp_name = executed_promp_name
        top = self.top = tk.Toplevel(parent, bg="blue", padx=10, pady=10)
        top.title("Execution Result")
        btn_font = tkFont.Font(family='Helvetica', size=20, weight='bold')
        success = tk.Button(top, text="Keep", command=self.mark_success)
        success['font'] = btn_font
        success.pack(side=tk.LEFT, pady=5, padx=5)
        discard = tk.Button(top, text="Discard", command=self.discard)
        discard['font'] = btn_font
        discard.pack(side=tk.LEFT, pady=5, padx=5)
        infeasible = tk.Button(top, text="Mark Infeasible", command=self.mark_infeasible)
        infeasible['font'] = btn_font
        infeasible.pack(side=tk.LEFT, pady=5, padx=5)

    def mark_success(self):
        """
        Mark the task instance as successful.
        """
        # Set the success label on the action server for the recently generated pose
        req = SetTaskLabelRequest()
        req.label = 1
        try:
            set_label = rospy.ServiceProxy(self.srvs["set_label"], SetTaskLabel)
            set_label(req)
        except rospy.ServiceException as e:
            rospy.logwarn("Service call to set task label failed %s" % e)
            self.top.destroy()
            return False

        self.top.destroy()
        return True

    def discard(self):
        """        
        Get rid of the rosbag that was just recorded.
        """
        try:
            discard = rospy.ServiceProxy(self.srvs["del_latest_rosbag"], RosbagAction)
            discard(RosbagActionRequest(path=self.path))
        except rospy.ServiceException as e:
            rospy.logwarn("Service call to delete rosbag failed %s" % e)
        self.top.destroy()

    def mark_infeasible(self):
        """
        Set the label for task instance as infeasible.
        """
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

        # Discard the recorded rosbag
        self.discard()

        self.top.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    dialog = PostExecutionDialog(root, "", "")
    root.mainloop()
