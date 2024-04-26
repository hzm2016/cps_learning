import rospy
import Tkinter as tk
import tkFont
from std_srvs.srv import Trigger
from ll4ma_rosbag_utils.srv import RosbagAction, RosbagActionRequest
from ll4ma_movement_primitives.srv import SetTaskLabel, SetTaskLabelRequest
from rospy_service_helper import set_review_status, SRV_NAMES


class PreviewExecutionDialog:
    """
    Dialog popup that requests user's feedback on what to do with the most
    recently previewed task instance (approve, deny, or mark infeasible).

    The dialog is intended to be created from a main window like the
    PolicyExecutionGUI. You can view it by running this file as a script:

        $ python preview_execution_gui.py
    """
    
    def __init__(self, parent):
        self.srvs = {"set_policy_rev_stat": SRV_NAMES["set_policy_rev_stat"]}
        top = self.top = tk.Toplevel(parent, bg="blue", padx=10, pady=10)
        top.title("Execution Preview")
        btn_font = tkFont.Font(family='Helvetica', size=20, weight='bold')
        approve = tk.Button(top, text="Approve", command=self.approve)
        approve['font'] = btn_font
        approve.pack(side=tk.LEFT, pady=5, padx=5)
        deny = tk.Button(top, text="Deny", command=self.deny)
        deny['font'] = btn_font
        deny.pack(side=tk.LEFT, pady=5, padx=5)
        mark_infeasible = tk.Button(top, text="Mark Infeasible", command=self.mark_infeasible)
        mark_infeasible['font'] = btn_font
        mark_infeasible.pack(side=tk.LEFT, pady=5, padx=5)

    def approve(self):
        """
        Set review status to True on the action server.
        """
        set_review_status(self.srvs["set_policy_rev_stat"], True)
        self.top.destroy()

    def deny(self):
        """
        Don't need to do anything except close dialog.
        """
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
        self.top.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    dialog = PreviewExecutionDialog(root)
    root.mainloop()
