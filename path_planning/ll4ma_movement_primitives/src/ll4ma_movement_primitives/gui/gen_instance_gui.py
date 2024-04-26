import rospy
import Tkinter as tk
import tkFont
from std_srvs.srv import Trigger
from ll4ma_movement_primitives.srv import SetTaskLabel, SetTaskLabelRequest
from rospy_service_helper import SRV_NAMES


class GenerateTaskInstanceDialog:
    """
    Dialog popup that requests the user's feedback on how what to do with
    the generated task instance (keep it, mark infeasible, or discard it).

    The dialog is intended to be created from a main window like the
    PolicyExecutionGUI. You can view it by running this file as a script:

        $ python gen_instance_gui.py
    """
    
    def __init__(self, parent):
        self.srvs = {
            "set_label": SRV_NAMES["set_label"],
            "add_negative_inst": SRV_NAMES["add_negative_inst"],
            "discard_instance": SRV_NAMES["discard_instance"]
        }
        top = self.top = tk.Toplevel(parent, bg="blue", padx=10, pady=10)
        top.title("Task Instance")
        btn_font = tkFont.Font(family='Helvetica', size=20, weight='bold')
        keep = tk.Button(top, text="Keep", command=self.keep)
        keep['font'] = btn_font
        keep.pack(side=tk.LEFT, pady=5, padx=5)
        discard = tk.Button(top, text="Discard", command=self.discard)
        discard['font'] = btn_font
        discard.pack(side=tk.LEFT, pady=5, padx=5)
        infeasible = tk.Button(top, text="Mark Infeasible", command=self.mark_infeasible)
        infeasible['font'] = btn_font
        infeasible.pack(side=tk.LEFT, pady=5, padx=5)

    def keep(self):
        """
        No functionality, just close dialog so user can give demo.
        """
        self.top.destroy()

    def mark_infeasible(self):
        """
        Set the label for task instance as infeasible.
        """
        req = SetTaskLabelRequest(label=0)
        try:
            set_label = rospy.ServiceProxy(self.srvs["set_label"], SetTaskLabel)
            set_label(req)
        except rospy.ServiceException as e:
            rospy.logwarn("Service call to set task label failed: %s" % e)

        # Add the instance as a negative instance to the GMM
        try:
            add_gmm_inst = rospy.ServiceProxy(self.srvs["add_negative_inst"], Trigger)
            add_gmm_inst()
        except rospy.ServiceException as e:
            rospy.logwarn("Service call to add negative instance to GMM failed: %s" % e)
        self.top.destroy()

    def discard(self):
        """
        Get rid of the task instance that was just generated.
        """
        try:
            discard = rospy.ServiceProxy(self.srvs["discard_instance"], Trigger)
            discard()
        except rospy.ServiceException as e:
            rospy.logwarn("Service call to discard task instance failed: {}".format(e))
        self.top.destroy()

        
if __name__ == '__main__':
    root = tk.Tk()
    dialog = GenerateTaskInstanceDialog(root)
    root.mainloop()
