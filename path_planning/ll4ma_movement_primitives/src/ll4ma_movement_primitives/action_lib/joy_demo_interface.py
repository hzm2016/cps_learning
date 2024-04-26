#!/usr/bin/env python
import rospy
import Tkinter as tk
from trajectory_action_lib import JointTrajectoryActionClient
from sensor_msgs.msg import Joy
from graphical_user_interface import DemonstrationDialog
from std_srvs.srv import Trigger
from ll4ma_logger.srv import FileAction, FileActionRequest
from ll4ma_policy_learning.srv import (
    AddProMPDemo,
    AddProMPDemoRequest,
    SetTaskLabel,
    SetTaskLabelRequest,
)
from rospy_service_helper import (
    set_controller_mode, set_move_to_start, set_rosbag_status,
    zero_reflex_tactile, command_reflex_hand, get_task_trajectories,
    get_joint_trajectories, learn_weights, SRV_NAMES)

_PRESSED = 1
_RIGHT_TRIGGER = 5
_LEFT_RIGHT_ARROW = 6
_RIGHT_ARROW = -1


class JoystickDemonstrationInterface:
    """
    This class creates a state machine for button presses on the Xbox controller
    used for experiment data collection.
    """

    def __init__(self):
        rospy.Subscriber("/joy", Joy, self._joy_cb, queue_size=1)
        self.demo_path = rospy.get_param("~demo_path")
        self.num_bfs = rospy.get_param("~num_bfs", 6)
        self.use_lift = rospy.get_param("~use_lift", False)
        self.use_reflex = rospy.get_param("/use_reflex", False)
        self.learn = False
        self.offline_mode = rospy.get_param("/offline_mode", True)
        if not self.offline_mode:
            self._joint_traj_client = JointTrajectoryActionClient(
                rospy_init=False)
        self.rate = rospy.Rate(100)
        self._btn_map = [
            "green_A", "red_B", "blue_X", "yellow_Y", "left_bumper",
            "right_bumper", "window", "menu", "power"
        ]
        self.srvs = {}
        self.srvs["set_grav_comp_mode"] = SRV_NAMES["set_grav_comp_mode"]
        self.srvs["set_krc_mode"] = SRV_NAMES["set_krc_mode"]
        self.srvs["set_move_to_start"] = SRV_NAMES["set_move_to_start"]
        self.srvs["set_recording"] = SRV_NAMES["set_recording"]
        self.srvs["del_latest_rosbag"] = SRV_NAMES["delete_latest_rosbag"]
        self.srvs["zero_reflex_tactile"] = SRV_NAMES["zero_reflex_tactile"]
        self.srvs["calib_fingers"] = SRV_NAMES["calibrate_reflex_fingers"]
        self.srvs["open_reflex_hand"] = SRV_NAMES["open_reflex_hand"]
        self.srvs["grasp_reflex"] = SRV_NAMES["grasp_reflex"]
        self.srvs["get_traj_name"] = SRV_NAMES["get_traj_name"]
        self.srvs["register_filename"] = SRV_NAMES["register_filename"]
        self.srvs["get_task_trajs"] = SRV_NAMES["get_task_trajs"]
        self.srvs["get_joint_trajs"] = SRV_NAMES["get_joint_trajs"]
        self.srvs["learn_weights"] = SRV_NAMES["learn_weights"]
        self.srvs["add_demo"] = SRV_NAMES["add_demo"]
        self.srvs["discard_instance"] = SRV_NAMES["discard_instance"]
        self.srvs["add_negative_inst"] = SRV_NAMES["add_negative_inst"]
        self.srvs["set_label"] = SRV_NAMES["set_label"]
        self.srvs["record_ee_in_obj"] = SRV_NAMES["record_ee_in_obj"]
        self.srvs["lift_object"] = SRV_NAMES["lift_object"]
        self.srvs["set_current_obj_pose"] = SRV_NAMES["set_current_obj_pose"]
        self.srvs["gen_task_instance"] = SRV_NAMES["gen_task_instance"]

        self.mts_set = False
        self.demo_active = False
        self.trigger_down = False
        self.demo_given = True  # Set to true initially so first instance is generated

    def run(self):
        rospy.loginfo("Listening for button presses...")
        while not rospy.is_shutdown():
            self.rate.sleep()

    def exit(self):
        rospy.loginfo("Exiting.")

    def _joy_cb(self, joy_msg):
        """
        Callback function for Xbox controller commands.
        """
        if joy_msg.buttons.count(_PRESSED) > 1:
            rospy.logwarn(
                "Only one button should be pressed at a time. No action.")
        elif joy_msg.buttons.count(_PRESSED) == 1:
            pressed_btn = self._btn_map[list(joy_msg.buttons).index(_PRESSED)]
            if pressed_btn == "green_A":
                self._handle_demo()
            elif pressed_btn == "red_B":
                self._abort_demo()
            elif pressed_btn == "blue_X":
                pass
            elif pressed_btn == "yellow_Y":
                pass
            elif pressed_btn == "window":
                self._move_to_start()
            elif pressed_btn == "left_bumper":
                self._open_hand()
            elif pressed_btn == "right_bumper":
                self._open_demo_dialog()
            else:
                rospy.logwarn("Button unassigned")
        elif not self.trigger_down and joy_msg.axes[_RIGHT_TRIGGER] == -1.0:
            # Enable gravity compensation mode
            if set_controller_mode(self.srvs["set_grav_comp_mode"]):
                rospy.loginfo(
                    "Gravity compensation mode successfully enabled.")
                self.trigger_down = True
        elif self.trigger_down and joy_msg.axes[_RIGHT_TRIGGER] > -1.0:
            # Disable gravity compensation mode
            if set_controller_mode(self.srvs["set_krc_mode"]):
                rospy.loginfo(
                    "Gravity compensation mode successfully disabled.")
            self.trigger_down = False
        elif joy_msg.axes[_LEFT_RIGHT_ARROW] == _RIGHT_ARROW:
            self._generate_instance()

    def _handle_demo(self):
        """
        Start/stop demonstration. For starting, will zero tactile sensors,
        enable gravity compensation, and start rosbag recorder. For stopping,
        will disable gravity compensation and stop rosbag recorder.
        """
        if self.demo_active:
            # Disable gravity compensation mode
            if set_controller_mode(self.srvs["set_krc_mode"]):
                rospy.loginfo("Gravity compensation successfully disabled.")
                self.demo_active = False
            else:
                rospy.logerr("Gravity compensation could not be disabled.")
            # Stop rosbag recorder
            set_rosbag_status(
                self.srvs["set_recording"], self.demo_path, recording=False)

            # Record the end-effector pose in the object frame
            try:
                record_ee_in_obj = rospy.ServiceProxy(
                    self.srvs["record_ee_in_obj"], Trigger)
                success = record_ee_in_obj()
            except rospy.ServiceException as e:
                rospy.logwarn("Service call to record EE pose failed: %s" % e)
                return False
            if not success:
                rospy.logwarn("Could not record EE pose.")
                return False

            self.demo_given = True

            if self.use_lift:
                # Lift the object
                try:
                    lift = rospy.ServiceProxy(self.srvs["lift_object"],
                                              Trigger)
                    lift()
                except rospy.ServiceException as e:
                    rospy.logerr("Could not execute lift: {}".format(e))
            rospy.loginfo("Demonstration complete.")
        else:
            # Zero the reflex sensors
            if self.use_reflex:
                zero_reflex_tactile(self.srvs["zero_reflex_tactile"])
            # Set the current object pose on the current task instance
            try:
                set_pose = rospy.ServiceProxy(
                    self.srvs["set_current_obj_pose"], Trigger)
                set_pose()
            except rospy.ServiceException as e:
                rospy.logwarn(
                    "Service call to set object pose failed: {}".format(e))
            # Enable gravity compensation mode
            if set_controller_mode(self.srvs["set_grav_comp_mode"]):
                rospy.loginfo("Gravity compensation successfully enabled.")
                self.demo_active = True
            else:
                rospy.logerr("Gravity compensation could not be enabled.")
            # Start rosbag recorder
            if self.demo_active:
                set_rosbag_status(
                    self.srvs["set_recording"], self.demo_path, recording=True)

    def _abort_demo(self):
        """
        Put controller back in idle mode and open dialog.
        """
        rospy.logwarn("Aborting demonstration.")
        # Disable gravity compensation mode
        if set_controller_mode(self.srvs["set_krc_mode"]):
            rospy.loginfo("Gravity compensation successfully disabled.")
            self.demo_active = False
        else:
            rospy.logerr("Gravity compensation could not be disabled.")
        # Stop rosbag recorder
        set_rosbag_status(
            self.srvs["set_recording"], self.demo_path, recording=False)
        self._open_demo_dialog()

    def _open_demo_dialog(self):
        """
        Open demo dialog so user can choose to discard it or mark infeasible.
        """
        root = tk.Tk()
        root.withdraw()
        dialog = DemonstrationDialog(
            root, self.demo_path, self.num_bfs, learn=self.learn)
        root.wait_window(dialog.top)

    def _open_hand(self):
        """
        Open reflex hand.
        """
        rospy.loginfo("Opening hand...")
        command_reflex_hand(self.srvs["open_reflex_hand"])
        rospy.loginfo("Hand opened.")

    def _move_to_start(self):
        """
        Execute the Move-To-Start trajectory going from the current robot 
        position to the nominal starting position.
        """
        if self.trigger_down:
            rospy.logwarn("Gravity compensation is enabled.")
            return False
        if self.demo_active:
            rospy.logwarn("Demonstration is active.")
            return False

        if self.mts_set:
            rospy.loginfo("Moving robot to start position.")
            self._joint_traj_client.send_goal(wait_for_result=False)
            self.mts_set = False
        else:
            rospy.loginfo("Ready to move to start.")
            self.mts_set = True

    def _generate_instance(self):
        """
        Generate the next task instance. Intended for finite data set collection
        where no review is necessary.
        """
        if self.demo_given:
            success = False
            try:
                gen = rospy.ServiceProxy(self.srvs["gen_task_instance"],
                                         Trigger)
                resp = gen()
                success = resp.success
            except rospy.ServiceException as e:
                rospy.logerr(
                    "Service call to generate next task instance failed: {}".
                    format(e))
                return False
            if not success:
                rospy.logwarn("Could not generate task instance: {}".format(
                    resp.message))
                return False
            self.demo_given = False
        else:
            rospy.logwarn(
                "Instance is already generated! Demo the current instance.")


if __name__ == '__main__':
    rospy.init_node("joystick_demo_interface")
    joy = JoystickDemonstrationInterface()
    rospy.on_shutdown(joy.exit)
    joy.run()
