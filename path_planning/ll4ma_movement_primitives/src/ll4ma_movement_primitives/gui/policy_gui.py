#!/usr/bin/env python
import os
import rospy
import numpy as np
import Tkinter as tk
import tkFont
import tkMessageBox
from matplotlib import colors
from trajectory_action_lib import JointTrajectoryActionClient
from ll4ma_trajectory_msgs.srv import RobotTrajectory, RobotTrajectoryRequest
from std_srvs.srv import Trigger
from std_msgs.msg import ColorRGBA
from ll4ma_movement_primitives.action_lib import TaskProMPActionClient
from ll4ma_movement_primitives.gui import PostExecutionDialog, PreviewExecutionDialog
from ll4ma_movement_primitives.gui import GenerateTaskInstanceDialog, DemonstrationDialog
from ll4ma_logger_msgs.srv import SaveTrajectoryExecution, SaveTrajectoryExecutionRequest
from rospy_service_helper import (
    set_controller_mode, zero_reflex_tactile, calibrate_reflex_fingers,
    command_reflex_hand, set_rosbag_status, review_trajectory,
    set_move_to_start, clear_all, get_task_trajectories,
    get_joint_trajectories, set_review_status, SRV_NAMES)

end_poses = []


class PolicyExecutionGUI:
    """
    Main GUI for a ProMP active learning session.

    This GUI offers the following functionality:
      1. Start/Stop recording of a demonstration
      2. Enable/Disable gravity compensation mode
      3. Move robot back to nominal starting position
      4. Generate a task instance for active learning
      5. Preview policy execution
      6. Execute a policy once previewed

    This is intended to be run either via rosrun or as a node in a launch file.
    You can also view the GUI in debug mode with this command:

        $ python policy_gui.py --debug
    """
    
    def __init__(self, debug=False):
        if debug:
            self._create_board()
            self.run()
            rospy.signal_shutdown('Quit')
        self.demo_path = rospy.get_param("~demo_path", "~/.rosbags/demos")
        self.offline_mode = rospy.get_param("/offline_mode", True)
        self.use_reflex = rospy.get_param("/use_reflex", False)
        self.use_pandas = rospy.get_param("/use_pandas", True)
        self.recording = False
        if not self.offline_mode:
            self._policy_client = TaskProMPActionClient(rospy_init=False)
            self._traj_client = JointTrajectoryActionClient(rospy_init=False)

        # Services this node will call
        self.srvs = {
            "preview_policy": SRV_NAMES["preview_policy"],
            "clear_all_policy": SRV_NAMES["clear_all_policy"],
            "add_demo": SRV_NAMES["add_demo"],
            "gen_task_instance": SRV_NAMES["gen_task_instance"],
            "set_label": SRV_NAMES["set_label"],
            "discard_instance": SRV_NAMES["discard_instance"],
            "add_negative_inst": SRV_NAMES["add_negative_inst"],
            "set_policy_rev_stat": SRV_NAMES["set_policy_rev_stat"],
            "clear_viz": SRV_NAMES["clear_viz"],
            "learn_weights": SRV_NAMES["learn_weights"],
        }
        if not self.offline_mode:
            self.srvs["set_grav_comp_mode"] = SRV_NAMES["set_grav_comp_mode"]
            self.srvs["set_krc_mode"] = SRV_NAMES["set_krc_mode"]
            self.srvs["set_recording"] = SRV_NAMES["set_recording"]
            self.srvs["del_latest_rosbag"] = SRV_NAMES["delete_latest_rosbag"]
            self.srvs["record_ee_in_obj"] = SRV_NAMES["record_ee_in_obj"]
        if self.use_reflex:
            self.srvs["zero_reflex_tactile"] = SRV_NAMES["zero_reflex_tactile"]
            self.srvs["calib_fingers"] = SRV_NAMES["calibrate_reflex_fingers"]
            self.srvs["open_reflex_hand"] = SRV_NAMES["open_reflex_hand"]
            self.srvs["grasp_reflex"] = SRV_NAMES["grasp_reflex"]
        if self.use_pandas:
            self.srvs["get_traj_name"] = SRV_NAMES["get_traj_name"]
            self.srvs["register_filename"] = SRV_NAMES["register_filename"]
            self.srvs["get_task_trajs"] = SRV_NAMES["get_task_trajs"]
            self.srvs["get_joint_trajs"] = SRV_NAMES["get_joint_trajs"]

        rospy.loginfo("Waiting for ROS services...")
        for srv in self.srvs.keys():
            rospy.loginfo("    %s" % self.srvs[srv])
            rospy.wait_for_service(self.srvs[srv])
        rospy.loginfo("Services are up!")

        self._create_board()

    def run(self):
        """
        Display the GUI and start monitoring button presses.
        """
        self.root.mainloop()

    # === BEGIN Demo button press callback functions ==========================

    def _demo_btn_cb(self):
        """
        Start and stop gravity compensation while logging trajectory data.
        """
        if not self.recording and self.demo_btn_txt.get() == "Start\nDemo":
            self._start_demo()
        elif self.recording and self.demo_btn_txt.get() == "Stop\nDemo":
            self._stop_demo()

    def _grav_btn_cb(self):
        """
        Start and stop gravity compensation without logging any data.
        """
        if self.grav_btn_txt.get() == "Start\nGrav\nComp":
            # Start gravity compensation mode for the robot
            if set_controller_mode(self.srvs["set_grav_comp_mode"]):
                self.grav_btn_txt.set("Stop\nGrav\nComp")
                self.status_label.config(text="GRAV\nCOMP\nMODE")
                self.indicator_frame['highlightbackground'] = 'dark green'
                self.status_label['fg'] = 'dark green'
                self.status_label['bg'] = 'lime green'
        else:
            # Stop gravity compensation mode for the robot
            if set_controller_mode(self.srvs["set_krc_mode"]):
                self.grav_btn_txt.set("Start\nGrav\nComp")
                self.status_label.config(text="POSITION\nCONTROL\nACTIVE")
                self.indicator_frame['highlightbackground'] = 'red4'
                self.status_label['fg'] = 'red4'
                self.status_label['bg'] = 'IndianRed2'

    # === END Demo button press callback functions ============================

    # === BEGIN Trajectory button press callback functions ====================

    def _set_move_to_start_btn_cb(self):
        """
        Populate the trajectory points that will move robot from current 
        position to start position.
        """
        set_move_to_start(self.srvs["set_move_to_start"])

    def _clear_btn_cb(self):
        """
        Get rid of all data action server potentially has stored for trajectory 
        to be executed.
        """
        clear_all(self.srvs["clear_all_jtas"])

    def _execute_traj_btn_cb(self):
        """
        Request action server to start executing whatever reviewed trajectory 
        it has stored.
        """
        self._traj_client.send_goal(wait_for_result=True)

    # === END Trajectory button press callback functions ======================

    # === BEGIN ReFlex Hand button press callback functions ===================

    def _grasp_btn_cb(self):
        """
        Handle closing the hand as well as recording the end-effector pose in
        the object frame for active learning.
        """
        if not self.offline_mode:
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
        if self.use_reflex:
            command_reflex_hand(self.srvs["grasp_reflex"])

    def _open_hand_btn_cb(self):
        """
        Opens the hand to its zero position.
        """
        if self.use_reflex:
            command_reflex_hand(self.srvs["open_reflex_hand"])

    # === END ReFlex Hand button press callback functions =====================

    # === BEGIN Policy button press callback functions ========================

    def _preview_policy_btn_cb(self):
        """
        Request to action server to show virtual rendering of what will be 
        executed on robot.
        """
        success, self.promp_name = review_trajectory(self.srvs["preview_policy"])
        if success:
            # See if the user liked what they saw
            dialog = PreviewExecutionDialog(self.root)
            self.root.wait_window(dialog.top)

    def _execute_policy_btn_cb(self):
        """
        Request action server to start executing whatever reviewed policy it 
        has stored.
        """
        success = self._policy_client.send_goal(wait_for_result=True)
        if success:
            dialog = PostExecutionDialog(self.root, self.demo_path,
                                         self.promp_name)
            self.root.wait_window(dialog.top)

    def _clear_policy_btn_cb(self):
        """
        Clear all policy information on the action server.
        """
        clear_all(self.srvs["clear_all_policy"])
        # clear visualizations
        try:
            clear = rospy.ServiceProxy(self.srvs["clear_viz"], Trigger)
            clear()
        except rospy.ServiceException as e:
            rospy.logwarn(
                "Service call to clear visualizations failed: %s" % e)

    def _gen_instance_btn_cb(self):
        """
        Genererate a new task instance that the user should give a demo for.
        """
        success = False
        try:
            gen = rospy.ServiceProxy(self.srvs["gen_task_instance"], Trigger)
            resp = gen()
            success = resp.success
        except rospy.ServiceException as e:
            rospy.logerr(
                "Service call to generate next task instance failed: %s" % e)
            return False
        if not success:
            rospy.logwarn("Could not generate task instance: {}".format(
                resp.message))
            return False

        # See if the user wants to keep, discard, or mark infeasible
        dialog = GenerateTaskInstanceDialog(self.root)
        self.root.wait_window(dialog.top)

    # === END Policy button press callback functions ==========================

    def _start_demo(self):
        """
        Put robot in gravity compensation, zero out reflex sensors, and start 
        recording demo.
        """
        if not self.recording:
            # Zero out the ReFlex tactile sensors
            if self.use_reflex:
                zero_reflex_tactile(self.srvs["zero_reflex_tactile"])
            # Start gravity compensation mode for the robot
            if not set_controller_mode(self.srvs["set_grav_comp_mode"]):
                return False
            # Start rosbag recorder
            set_rosbag_status(
                self.srvs["set_recording"], self.demo_path, recording=True)
            # Change buttons appearance
            self.recording = True
            self.status_label.config(text="GRAV\nCOMP\nMODE")
            self.indicator_frame['highlightbackground'] = 'dark green'
            self.status_label['fg'] = 'dark green'
            self.status_label['bg'] = 'lime green'
            self.demo_btn_txt.set("Stop\nDemo")

        return True

    def _stop_demo(self):
        """
        Stop recording the current demo and get out of gravity compensation mode.
        """
        if self.recording:
            # Stop gravity compensation mode
            if not set_controller_mode(self.srvs["set_krc_mode"]):
                return False

            # Gravity comp should not be active, change button and indicator
            self.recording = False
            self.status_label.config(text="POSITION\nCONTROL\nACTIVE")
            self.indicator_frame['highlightbackground'] = 'red4'
            self.status_label['fg'] = 'red4'
            self.status_label['bg'] = 'IndianRed2'
            self.demo_btn_txt.set("Start\nDemo")

            # Stop rosbag recorder
            set_rosbag_status(self.srvs["set_recording"], self.demo_path, recording=False)

            # Get user input on what to do with the demonstration
            dialog = DemonstrationDialog(self.root, self.demo_path)
            self.root.wait_window(dialog.top)
        return True

    def _create_board(self):
        """
        Creates the GUI using tKinter.
        """
        self.root = tk.Tk()
        self.root.title("Policy Execution Session")
        w = 940
        h = 940
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        x = w + 100
        y = (hs / 2) - (h / 2)
        self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        big_btn_font = tkFont.Font(family='Helvetica', size=50, weight='bold')
        mid_btn_font = tkFont.Font(family='Helvetica', size=30, weight='bold')
        normal_btn_font = tkFont.Font(
            family='Helvetica', size=20, weight='bold')
        indicator_font = tkFont.Font(
            family='Helvetica', size=40, weight='bold')

        # === BEGIN Demo frame ================================================

        ctrl_switch_frame = tk.Frame(self.root, bd=10)

        self.demo_btn_txt = tk.StringVar(value="Start\nDemo")
        self.demo_btn = tk.Button(
            ctrl_switch_frame,
            textvariable=self.demo_btn_txt,
            command=self._demo_btn_cb,
            width=9,
            height=3,
            bd=5)
        self.demo_btn['font'] = big_btn_font
        self.demo_btn.pack()

        self.indicator_frame = tk.Frame(
            ctrl_switch_frame,
            highlightthickness=20,
            highlightbackground="red4",
            bd=0)
        self.status_label = tk.Label(self.indicator_frame, width=11, height=4)
        self.status_label.config(text="POSITION\nCONTROL\nACTIVE")
        self.status_label['font'] = indicator_font
        self.status_label['fg'] = 'red4'
        self.status_label['bg'] = 'IndianRed2'
        self.status_label.pack()
        self.indicator_frame.pack()

        self.grav_btn_txt = tk.StringVar(value="Start\nGrav\nComp")
        self.grav_btn = tk.Button(
            ctrl_switch_frame,
            textvariable=self.grav_btn_txt,
            command=self._grav_btn_cb,
            width=15,
            height=4,
            bd=5)
        self.grav_btn['font'] = mid_btn_font
        self.grav_btn.pack()

        ctrl_switch_frame.pack(side=tk.LEFT)

        # === END Demo frame ==================================================

        grasp_traj_frame = tk.Frame(self.root)

        # === BEGIN Grasp frame ===============================================

        grasp_btn_frame = tk.Frame(
            grasp_traj_frame,
            highlightthickness=28,
            highlightbackground="blue",
            bd=2)
        grasp_btn_txt = tk.StringVar(value="Grasp")
        grasp_btn = tk.Button(
            grasp_btn_frame,
            textvariable=grasp_btn_txt,
            command=self._grasp_btn_cb,
            width=10,
            height=2,
            bd=5)
        grasp_btn['font'] = normal_btn_font
        grasp_btn.pack(side=tk.TOP)

        open_hand_btn_txt = tk.StringVar(value="Open Hand")
        open_hand_btn = tk.Button(
            grasp_btn_frame,
            textvariable=open_hand_btn_txt,
            command=self._open_hand_btn_cb,
            width=10,
            height=2,
            bd=5)
        open_hand_btn['font'] = normal_btn_font
        open_hand_btn.pack()

        grasp_btn_frame.pack(side=tk.TOP)

        # === BEGIN Grasp frame ===============================================

        # === BEGIN Trajectory frame ==========================================

        traj_btn_frame = tk.Frame(
            grasp_traj_frame,
            bd=10,
            highlightthickness=0,
            highlightbackground="gray17")

        other_btn_frame = tk.Frame(
            traj_btn_frame,
            highlightthickness=28,
            highlightbackground="dark olive green",
            bd=2)

        set_move_to_start_btn = tk.Button(
            other_btn_frame,
            width=10,
            height=2,
            bd=5,
            textvariable=tk.StringVar(value="Set MTS"),
            command=self._set_move_to_start_btn_cb)
        clear_btn = tk.Button(
            other_btn_frame,
            width=10,
            height=2,
            bd=5,
            textvariable=tk.StringVar(value="Clear All"),
            command=self._clear_btn_cb)

        set_move_to_start_btn['font'] = normal_btn_font
        clear_btn['font'] = normal_btn_font

        set_move_to_start_btn.pack()
        clear_btn.pack()

        other_btn_frame.pack()

        execute_traj_btn_frame = tk.Frame(
            traj_btn_frame,
            highlightthickness=28,
            highlightbackground="firebrick4",
            bd=2)
        execute_traj_btn = tk.Button(
            execute_traj_btn_frame,
            width=10,
            height=3,
            bd=5,
            textvariable=tk.StringVar(value="EXECUTE\nTrajectory"),
            command=self._execute_traj_btn_cb)
        execute_traj_btn['font'] = normal_btn_font
        execute_traj_btn.pack()
        execute_traj_btn_frame.pack()

        traj_btn_frame.pack(side=tk.LEFT)

        # === END Trajectory frame ============================================

        grasp_traj_frame.pack(side=tk.LEFT)

        # === BEGIN Policy frame ==============================================

        policy_btn_frame = tk.Frame(
            self.root,
            bd=10,
            highlightthickness=0,
            highlightbackground="gray17")

        go_btn_frame = tk.Frame(
            policy_btn_frame,
            highlightthickness=28,
            highlightbackground="medium sea green",
            bd=2)

        gen_instance_btn = tk.Button(
            go_btn_frame,
            width=10,
            height=2,
            bd=5,
            textvariable=tk.StringVar(value="Generate\nInstance"),
            command=self._gen_instance_btn_cb)
        clear_policy_btn = tk.Button(
            go_btn_frame,
            width=10,
            height=2,
            bd=5,
            textvariable=tk.StringVar(value="Clear All"),
            command=self._clear_policy_btn_cb)

        gen_instance_btn['font'] = normal_btn_font
        clear_policy_btn['font'] = normal_btn_font
        clear_policy_btn['fg'] = 'DodgerBlue4'

        gen_instance_btn.pack()
        clear_policy_btn.pack()

        go_btn_frame.pack()

        policy_review_btn_frame = tk.Frame(
            policy_btn_frame,
            highlightthickness=28,
            highlightbackground="goldenrod1",
            bd=2)
        policy_review_btn = tk.Button(
            policy_review_btn_frame,
            width=10,
            height=3,
            bd=5,
            textvariable=tk.StringVar(value="Preview\nPolicy"),
            command=self._preview_policy_btn_cb)
        policy_review_btn['font'] = normal_btn_font
        policy_review_btn['fg'] = 'DodgerBlue4'
        policy_review_btn.pack()
        policy_review_btn_frame.pack()

        execute_policy_btn_frame = tk.Frame(
            policy_btn_frame,
            highlightthickness=28,
            highlightbackground="firebrick1",
            bd=2)
        execute_policy_btn = tk.Button(
            execute_policy_btn_frame,
            width=10,
            height=3,
            bd=5,
            textvariable=tk.StringVar(value="EXECUTE\nPolicy"),
            command=self._execute_policy_btn_cb)
        execute_policy_btn['font'] = normal_btn_font
        execute_policy_btn['fg'] = 'DodgerBlue4'
        execute_policy_btn.pack()
        execute_policy_btn_frame.pack()

        policy_btn_frame.pack(side=tk.LEFT)

        # === END Policy frame ================================================


if __name__ == '__main__':
    rospy.init_node("policy_execution_gui")
    import sys
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--debug", dest="debug", action="store_true", default=False)
    args = parser.parse_args(rospy.myargv(sys.argv)[1:])
    try:
        gui = PolicyExecutionGUI(debug=args.debug)
        gui.run()
    except rospy.ROSInterruptException:
        pass
