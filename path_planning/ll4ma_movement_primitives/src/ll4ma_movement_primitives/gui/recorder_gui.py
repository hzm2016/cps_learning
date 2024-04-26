#!/usr/bin/env python
import os
import rospy
import Tkinter as tk
import tkFont
import tkMessageBox
from std_srvs.srv import Trigger
from trajectory_action_lib import JointTrajectoryActionClient

from rospy_service_helper import (
    set_controller_mode,
    set_rosbag_status,
    review_trajectory,
    set_review_status,
    zero_reflex_tactile,
    command_reflex_hand,
)


class DemonstrationRecorder:

    def __init__(self, robot_name="lbr4", debug=False):
        self.robot_name = robot_name
        self.recording = False

        # Services this node will call
        self.srvs = {
            "set_grav_comp_mode"  : "/lwr_controller_manager/enable_grav_comp",
            "set_krc_mode"        : "/lwr_controller_manager/enable_krc_stiffness",
            "set_rosbag_status"   : "/rosbag/set_recording",
            "review_jnt_traj"     : "/%s/review_joint_trajectory" % self.robot_name,
            "set_review_status"   : "/%s/set_trajectory_review_status" % self.robot_name,
            "zero_reflex_tactile" : "/reflex_takktile/calibrate_tactile",
            "open_reflex_hand"    : "/reflex_interface/open_hand",
            "close_reflex_hand"   : "/reflex_interface/close_hand",
            "tighten_reflex_hand" : "/reflex_interface/tighten_hand"
        }

        if not debug:
            rospy.loginfo("Waiting for services...")
            for srv in self.srvs.keys():
                rospy.loginfo("    %s" % self.srvs[srv])
                rospy.wait_for_service(self.srvs[srv])
            rospy.loginfo("Services are up!")        

            self._client = JointTrajectoryActionClient(robot_name="lbr4", rospy_init=False) # TODO name
        
        self._create_board()

    def run(self):
        """
        Display the GUI and start monitoring button presses.
        """
        self.root.mainloop()


        
    # === BEGIN button press callback functions ====================================================
        
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
        if self.grav_btn_txt.get() == "Start Grav Comp":
            # Start gravity compensation mode for the robot
            success = set_controller_mode(self.srvs["set_grav_comp_mode"])
            if success:
                self.grav_btn_txt.set("Stop Grav Comp")
                self.status_label.config(text="GRAV\nCOMP\nMODE")
                self.indicator_frame['highlightbackground'] = 'dark green'
                self.status_label['fg'] = 'dark green'
                self.status_label['bg'] = 'lime green'
        else:
            # Stop gravity compensation mode for the robot
            success = set_controller_mode(self.srvs["set_krc_mode"])
            if success:
                self.grav_btn_txt.set("Start Grav Comp")
                self.status_label.config(text="POSITION\nCONTROL\nACTIVE")
                self.indicator_frame['highlightbackground'] = 'red4'
                self.status_label['fg'] = 'red4'
                self.status_label['bg'] = 'IndianRed2'
            
    def _review_btn_cb(self):
        """
        Request to action server to show virtual rendering of what will be executed on robot.
        """
        success = review_trajectory(self.srvs["review_jnt_traj"])
        if not success:
            return False

        # Open dialog to ask user if it looked okay
        user_response = tkMessageBox.askquestion("Trajectory Review", "Did that look okay?")
        success = set_review_status(self.srvs["set_review_status"], user_response == 'yes')
        return success

    def _execute_btn_cb(self):
        """
        Request action server to start executing whatever reviewed trajectory it has stored.
        """
        #TODO the goal here is kind of bogus because I set it up to require a review first before
        # even trying to execute. So this can be changed later if there's a better way.

        # Send goal to action server
        self._client.send_goal(True, wait_for_result=True)
        
    def _set_start_btn_cb(self):
        """
        Change the demo start position.
        """
        success = False
        try:
            set_start = rospy.ServiceProxy("/%s/set_demo_start" % self.robot_name, Trigger)
            resp = set_start()
            success = resp.success
        except rospy.ServiceException as e:
            rospy.logwarn("Service request to set demo start failed: %s" % e)
            return False
        if not success:
            rospy.logwarn("Could not set demo start.")
            return False
            
    def _set_move_to_start_btn_cb(self):
        """
        Populate the trajectory points that will move robot from current position to start position.
        """
        success = False
        try:
            set_move_to_start = rospy.ServiceProxy("/%s/set_move_to_start" % self.robot_name, Trigger)
            resp = set_move_to_start()
            success = resp.success
        except rospy.ServiceException as e:
            rospy.logwarn("Service request to set move-to-start points failed: %s" %e)
            return False
        if not success:
            rospy.logwarn("Could not set move-to-start points.")
            return False

    def _clear_btn_cb(self):
        """
        Get rid of all data action server potentially has stored for trajectory to be executed.
        """
        success = False
        try:
            clear = rospy.ServiceProxy("/%s/clear_all" % self.robot_name, Trigger)
            resp = clear()
            success = resp.success
        except rospy.ServiceException as e:
            rospy.logwarn("Service request to clear data failed: %s" %e)
            return False
        if not success:
            rospy.logwarn("Could not clear data.")
            return False

    def _grasp_btn_cb(self):
        command_reflex_hand(self.srvs["close_reflex_hand"])
        rospy.sleep(2.5) # TODO hack to wait to close, should really monitor tactile stops
        command_reflex_hand(self.srvs["tighten_reflex_hand"])

    def _open_hand_btn_cb(self):
        command_reflex_hand(self.srvs["open_reflex_hand"])
        
        
    # === END button press callback functions ======================================================

    
    def _start_demo(self):
        """
        Put robot in gravity compensation, zero out reflex sensors, and start recording demo.
        """
        success = False
        if not self.recording:
            # Zero out the ReFlex tactile sensors
            zero_reflex_tactile(self.srvs["zero_reflex_tactile"])

            # Start gravity compensation mode for the robot
            success = set_controller_mode(self.srvs["set_grav_comp_mode"])
            if not success:
                return False

            # Start rosbag recorder (backup for logger and records more information)
            success = set_rosbag_status(self.srvs["set_rosbag_status"], set_recording=True)
                        
            # Data should be recording and grav comp active, change buttons appearance
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
        success = False
        if self.recording:
            # Stop gravity compensation mode
            success = set_controller_mode(self.srvs["set_krc_mode"])
            if not success:
                return False
                
            # Gravity comp should not be active, change button and indicator
            self.recording = False
            self.status_label.config(text="POSITION\nCONTROL\nACTIVE")
            self.indicator_frame['highlightbackground'] = 'red4'
            self.status_label['fg'] = 'red4'
            self.status_label['bg'] = 'IndianRed2'
            self.demo_btn_txt.set("Start\nDemo")
            
            # Stop rosbag recorder
            success = set_rosbag_status(self.srvs["set_rosbag_status"], set_recording=False)
        return success
    
    def _create_board(self):
        self.root = tk.Tk()
        self.root.title("Demonstration Recorder")
        w = 1000
        h = 1000
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        x = w + w/3
        y = (hs/2) - (h/2)
        self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        big_btn_font = tkFont.Font(family='Helvetica', size=50, weight='bold')
        mid_btn_font = tkFont.Font(family='Helvetica', size=30, weight='bold')
        normal_btn_font = tkFont.Font(family='Helvetica', size=20, weight='bold')
        indicator_font = tkFont.Font(family='Helvetica', size=40, weight='bold')

        ctrl_switch_frame = tk.Frame(self.root, bd=10)

        self.demo_btn_txt = tk.StringVar(value="Start\nDemo")
        self.demo_btn = tk.Button(ctrl_switch_frame, textvariable=self.demo_btn_txt,
                                  command=self._demo_btn_cb, width=10, height=4, bd=5)
        self.demo_btn['font'] = big_btn_font
        self.demo_btn.pack()
        
        self.indicator_frame = tk.Frame(ctrl_switch_frame, highlightthickness=20,
                                        highlightbackground="red4", bd=0)
        self.status_label = tk.Label(self.indicator_frame, width=12, height=4)
        self.status_label.config(text="POSITION\nCONTROL\nACTIVE")
        self.status_label['font'] = indicator_font
        self.status_label['fg'] = 'red4'
        self.status_label['bg'] = 'IndianRed2'
        self.status_label.pack()
        self.indicator_frame.pack()

        self.grav_btn_txt = tk.StringVar(value="Start Grav Comp")
        self.grav_btn = tk.Button(ctrl_switch_frame, textvariable=self.grav_btn_txt,
                                  command=self._grav_btn_cb, width=17, height=4, bd=5)
        self.grav_btn['font'] = mid_btn_font
        self.grav_btn.pack()
    
        ctrl_switch_frame.pack(side=tk.LEFT)

        btn_frame = tk.Frame(self.root, bd=10)

        grasp_btn_frame = tk.Frame(btn_frame, highlightthickness=28, highlightbackground="blue",
                                   bd=2)
        grasp_btn_txt = tk.StringVar(value="Grasp")
        grasp_btn = tk.Button(grasp_btn_frame, textvariable=grasp_btn_txt,
                              command=self._grasp_btn_cb, width=10, height=2, bd=5)
        grasp_btn['font'] = mid_btn_font
        grasp_btn.pack(side=tk.LEFT)

        open_hand_btn_txt = tk.StringVar(value="Open\nHand")
        open_hand_btn = tk.Button(grasp_btn_frame, textvariable=open_hand_btn_txt,
                              command=self._open_hand_btn_cb, width=10, height=2, bd=5)
        open_hand_btn['font'] = mid_btn_font
        open_hand_btn.pack()


        grasp_btn_frame.pack()

        other_btn_frame = tk.Frame(btn_frame, highlightthickness=28, highlightbackground="green",
                                    bd=2)
        set_start_btn_txt = tk.StringVar(value="Reset Demo Start")
        set_start_btn = tk.Button(other_btn_frame, textvariable=set_start_btn_txt,
                                  command=self._set_start_btn_cb, width=31, height=2, bd=5)
        set_start_btn['font'] = normal_btn_font
        set_start_btn.pack()

        set_move_to_start_btn_txt = tk.StringVar(value="Set Move-To-Start")
        set_move_to_start_btn = tk.Button(other_btn_frame, textvariable=set_move_to_start_btn_txt,
                                          command=self._set_move_to_start_btn_cb, width=31,
                                          height=2, bd=5)
        set_move_to_start_btn['font'] = normal_btn_font
        set_move_to_start_btn.pack()

        clear_btn_txt = tk.StringVar(value="Clear All")
        clear_btn = tk.Button(other_btn_frame, textvariable=clear_btn_txt,
                              command=self._clear_btn_cb, width=31, height=2, bd=5)
        clear_btn['font'] = normal_btn_font
        clear_btn.pack()

        other_btn_frame.pack()

        review_btn_frame = tk.Frame(btn_frame, highlightthickness=28, highlightbackground="yellow",
                                    bd=2)
        review_btn_txt = tk.StringVar(value="Review")
        review_btn = tk.Button(review_btn_frame, textvariable=review_btn_txt,
                               command=self._review_btn_cb, width=31, height=3, bd=5)
        review_btn['font'] = normal_btn_font
        review_btn.pack()
        review_btn_frame.pack()

        execute_btn_frame = tk.Frame(btn_frame, highlightthickness=28, highlightbackground="red", bd=2)
        execute_btn_txt = tk.StringVar(value="EXECUTE")
        execute_btn = tk.Button(execute_btn_frame, textvariable=execute_btn_txt,
                                command=self._execute_btn_cb, width=31, height=3, bd=5)
        execute_btn['font'] = normal_btn_font
        execute_btn.pack()
        execute_btn_frame.pack()
        
        btn_frame.pack(side=tk.LEFT)
        
            
if __name__ == '__main__':
    rospy.init_node("demonstration_recorder")
    import sys
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--debug", dest="debug", action="store_true", default=False)
    args = parser.parse_args(rospy.myargv(sys.argv)[1:])

    try:
        demo_recorder = DemonstrationRecorder(debug=args.debug)
        demo_recorder.run()
    except rospy.ROSInterruptException:
        pass
