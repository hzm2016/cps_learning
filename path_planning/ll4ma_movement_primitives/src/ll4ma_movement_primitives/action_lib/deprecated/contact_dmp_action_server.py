#!/usr/bin/env python
import os
import roslib
import rospy
import h5py
import actionlib
import tf
import numpy as np
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Wrench, PoseStamped
from force_sensor import ContactClassifier
from policy_learning.dmps import DynamicMovementPrimitive, CanonicalSystem
from policy_learning.util import quaternion
from ll4ma_robot_control.msg import (
    OpSpaceTrajectoryAction,
    OpSpaceTrajectoryFeedback,
    OpSpaceTrajectoryResult,
    OpSpaceCommand,
    RobotState
)


class TrajectoryActionServer:

    def __init__(self, pose_tolerance = 0.008, wrench_tolerance = 0.1, window_size=50, mu_noise=0.8):
        rospy.init_node("trajectory_action_server")
        self.pose_tolerance = pose_tolerance
        # self.error_tolerance = error_tolerance
        self.wrench_tolerance = wrench_tolerance
        self.log_filename = rospy.get_param("/db_name")
        self.robot_name = rospy.get_param("/robot_name")
        self.root_link = rospy.get_param("/%s/root_link" % self.robot_name)
        self.rate = rospy.Rate(rospy.get_param("/%s/run_rate" % self.robot_name))
        self.ns = "/%s/execute_trajectory" % self.robot_name
        self.server = actionlib.SimpleActionServer(self.ns, OpSpaceTrajectoryAction,
                                                   self.goal_cb, False)
        self.feedback = OpSpaceTrajectoryFeedback()
        self.result = OpSpaceTrajectoryResult()
        self.robot_state = None
        self.jnt_state = None
        self.goal_pose = None
        self.make_contact_cf = None
        self.make_contact_mag = 0.0
        self.goal_wrench = np.zeros(6)
        self.goal_mask = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.current_wrench = np.zeros(6)
        self.op_cmd = OpSpaceCommand()
        self.jnt_cmd = JointState()
        self.classifier = ContactClassifier(window_size=window_size, mu_noise=mu_noise)
        self.tf_broadcaster = tf.TransformBroadcaster()

        # TEMPORARY for debugging
        self.contact_pub = rospy.Publisher("/in_contact", Bool, queue_size=1)

        self.jnt_des_cmd_topic = rospy.get_param("/%s/jnt_des_cmd_topic" % self.robot_name)
        self.jnt_state_topic = rospy.get_param("/%s/jnt_state_topic" % self.robot_name)
        self.os_cmd_topic = rospy.get_param("/%s/os_cmd_topic" % self.robot_name)

        self.op_cmd_pub = rospy.Publisher(self.os_cmd_topic, OpSpaceCommand, queue_size=1)
        self.jnt_des_cmd_pub = rospy.Publisher(self.jnt_des_cmd_topic, JointState, queue_size=1)
        self.cf_pub = rospy.Publisher("/visualize_constraint_frame", PoseStamped, queue_size=1)
        rospy.Subscriber(self.jnt_state_topic, JointState, self._jnt_state_cb)
        rospy.Subscriber("/%s/robot_state" % self.robot_name, RobotState, self._robot_state_cb)

        self.h5_path = os.path.abspath(os.path.expanduser("~/.ros/ll4ma_robot_control"))
        self._log = {}
        # get the index we want
        try:
            with open(os.path.join(self.h5_path, ".%s.index" % self.log_filename), 'r') as f:
                self.log_idx = int(f.readline())
        except (IOError):
            self.log_idx = 1



    def goal_cb(self, goal):
        rospy.loginfo("[TrajectoryActionServer] New goal received.")
        self.result.success = False

        if len(goal.trajectory.poses) > 0 or len(goal.trajectory.dmp_configs) > 0:
            # running op space trajectory, wait for robot state
            if self.robot_state is None:
                rospy.logwarn("[TrajectoryActionServer] Robot state is unknown. "
                              "Waiting for 10 seconds...")
                i = 0
                while i < 100 and self.robot_state is None:
                    rospy.sleep(0.1)
                    i += 1
                # don't try to run trajectory if you don't know where the robot is
                if self.robot_state is None:
                    self.server.set_aborted()
                    rospy.logerr("[TrajectoryActionServer] GOAL ABORTED. Current robot state unknown.")
                    self.result.success = False
        else:
            # running joint trajectory, need to get joint state
            if self.jnt_state is None:
                rospy.logwarn("[TrajectoryActionServer] Joint state is unknown. "
                              "Waiting for 10 seconds...")
                i = 0
                while i < 100 and self.jnt_state is None:
                    rospy.sleep(0.1)
                    i +=1
                if self.jnt_state is None:
                    self.server.set_aborted()
                    rospy.logerr("[TrajectoryActionServer] GOAL ABORTED. Current joint state unknown.")
                    self.result.success = False

        # run as joint trajectory if desired joint states provided
        if len(goal.trajectory.joint_states) > 0:
            self._execute_joint_trajectory(goal)
        # run as a DMP if parameters are provided
        elif len(goal.trajectory.dmp_configs) > 0:
            self._execute_dmp_trajectory(goal)
        # otherwise op space trajectory, check that you actually have something to command
        elif not goal.trajectory.poses:
            self.server.set_aborted()
            rospy.logerr("[TrajectoryActionServer] GOAL ABORTED. Trajectory was empty.")
            self.result.success = False
        # command a normal trajectory
        else:
            self._execute_trajectory(goal)
            
        # report results to console
        if self.result.success:
            rospy.loginfo("[TrajectoryActionServer] GOAL REACHED.")
            self.server.set_succeeded(self.result)
        else:
            rospy.logwarn("[TrajectoryActionServer] FAILURE. Goal was not reached.")
            self.server.set_aborted()

        rospy.loginfo("Pose Error = %s" % str(self._pose_error(orientation=False))) # TODO temporary
        rospy.loginfo("Wrench Error = %s" % str(self.wrench_error))
            
    def start(self):
        rospy.loginfo("[TrajectoryActionServer] Server is running. Waiting for a goal...")
        self.server.start()
        while not rospy.is_shutdown():
            self.rate.sleep()
        rospy.loginfo("[TrajectoryActionServer] Server shutdown. Exiting.")

    # TODO need to fix these to be consistent with how DMP pose error and such are being handled.
    # OR better yet, put them in their own action server, they really don't belong here.
        
    # def _execute_trajectory(self, goal):
    #     rospy.loginfo("[TrajectoryActionServer] Publishing trajectory commands to %s"
    #                   % self.os_cmd_topic)
        
    #     # set goal to be final pose in the trajectory
    #     self.goal_pose = goal.trajectory.poses[-1]
        
    #     for i, pose in enumerate(goal.trajectory.poses):
    #         # check to see if goal was preempted
    #         if self.server.is_preempt_requested():
    #             self.server.set_preempted()
    #             rospy.logwarn("[TrajectoryActionServer] GOAL PREEMPTED")
    #             self.result.success = False
    #             break

    #         # check pose to determine if goal is reached or error tolerance is exceeded
    #         pose_error = self._pose_error()
    #         # if pose_error > self.error_tolerance:
    #         #     rospy.logwarn("[TrajectoryActionServer] GOAL ABORTED.")
    #         #     rospy.logwarn("Pose error %f exceeded acceptable limit of %f"
    #         #                  % (pose_error, self.error_tolerance))
    #         #     self.result.success = False
    #         #     return
                
    #         self.op_cmd.pose = pose
    #         # populate also twist and wrench if available
    #         if len(goal.trajectory.twists) > i:
    #             self.op_cmd.twist = goal.trajectory.twists[i]
    #         if len(goal.trajectory.accels) > i:
    #             self.op_cmd.accel = goal.trajectory.accels[i]
    #         if len(goal.trajectory.wrenches) > i:
    #             self.op_cmd.wrench = goal.trajectory.wrenches[i]
    #         if len(goal.trajectory.constraints) > i:
    #             self.op_cmd.constraints = goal.trajectory.constraints[i].values
    #         if len(goal.trajectory.constraint_frames) > i:
    #             self.op_cmd.constraint_frame = goal.trajectory.constraint_frames[i]

    #         self.op_cmd_pub.publish(self.op_cmd)
    #         self.server.publish_feedback(self.feedback)
    #         self.rate.sleep()

    #     # check if goal was met
    #     if self._pose_error() < self.pose_tolerance:
    #         self.result.success = True

    #     # publish last command a while longer to try to reach goal if it wasn't met
    #     if not self.result.success and self.server.is_active():
    #         # TODO can make this variable time if desired
    #         msg = ("[TrajectoryActionServer] Goal was not reached after commanding full "
    #                "trajectory. Trying a while longer...")
    #         rospy.logwarn(msg)
    #         i = 0
    #         while i < 100 and not self.result.success:
    #             self.op_cmd_pub.publish(self.op_cmd)
    #             if self._pose_error() < self.pose_tolerance:
    #                 self.result.success = True
    #             i += 1

    # def _execute_joint_trajectory(self, goal):
    #     rospy.loginfo("[TrajectoryActionServer] Publishing joint trajectory commands to %s"
    #                   % self.jnt_des_cmd_topic)
        
    #     self.goal_state = goal.trajectory.joint_states[-1]
        
    #     for i, joint_state in enumerate(goal.trajectory.joint_states):
    #         # check to see if goal was preempted
    #         if self.server.is_preempt_requested():
    #             self.server.set_preempted()
    #             rospy.logwarn("[TrajectoryActionServer] GOAL PREEMPTED")
    #             self.result.success = False
    #             break

    #         # check position to determine if error tolerance is exceeded
    #         error = self._joint_error(self.jnt_state.position, self.jnt_cmd.position, norm=True)
    #         # if error > self.error_tolerance:
    #         #     rospy.logwarn("[TrajectoryActionServer] GOAL ABORTED.")
    #         #     rospy.logwarn("Pose error %f exceeded acceptable limit of %f"
    #         #                  % (error, self.error_tolerance))
    #         #     self.result.success = False
    #         #     return
                
    #         self.jnt_cmd.position = joint_state.position
    #         self.jnt_des_cmd_pub.publish(self.jnt_cmd)
    #         self.server.publish_feedback(self.feedback)
    #         self.rate.sleep()

    #     # check if goal was met
    #     self.pose_error = self._joint_error(self.jnt_state.position, self.goal_state.position,
    #                                         norm=True)
    #     if self.pose_error < self.pose_tolerance:
    #         self.result.success = True

    #     # publish last command a while longer to try to reach goal if it wasn't met
    #     if not self.result.success and self.server.is_active():
    #         # TODO can make this variable time if desired
    #         msg = ("[TrajectoryActionServer] Goal was not reached after commanding full "
    #                "trajectory. Trying a while longer...")
    #         rospy.logwarn(msg)
    #         i = 0
    #         while i < 100 and not self.result.success:
    #             self.jnt_des_cmd_pub.publish(self.jnt_cmd)
    #             self.pose_error = self._joint_error(self.jnt_state.position, self.goal_state.position,
    #                                                 norm=True)
    #             if self.pose_error < self.pose_tolerance:
    #                 self.result.success = True
    #             i += 1

    def _execute_dmp_trajectory(self, goal):
        self.commanded_trajectory = goal.trajectory.commanded_trajectory
        self.op_cmd = OpSpaceCommand()
        self.op_cmd.constraint_frame = Pose()
        self.op_cmd.constraint_frame.position.x = 0.0
        self.op_cmd.constraint_frame.position.y = 0.0
        self.op_cmd.constraint_frame.position.z = 0.0
        self.op_cmd.constraint_frame.orientation.x = 0.0
        self.op_cmd.constraint_frame.orientation.y = 0.0
        self.op_cmd.constraint_frame.orientation.z = 0.0
        self.op_cmd.constraint_frame.orientation.w = 1.0
        
        # use a single canonical system to coordinate DMPs
        self.cs = CanonicalSystem()

        phase_types = {}
        for dmp_config in goal.trajectory.dmp_configs:
            phase_types[dmp_config.phase_name] = dmp_config.phase_type
        phase_dmps = self._get_dmps(goal.trajectory.dmp_configs)
        if phase_dmps is None:
            rospy.logerr("[TrajectoryActionServer] Could not create DMPs from configs.")
            self.result.success = False
            return False

        phases = phase_dmps.keys()
        
        success = False
        
        for i, phase in enumerate(sorted(phases)):
            rospy.loginfo("[TrajectoryActionServer] Executing phase '%s'..." % phase)
            
            elements = phase_dmps[phase]
            # initialize logging storage
            self._log[phase] = {}
            self._log[phase]['cs'] = []
            self._log[phase]['goal'] = {}
            self._log[phase]['val'] = {}
            self._log[phase]['val_dot'] = {}
            self._log[phase]['val_dotdot'] = {}
            self._log[phase]['fb'] = {}
            for element in elements:
                dims = elements[element]
                self._log[phase]['goal'][element] = {}
                self._log[phase]['val'][element] = {}
                self._log[phase]['val_dot'][element] = {}
                self._log[phase]['val_dotdot'][element] = {}
                self._log[phase]['fb'][element] = {}
                for dim in dims:
                    self._log[phase]['goal'][element][dim] = []
                    self._log[phase]['val'][element][dim] = []
                    self._log[phase]['val_dot'][element][dim] = []
                    self._log[phase]['val_dotdot'][element][dim] = []
                    self._log[phase]['fb'][element][dim] = []
                    
            # execute DMP trajectory based on phase type
            start_time = rospy.get_time()
            if phase_types[phase] == 'free_space':
                success = self._execute_free_space(phase, elements, start_time)
            elif phase_types[phase] == 'making_contact':
                success = self._execute_make_contact(phase, elements, start_time)
            elif phase_types[phase] == 'in_contact':
                success = self._execute_in_contact(phase, elements, start_time, use_cf=True)
            elif phase_types[phase] == 'breaking_contact':
                success = self._execute_break_contact(phase, elements, start_time)
            else:
                rospy.logerr("[TrajectoryActionServer] Unknown phase type: %s" % phase_types[phase])
                return False

            # adapt to end conditions based on phase type
            if success:
                if phase_types[phase] == 'making_contact':
                    start_time = rospy.get_time()
                    success = self._adapt_make_contact(phase, elements, start_time)

        if success:
            rospy.loginfo("[TrajectoryActionServer] All phases completed successfully.")
            self.result.success = True
            return True
        else:
            rospy.logwarn("[TrajectoryActionServer] Phases could not be fully executed.")


    def _execute_free_space(self, phase, elements, start_time, timeout=20.0):
        rospy.loginfo("[TrajectoryActionServer] Executing FREE SPACE motion...")
        timed_out = False
        self.op_cmd.constraints = [1, 1, 1, 1, 1, 1]
        prev_time = 0.0
        self.rate.sleep()
        
        # initialize DMPs with current robot state
        self._set_dmp_init(elements)
        self.cs.reset()
        
        while not timed_out:
            # update system state
            current_time = rospy.get_time() - start_time
            dt = current_time - prev_time
            pose_error = self._pose_error()
            wrench_mag = self._get_wrench_mag(self.robot_state.wrench) # TODO magnitude, or per-d?
            self.in_contact = self.classifier.in_contact()
            x = self.cs.get_value(dt, pose_error, wrench_mag, self.in_contact)
            self._log[phase]['cs'].append(x)
            for element in elements.keys():
                dims = elements[element]
                if element == 'x':
                    self._populate_pose_from_dmps(phase, dims, x, dt, pose_error, wrench_mag, self.in_contact)
                elif element in ['w', 'w_filt_base']:
                    self._populate_wrench_from_dmps(phase, dims, x, dt, pose_error, wrench_mag, self.in_contact)
                elif element == 'cf':
                    self._populate_cf_from_dmps(dims, x, dt)
                else:
                    rospy.logerr("[TrajectoryActionServer] Unknown element type: %s" % element)

            self.op_cmd.header.stamp = rospy.Time.now()
            self.op_cmd_pub.publish(self.op_cmd)

            # TODO position error since orientation isn't factoring in right now
            if self._goal_pose_error(orientation=False) < self.pose_tolerance:
                rospy.loginfo("[TrajectoryActionServer] Desired pose achieved "
                              "for FREE SPACE motion.")
                return True
            
            prev_time = current_time
            timed_out = current_time > timeout
            self.rate.sleep()
            rospy.loginfo(pose_error)

        if timed_out:
            rospy.logwarn("[TrajectoryActionServer] Timed out executing FREE SPACE motion!")
            return False
        else:
            return True

    def _execute_make_contact(self, phase, elements, start_time, pose_tolerance=0.004, timeout=30.0):
        rospy.loginfo("[TrajectoryActionServer] Trying to MAKE CONTACT...")
        first_contact = True
        timed_out = False
        S_f_z = 1
        self.op_cmd.wrench = Wrench()
        self.op_cmd.wrench.force.x = self.goal_wrench[0]
        self.op_cmd.wrench.force.y = self.goal_wrench[1]
        self.op_cmd.wrench.force.z = self.goal_wrench[2]
        self.op_cmd.wrench.torque.x = self.goal_wrench[3]
        self.op_cmd.wrench.torque.y = self.goal_wrench[4]
        self.op_cmd.wrench.torque.z = self.goal_wrench[5]
        self.op_cmd.desired_in_cf = False
        self.op_cmd.constraints = [1, 1, 1, 1, 1, 1]
        prev_time = 0.0
        self.rate.sleep()
        
        # initialize DMPs with current robot state
        self._set_dmp_init(elements)
        self.cs.reset()
        
        while not timed_out:
            # update system state
            current_time = rospy.get_time() - start_time
            dt = current_time - prev_time
            pose_error = self._pose_error()
            wrench_mag = self._get_wrench_mag(self.robot_state.wrench) # TODO magnitude, or per-d?
            self.in_contact = self.classifier.in_contact()
            x = self.cs.get_value(dt, pose_error, wrench_mag, self.in_contact)
            self._log[phase]['cs'].append(x)
            for element in elements.keys():
                dims = elements[element]
                if element == 'x':
                    self._populate_pose_from_dmps(phase, dims, x, dt, pose_error, wrench_mag, self.in_contact)
                elif element in ['w', 'w_filt_base']:
                    self._populate_wrench_from_dmps(phase, dims, x, dt, pose_error, wrench_mag, self.in_contact)
                elif element == 'cf':
                    self._populate_cf_from_dmps(dims, x, dt)
                else:
                    rospy.logerr("[TrajectoryActionServer] Unknown element type: %s" % element)

            if self.in_contact:
                # self.op_cmd.constraints = [1, 1, 0, 1, 1, 1]
                if first_contact:
                    rospy.loginfo("[TrajectoryActionServer] First contact made!")
                    g_c = self.goal_pose.position.z
                    y_c = self.robot_state.pose.position.z
                    first_contact = False
                    
                # compute force control coefficient for transition free space -> in-contact
                S_f_z = np.exp(1.0 - (abs(g_c - y_c) / abs(self.goal_pose.position.z
                                                           - self.robot_state.pose.position.z)))

            self.op_cmd.constraints = [1, 1, S_f_z, 1, 1, 1]
            # self.op_cmd.constraints = [1, 1, 1, 1, 1, 1]
            self.op_cmd.header.stamp = rospy.Time.now()
            self.op_cmd_pub.publish(self.op_cmd)

            # TODO only checking z position for now since that's all that's changing in straight down experiment
            if self._goal_pose_error(dim=2) < pose_tolerance:
            # if self._goal_pose_error(orientation=False) < pose_tolerance:
                rospy.loginfo("[TrajectoryActionServer] Desired pose achieved "
                              "for MAKING CONTACT phase.")
                return True
            
            prev_time = current_time
            timed_out = current_time > timeout
            self.rate.sleep()

        if timed_out:
            rospy.logwarn("[TrajectoryActionServer] Timed out executing MAKE CONTACT phase!")
            return False
        else:
            return True

    def _execute_in_contact(self, phase, elements, start_time, use_cf=True, pose_tolerance=0.03, timeout=30.0):
        rospy.loginfo("[TrajectoryActionServer] Executing IN CONTACT phase...")
        self.op_cmd.constraints = [1, 1, 0, 1, 1, 1]
        # self.op_cmd.desired_in_cf = False # SET THIS ONLY WHEN TESTING BASE FRAME
        self.op_cmd.desired_in_cf = True

        timed_out = False
        pose_tracked = False
        prev_time = 0.0
        self.rate.sleep()
        
        # initialize DMPs with current robot state
        self._set_dmp_init(elements)
        self.cs.reset()
        
        while not timed_out:
            # update system state
            current_time = rospy.get_time() - start_time
            dt = current_time - prev_time
            pose_error = self._pose_error()
            x = self.cs.get_value(dt, pose_error, 0.0, 0) # don't let contact FB affect it
            self._log[phase]['cs'].append(x)
            for element in elements.keys():
                dims = elements[element]
                if element == 'x':
                    self._populate_pose_from_dmps(phase, dims, x, dt, pose_error, 0.0, 0)
                elif element in ['w', 'w_filt_base']:
                    self._populate_wrench_from_dmps(phase, dims, x, dt, pose_error, 0.0, 0)
                elif element == 'cf':
                    self._populate_cf_from_dmps(dims, x, dt, pose_error, 0.0, use_cf, 0)
                else:
                    rospy.logerr("[TrajectoryActionServer] Unknown element type: %s" % element)
                    
            self.op_cmd.header.stamp = rospy.Time.now()
            self.op_cmd_pub.publish(self.op_cmd)

            # TODO a bit of a hack with prev time, just to get past case where start and end coincide
            # TODO position error since change in goal for quaternion is not totally working yet.
            if pose_tracked == False and prev_time > 2.0 and self._goal_pose_error(orientation=False) < pose_tolerance:
                rospy.loginfo("[TrajectoryActionServer] Desired pose achieved for IN CONTACT phase.")
                pose_tracked = True

            # TODO a bit of a hack with prev time, just to get past case where start and end coincide
            if pose_tracked and prev_time > 2.0 and self._wrench_error() < self.wrench_tolerance:
                rospy.loginfo("[TrajectoryActionServer] Desired wrench achieved for IN CONTACT phase.")
                # put the visualized CF out somewhere so it can't be seen
                pt = PoseStamped()
                pt.header.stamp = rospy.Time.now()
                pt.header.frame_id = 'push_ball_center'
                pt.pose = self.op_cmd.constraint_frame
                pt.pose.position.x = 100
                self.cf_pub.publish(pt)
                return True

            # rviz visualization of constraint frame
            pt = PoseStamped()
            pt.header.stamp = rospy.Time.now()
            pt.header.frame_id = 'push_ball_center'
            pt.pose = self.op_cmd.constraint_frame
            pt.pose.position.x = 0
            pt.pose.position.y = 0
            pt.pose.position.z = 0
            self.cf_pub.publish(pt)
            
            prev_time = current_time
            timed_out = current_time > timeout
            self.rate.sleep()

        if timed_out:
            rospy.logwarn("[TrajectoryActionServer] Timed out executing IN CONTACT phase!")
            return False
        else:
            return True

    def _execute_break_contact(self, phase, elements, start_time, pose_tolerance=0.009, timeout=30.0):
        rospy.loginfo("[TrajectoryActionServer] Executing BREAK CONTACT phase...")
        self.op_cmd.constraints = [1, 1, 1, 1, 1, 1]

        timed_out = False
        pose_tracked = False
        prev_time = 0.0
        self.rate.sleep()
        
        # initialize DMPs with current robot state
        self._set_dmp_init(elements)
        self.cs.reset()
        
        while not timed_out:
            # update system state
            current_time = rospy.get_time() - start_time
            dt = current_time - prev_time
            pose_error = self._pose_error()
            x = self.cs.get_value(dt, pose_error, 0.0, 0) # don't let contact FB affect it
            self._log[phase]['cs'].append(x)
            for element in elements.keys():
                dims = elements[element]
                if element == 'x':
                    self._populate_pose_from_dmps(phase, dims, x, dt, pose_error, 0.0, 0)
                elif element == 'cf':
                    self._populate_cf_from_dmps(dims, x, dt, pose_error, 0.0, use_cf, 0)
                else:
                    rospy.logerr("[TrajectoryActionServer] Unknown element type: %s" % element)
                    
            self.op_cmd.header.stamp = rospy.Time.now()
            self.op_cmd_pub.publish(self.op_cmd)

            # TODO position error since change in goal for quaternion is not totally working yet.
            if pose_tracked == False and self._goal_pose_error(orientation=False) < pose_tolerance:
                rospy.loginfo("[TrajectoryActionServer] Desired pose achieved for BREAK CONTACT phase.")
                return True
            
            prev_time = current_time
            timed_out = current_time > timeout
            self.rate.sleep()

        if timed_out:
            rospy.logwarn("[TrajectoryActionServer] Timed out executing BREAK CONTACT phase!")
            return False
        else:
            return True

    def _adapt_make_contact(self, phase, elements, start_time, cf=True, epsilon=0.00005, timeout=20.0):
        timed_out = False
        prev_time = 0.0
        self.rate.sleep()
        
        if not self.in_contact:
            rospy.loginfo("[TrajectoryActionServer] Contact expected but not yet in contact.")
            rospy.loginfo("[TrajectoryActionServer] Trying to make contact...")

            # figure out which dimensions should be making contact
            contact_dmps = []
            for i in range(3):
                if self.goal_mask[i] > 0:
                    contact_dmps.append(elements['x'][str(i)])
            if not contact_dmps:
                rospy.logerr("[TrajectoryActionServer] No contact DMPs for MAKING CONTACT phase.")
                return False
            
            # shift the goal by small increments until contact is made
            while not self.in_contact and not timed_out:
                current_time = rospy.get_time() - start_time
                dt = current_time - prev_time
                for dmp in contact_dmps:
                    dmp.set_current_goal(dmp.get_current_goal() - epsilon)
                    pose_error = self._pose_error()
                    wrench_mag = self._get_wrench_mag(self.robot_state.wrench) # TODO magnitude, or per-d?
                    x = self.cs.get_value(dt, pose_error, wrench_mag, self.in_contact)
                    for element in elements.keys():
                        dims = elements[element]
                        if element == 'x':
                            self._populate_pose_from_dmps(phase, dims, x, dt, pose_error, wrench_mag,
                                                          self.in_contact)
                        elif element == 'cf':
                            self._populate_cf_from_dmps(dims, x, dt)
                        else:
                            rospy.logerr("[TrajectoryActionServer] Unknown element type: %s" % element)
                self.op_cmd.header.stamp = rospy.Time.now()
                self.op_cmd_pub.publish(self.op_cmd)
                self.in_contact = self.classifier.in_contact()
                prev_time = current_time
                timed_out = current_time > timeout
                self.rate.sleep()

        if timed_out:
            rospy.logwarn("[TrajectoryActionServer] Timed out adapting MAKE CONTACT phase!")
            return False
        
        rospy.loginfo("[TrajectoryActionServer] In contact!")
        rospy.loginfo("[TrajectoryActionServer] Trying to track forces...")

        if cf:
            # set constraint frame and desired magnitude 
            self.op_cmd.constraint_frame = Pose()
            self.op_cmd.constraint_frame.orientation.x = self.make_contact_cf[0]
            self.op_cmd.constraint_frame.orientation.y = self.make_contact_cf[1]
            self.op_cmd.constraint_frame.orientation.z = self.make_contact_cf[2]
            self.op_cmd.constraint_frame.orientation.w = self.make_contact_cf[3]
            self.op_cmd.constraints = [1, 1, 0, 1, 1, 1]
            self.op_cmd.wrench = Wrench()
            self.op_cmd.wrench.force.z = self.make_contact_mag        
            self.op_cmd.desired_in_cf = True
        else:
            # TODO for now just assume desired wrench is in z direction for making contact with table
            self.op_cmd.wrench = Wrench()
            self.op_cmd.wrench.force.z = self.goal_wrench[2]
            self.op_cmd.constraints = [1, 1, 0, 1, 1, 1]
            self.op_cmd.desired_in_cf = False

        while not timed_out:
            current_time = rospy.get_time() - start_time
            self.op_cmd.header.stamp = rospy.Time.now()
            self.op_cmd_pub.publish(self.op_cmd)
            if self._wrench_error() < self.wrench_tolerance:
                rospy.loginfo("[TrajectoryActionServer] Forces tracking!")
                return True
            prev_time = current_time
            timed_out = current_time > timeout
            self.rate.sleep()
            
        if timed_out:
            rospy.logwarn("[TrajectoryActionServer] Timed out adapting MAKE CONTACT phase!")
            return False
        else:
            return True

    def _adapt_in_contact(self):
        # TODO not sure if this is needed yet
        pass
        
    def _get_dmps(self, dmp_configs):
        phase_dmps = {}
        for dmp_config in dmp_configs:
            phase_name = dmp_config.phase_name
            element_name = dmp_config.element_name
            dim = dmp_config.dimension
            if phase_name not in phase_dmps.keys():
                phase_dmps[phase_name] = {}
            if element_name not in phase_dmps[phase_name].keys():
                phase_dmps[phase_name][element_name] = {}

            w = self._get_weights(dmp_config.weight_vectors)
            dmp_init = self._get_init(dmp_config.init)
            dmp_goal = self._get_goal(dmp_config.goal)            
            dmp = DynamicMovementPrimitive(demos=[],
                                           name="%s_%s_%s" % (phase_name, element_name, str(dim)),
                                           cs=self.cs,
                                           quaternion=(dim == 'rot'),
                                           init=dmp_init,
                                           goal=dmp_goal,
                                           w=w,
                                           tau=dmp_config.tau,
                                           alpha=dmp_config.alpha,
                                           beta=dmp_config.beta,
                                           gamma=dmp_config.gamma,
                                           alpha_c=dmp_config.alpha_c,
                                           alpha_nc=dmp_config.alpha_nc,
                                           alpha_p=dmp_config.alpha_p,
                                           alpha_f=dmp_config.alpha_f,
                                           num_bfs=dmp_config.num_bfs,
                                           dt=dmp_config.dt)
            phase_dmps[phase_name][element_name][dim] = dmp
            # TODO kind of a hack
            if dim in ['0', '1', '2']:
                self.goal_wrench[int(dim)] = dmp_config.force_goal
                # self.goal_mask[int(dim)] = (abs(dmp_config.force_goal) > 0)
                # TEMPORARY when using world frame
                self.goal_mask[2] = 1.0
                self.make_contact_cf = dmp_config.make_contact_cf
                self.make_contact_mag = dmp_config.make_contact_mag

        # need to make sure each phase commands at least pose, otherwise bad things ensue
        for phase in phase_dmps.keys():
            if 'x' not in phase_dmps[phase].keys():
                rospy.logerr("[TrajectoryActionServer] No pose DMP for '%s'" % phase)
                return None

        return phase_dmps

    def _populate_pose_from_dmps(self, phase, dims, x, dt, pose_error=None, wrench_mag=None, in_contact=0):
        current_state = self.robot_state
        for dim in dims.keys():
            dmp = dims[dim]
            current_goal = dmp.get_current_goal()
            val, val_dot, val_dotdot = dmp.get_values(x, dt, pose_error, wrench_mag, in_contact)
            # need to continually update goal since it can change dynamically
            if dim == '0':
                self.goal_pose.position.x = current_goal
                self.op_cmd.pose.position.x = val
                self.op_cmd.twist.linear.x = val_dot
                self.op_cmd.accel.linear.x = val_dotdot
            elif dim == '1':
                self.goal_pose.position.y = current_goal
                self.op_cmd.pose.position.y = val
                self.op_cmd.twist.linear.y = val_dot
                self.op_cmd.accel.linear.y = val_dotdot
            elif dim == '2':
                self.goal_pose.position.z = current_goal
                self.op_cmd.pose.position.z = val
                self.op_cmd.twist.linear.z = val_dot
                self.op_cmd.accel.linear.z = val_dotdot
            elif dim == 'rot':
                self.goal_pose.orientation.x = current_goal[0]
                self.goal_pose.orientation.y = current_goal[1]
                self.goal_pose.orientation.z = current_goal[2]
                self.goal_pose.orientation.w = current_goal[3]
                self.op_cmd.pose.orientation.x = val[0]
                self.op_cmd.pose.orientation.y = val[1]
                self.op_cmd.pose.orientation.z = val[2]
                self.op_cmd.pose.orientation.w = val[3]
            else:
                rospy.logerr("[TrajectoryActionServer] Unknown pose dimension: %s" % str(dim))
                continue
            # log everything
            state = dmp.get_current_state()
            self._log[phase]['goal']['x'][dim].append(current_goal)
            self._log[phase]['val']['x'][dim].append(state['y'])
            self._log[phase]['val_dot']['x'][dim].append(state['dy'])
            self._log[phase]['val_dotdot']['x'][dim].append(state['ddy'])
            self._log[phase]['fb']['x'][dim].append(state['fb'])

    def _populate_wrench_from_dmps(self, phase, dims, x, dt, pose_error, wrench_mag, in_contact=0):
        for dim in dims.keys():
            dmp = dims[dim]
            w = dmp.get_values(x, dt, pose_error, wrench_mag, in_contact)[0]
            if dim == '0':
                self.op_cmd.wrench.force.x = w
            elif dim == '1':
                self.op_cmd.wrench.force.y = w
            elif dim == '2':
                self.op_cmd.wrench.force.z = w
            elif dim == '3':
                self.op_cmd.wrench.torque.x = w
            elif dim == '4':
                self.op_cmd.wrench.torque.y = w
            elif dim == '5':
                self.op_cmd.wrench.torque.z = w
            else:
                rospy.logerr("[TrajectoryActionServer] Unknown wrench dimension: %s" % str(dim))
                continue

    def _populate_cf_from_dmps(self, dims, x, dt, pose_error=None, wrench_mag=None, use_cf=True, in_contact=0):
        for dim in dims.keys():
            dmp = dims[dim]
            cmd_value = dmp.get_values(x, dt)[0]

            if dim == 'mag':
                cmd_value = max(cmd_value, 1.0)  # never command negative in constraint frame
                cmd_value = min(cmd_value, 20.0) # saturate to some max value
                self.op_cmd.wrench = Wrench()
                self.op_cmd.wrench.force.z = cmd_value
            elif dim == 'rot':
                if use_cf:
                    self.op_cmd.constraint_frame = Pose()
                    self.op_cmd.constraint_frame.orientation.x = cmd_value[0]
                    self.op_cmd.constraint_frame.orientation.y = cmd_value[1]
                    self.op_cmd.constraint_frame.orientation.z = cmd_value[2]
                    self.op_cmd.constraint_frame.orientation.w = cmd_value[3]
            else:
                rospy.logerr("[TrajectoryActionServer] Unknown element for constraint frame: %s" % dim)
                
    def _pose_error(self, orientation=False, dim=None):
        if dim is not None:
            return abs(self.pose_error[dim])
        elif orientation:
            return np.linalg.norm(self.pose_error)
        else:
            return np.linalg.norm(self.pose_error[:3])

    def _wrench_error(self):
        return np.linalg.norm(self.goal_mask * self.wrench_error)

    def _goal_pose_error(self, orientation=False, dim=None):
        # TODO this is hacked, for isolating dims in experiments
        if dim is not None:
            if dim == 0:
                return np.linalg.norm(self.robot_state.pose.position.x - self.goal_pose.position.x)
            elif dim == 1:
                return np.linalg.norm(self.robot_state.pose.position.y - self.goal_pose.position.y)
            elif dim == 2:
                return np.linalg.norm(self.robot_state.pose.position.z - self.goal_pose.position.z)
        else:
            return self._cartesian_error(self.robot_state.pose, self.goal_pose, orientation)
        
    def _cartesian_error(self, actual, desired, orientation=True):
        p1 = actual.position
        p2 = desired.position
        p_err = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])

        if orientation:
            oa = actual.orientation
            od = desired.orientation
            q1 = np.array([oa.x, oa.y, oa.z, oa.w])
            q2 = np.array([od.x, od.y, od.z, od.w])
            q_err = np.squeeze(quaternion.err(q1, q2))
            pose_error = np.hstack((p_err, q_err))
        else:
            pose_error = np.hstack((p_err, np.zeros(3)))
        return np.linalg.norm(pose_error)
        
    def _joint_error(self, actual, desired, norm=False):
        a = np.array(actual)
        d = np.array(desired)
        err = d - a
        if norm:
            err = np.linalg.norm(err)
        return err

    def _robot_state_cb(self, robot_state):        
        # if this is the first time getting the state, set it also as the desired
        if self.robot_state is None:
            self.op_cmd.pose = robot_state.pose
            rospy.loginfo("[TrajectoryActionServer] Robot state received.")
        self.robot_state = robot_state

        self.pose_error = self.robot_state.pose_error
        self.wrench_error = np.array(self.robot_state.wrench_error)
                
        # update the force window for classifer
        f = robot_state.wrench.force
        t = robot_state.wrench.torque
        self.current_wrench = np.array([f.x, f.y, f.z, t.x, t.y, t.z])
        self.classifier.update_window(self.current_wrench)

        # set goal to current if none has been received yet, in case a dim is missing
        if self.goal_pose is None:
            self.goal_pose = Pose()
            self.goal_pose.position = self.robot_state.pose.position
            self.goal_pose.orientation = self.robot_state.pose.orientation

    def _jnt_state_cb(self, jnt_state):
        # if this is the first time, set current to desired
        if self.jnt_state is None:
            self.jnt_cmd.position = jnt_state.position
            rospy.loginfo("[TrajectoryActionServer] Joint state received.")
        self.jnt_state = jnt_state

    def _set_dmp_init(self, elements):
        for element in elements.keys():
            dims = elements[element]
            if element == 'x':
                p = self.robot_state.pose
                for dim in dims.keys():
                    dmp = dims[dim]
                    if dim == '0':
                        dmp.set_init(p.position.x)
                    elif dim == '1':
                        dmp.set_init(p.position.y)
                    elif dim == '2':
                        dmp.set_init(p.position.z)
                    elif dim == 'rot':
                        dmp.set_init(np.array([p.orientation.x,
                                               p.orientation.y,
                                               p.orientation.z,
                                               p.orientation.w]))
                    else:
                        rospy.logerr("[TrajectoryActionServer] Unknown dim type: %s" % str(dim))
                    dmp.ts.reset() # so it registers new init
            # TODO need to handle case also for constraint frame force
            elif element in ['cf']:
                pass
            else:
                rospy.logerr("[TrajectoryActionServer] Unknown element type: %s" % element)

        
    def _get_wrench_mag(self, msg):
        force = np.array([msg.force.x, msg.force.y, msg.force.z])
        torque = np.array([msg.torque.x, msg.torque.y, msg.torque.z])
        return np.linalg.norm(np.hstack((force, torque)))

    def _get_weights(self, config_weights):
        weights = None
        for weight_vector in config_weights:
            if weights is None:
                weights = np.array(weight_vector.weights)
            else:
                weights = np.vstack((weights, np.array(weight_vector.weights)))
        return weights

    def _get_init(self, config_init):
        if len(config_init) > 1:
            init = np.array(config_init)
        else:
            init = config_init[0]
        return init

    def _get_goal(self, config_goal):
        if len(config_goal) > 1:
            goal = np.array(config_goal)
        else:
            goal = config_goal[0]
        return goal

    def _set_h5_path(self, path):
        self.h5_path = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(self.h5_path):
            print "Path does not exist: %s" % self.h5_path
        
    def _set_h5_abspath(self, h5_filename):
        self.h5_abspath = os.path.join(self.h5_path, h5_filename)
        if not os.path.exists(self.h5_abspath):
            print "File does not exist: %s" % self.h5_abspath

    def _log_data(self):
        if self._log.keys():
            h5_file = h5py.File(os.path.join(self.h5_path, "%s_dmp.h5" % self.log_filename), 'a')
            
            grp = h5_file.create_group("dmp_%d" % self.log_idx)
            grp.attrs['commanded_trajectory'] = self.commanded_trajectory

            rospy.loginfo("\n\n\n[TrajectoryActionClient] Logging data to: %s/%s/dmp_%d\n\n\n"
                          % (self.h5_path, self.log_filename + "_dmp.h5", self.log_idx))

            for phase in self._log.keys():
                for kind in self._log[phase]:
                    if kind == 'cs':
                        dset = grp.create_dataset("cs", data=self._log[phase]['cs'])
                    elif kind == 'goal':
                        for element in self._log[phase][kind].keys():
                            gs = np.array([])
                            for dim in self._log[phase]['goal'][element].keys():
                                # TODO ignoring for now since it won't be changing and it needs a fixed implementation
                                if dim != 'rot':
                                    dim_gs = np.array(self._log[phase]['goal'][element][dim])
                                    gs = np.vstack((gs, dim_gs)) if gs.size else dim_gs
                            dset = grp.create_dataset("goal/%s" % element, data=gs)
                    elif kind == 'fb':
                        for element in self._log[phase][kind].keys():
                            fbs = np.array([])
                            for dim in self._log[phase]['fb'][element].keys():
                                # TODO ignoring for now since it won't be changing and it needs a fixed implementation
                                if dim != 'rot':
                                    dim_fbs = np.array(self._log[phase]['fb'][element][dim])
                                    fbs = np.vstack((fbs, dim_fbs)) if fbs.size else dim_fbs
                            dset = grp.create_dataset("fb/%s" % element, data=fbs)
                        

                # TODO log other elements if desired (e.g. wrench goal)
                            
            rospy.loginfo("Saved DMP data to %s/%s/%s_%d" % (self.h5_path, self.log_filename + "_dmp.h5", "dmp", self.log_idx))
            h5_file.close()
                
                
if __name__ == '__main__':
    server = TrajectoryActionServer()
    rospy.on_shutdown(server._log_data)
    try:
        server.start()
    except rospy.ROSInterruptException:
        pass
        
