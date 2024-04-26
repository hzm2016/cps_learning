#!/usr/bin/env python
import rospy
import numpy as np
from actionlib import SimpleActionServer
from sensor_msgs.msg import JointState
from ll4ma_movement_primitives.util import quaternion
from ll4ma_movement_primitives.phase_variables import LinearPV
from ll4ma_movement_primitives.basis_functions import GaussianLinearBFS
from ll4ma_movement_primitives.promps import ProMP, ProMPConfig, ProMPLibrary, Waypoint
from ll4ma_movement_primitives.msg import ProMPPolicyAction, ProMPPolicyFeedback, ProMPPolicyResult
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from ll4ma_logger_msgs.msg import RobotState
from std_srvs.srv import SetBool, SetBoolResponse
from rospy_service_helper import zero_reflex_tactile, set_controller_mode


class ProMPActionServer:

    def __init__(self, joint_tolerance = 0.01):
        self.joint_tolerance = joint_tolerance
        self.rate_val = 100 # TODO get from server, set appropriately
        self.timeout = 20.0
        self.rate = rospy.Rate(self.rate_val)
        self.duration = 10.0 # seconds, TODO get from server
        self.ns = "/promp_action_server"
        self.server = SimpleActionServer(self.ns, ProMPPolicyAction, self._goal_cb, False)
        self.feedback = ProMPPolicyFeedback()
        self.result = ProMPPolicyResult()
        self.robot_state = None
        self.jnt_names = []
        self.jnt_cmd = JointState()
        self.virtual_jnt_cmd = JointState()
        self.review_status = False
        self.policy_start = None

        # Services that this node will call
        self.srvs = {
            "set_cmd_mode"        : "/lwr_tau_control/set_command_mode",
            "zero_reflex_tactile" : "/reflex_takktile/calibrate_tactile",
            "set_recording"       : "/rosbag/set_recording",
            "set_jnt_traj"        : "/lbr4/set_joint_trajectory",
            "review_jnt_traj"     : "/lbr4/review_joint_trajectory"
        }

        # Topics this node needs access to (TODO can pull from param server instead of hard-coded)
        self.topics = {
            "jnt_des_cmd"     : "/lbr4/joint_cmd",
            "robot_state"     : "/lbr4/robot_state",
            "virtual_jnt_cmd" : "/virtual_lbr4/virtual_joint_cmd"
        }

        # Make sure services that will be called are ready to receive requests
        rospy.loginfo("Waiting for services...")
        for srv in self.srvs.keys():
            rospy.loginfo("    %s" % self.srvs[srv])
            rospy.wait_for_service(self.srvs[srv])
        rospy.loginfo("Services are up!")        
                            
        # Publishers
        self.jnt_des_cmd_pub = rospy.Publisher(self.jnt_des_cmd_topic, JointState, queue_size=1)
        self.virtual_jnt_cmd_pub = rospy.Publisher(self.virtual_jnt_cmd_topic, JointState,
                                                   queue_size=1)
        # Subscribers
        rospy.Subscriber(self.robot_state_topic, RobotState, self._robot_state_cb)

        # Services being offered
        # self.set_promp_srv = rospy.Service(self.ns + "/set_promp", ReviewProMPExecution,
        #                                    self.set_promp)
        self.set_review_status_srv = rospy.Service(self.ns + "/set_review_status", SetBool,
                                                   self.set_review_status)
        

    def start_server(self):
        rospy.loginfo("Server is running. Waiting for a goal...")
        self.server.start()
        while not rospy.is_shutdown():
            self.rate.sleep()

    def stop(self):
        rospy.loginfo("Server shutdown. Exiting.")


    # === BEGIN service functions being offered ===================================================

    # def set_promp(self, req):
    #     resp = ReviewProMPExecutionResponse()

    #     # Create ProMP from configuration
    #     promp_config = self._ros_to_python_config(req.config)
    #     # TODO for now just creating one ProMP, since I don't know how up to date library is
    #     # library = self._get_promp_library(promp_config, duration=req.duration)
    #     # phase_var = LinearPV()
    #     # phase_var.max_time = duration
    #     # phase_var.reset()
    #     # bfs = GaussianLinearBFS(num_bfs=promp_config.num_bfs)
    #     # bfs.max_time = duration
    #     # bfs.reset()
    #     self.promp = ProMP(config=promp_config)
    #     # promp.phase = phase_var
    #     # promp.bfs = bfs

    #     # Condition on waypoints
    #     waypoints = [self._ros_to_python_waypoint(ros_waypoint) for ros_waypoint in req.waypoints]
    #     self.promp.update_w(waypoints)

    #     # Generate a joint trajectory
    #     promp_traj, dist = self.promp.generate_trajectory(dt=req.dt, duration=req.duration)
    #     j_traj = self._joint_traj_from_promp_traj(promp_traj)
    #     self.policy_start = j_traj.points[0].positions

    #     # Set trajectory on trajectory action server
    #     success = self.set_joint_trajectory(self.srvs["set_jnt_traj"], j_traj)
    #     if not success:
    #         resp.success = False
    #         return resp

    #     resp.success = True
    #     return resp

    def set_review_status(self, req):
        self.review_status = True
        resp = SetBoolResponse()
        resp.success = True
        return resp
    
    # === END service functions being offered =====================================================


    def _goal_cb(self, action_goal):
        rospy.loginfo("New ProMP action goal received.")
        self.result.success = False

        if not self.review_status:
            rospy.logerr("No ProMP has been reviewed! Review through GUI before execution.")
            self.server.set_aborted()
            return False

        if not self.policy_start:
            rospy.logerr("No starting point for the policy has been set.")
            self.server.set_aborted()
            return False
        
        if self.robot_state is None or not self.robot_state.lbr4.joint_state.position:
            rospy.logwarn("Robot state is unknown. Waiting for 10 seconds...")
            i = 0
            while i < 100 and self.robot_state is None:
                rospy.sleep(0.1)
                i += 1
        # Don't try to run trajectory if you don't know where the robot is
        if self.robot_state is None:
            self.server.set_aborted()
            rospy.logerr("GOAL ABORTED. Current robot state unknown.")
            return False
        else:
            start_error = self._joint_error(self.robot_state.lbr4.joint_state.position,
                                            self.policy_start, norm=True)
            at_start = start_error < self.joint_tolerance
            if not at_start:
                rospy.logerr("GOAL ABORTED. Too far from start. Run MTS from GUI.")
                self.server.set_aborted()
                return False
            else:
                # Run the policy
                self._execute_policy(action_goal)

                # Report results to console
                if self.result.success:
                    rospy.loginfo("GOAL REACHED.")
                    self.server.set_succeeded(self.result)
                else:
                    rospy.logwarn("FAILURE. Goal was not reached.")
                    self.server.set_aborted()

                    
    def _execute_policy(self, goal):
        rospy.loginfo("Executing the policy...")

        # If for some reason we are now not at the start, just bail
        start_error = self._joint_error(self.robot_state.lbr4.joint_state.position,
                                        self.policy_start, norm=True)
        at_start = start_error < self.joint_tolerance
        if not at_start:
            rospy.logerr("Not at start position!")
            self.result.success = False
            return False

        # # Get ready for exeuction
        phase_var.reset()
        start_time = rospy.get_time()
        timed_out = False
        goal_converged = False
        prev_time = 0.0
        self.rate.sleep()

        # Zero out the ReFlex tactile sensors
        success = self.zero_reflex_tactile(self.srvs["zero_reflex_tactile"])

        # TODO need to set up rosbag for logging this stuff to the right place with the right
        # filenames
        
        # # Start rosbag recorder
        # success = self._set_rosbag_recorder_status(run=True)


        # TODO set controller in command mode
        
                
        while not goal_converged and not timed_out:
            # Update system state
            current_time = rospy.get_time() - start_time
            dt = current_time - prev_time
            
            phase = phase_var.get_value(dt)
            block_phi, cmd_keys = library.get_block_phi(phase)

            # TODO this is library
            
            cmd = np.dot(block_phi, library.get_w("default"))
            self._execute_cmd(cmd, cmd_keys)
            
            # Check timeout
            timed_out = current_time > self.timeout
            prev_time = current_time

            # Check goal convergence
            goal_error = self._joint_error(self.robot_state.lbr4.joint_state.position,
                                           self._goal.positions, norm=True)
            goal_converged = goal_error < self.joint_tolerance

            self.rate.sleep()

        # # Stop rosbag recorder
        # success = self._set_rosbag_recorder_status(run=False)
        
        # # Take controller out of command mode
        # success = self.set_controller_mode(is_command=False)
        
        rospy.loginfo("ProMP execution complete.")
        
        # Report result and set status
        if goal_converged:
            rospy.loginfo("Converge to goal!")
            self.result.success = True
            return True
        elif timed_out:
            rospy.logwarn("Execution timed out!")
            self.result.success = False
            return False
        else:
            rospy.logwarn("Did not converge to goal.")
            self.result.success = False
            return False
        self.review_status = False

    def _execute_cmd(self, cmd, keys):
        # TODO for now assuming this is ONLY joint space
        self.jnt_cmd.position = []
        self.jnt_cmd.velocity = []
        self.jnt_cmd.effort   = [0.0]*self.num_jnts # for now not commanding any acceleration

        # Populate the joint command with ProMP output
        keys = sorted(keys)
        for c, k in zip(cmd, keys):
            state_type, dim = k.split(".")
            if state_type == 'q':
                self.jnt_cmd.position.append(c)
            elif state_type == 'qdot':
                self.jnt_cmd.velocity.append(c)

        # Error check the command
        if (len(self.jnt_cmd.position) != self.num_jnts or
            len(self.jnt_cmd.velocity) != self.num_jnts or
            len(self.jnt_cmd.effort)   != self.num_jnts or
            len(self.jnt_cmd.name)     != self.num_jnts):
            rospy.logerr("Joint command does not match expected length.\n"
                         "    Num Joints: %d\n"
                         "     Name Size: %d\n"
                         "      Pos Size: %d\n"
                         "      Vel Size: %d\n"
                         "      Acc Size: %d\n" % (self.num_jnts,
                                                   len(self.jnt_names),
                                                   len(self.jnt_cmd.position),
                                                   len(self.jnt_cmd.velocity),
                                                   len(self.jnt_cmd.effort)))

        # Make sure we don't command too high change in position or velocity
        # TODO not sure what a good value is, 1rad is roughly 60deg per second. Seems like a
        # reasonably limit. Can change as necessary. Being conservative now and completely
        # bailing on motion if we command this, since it's indicative that either the threshold
        # is too conservative, or we've done something horribly wrong that shouldn't be sent
        # to robot.

        thresh = 1.0
        max_dev = max(self._joint_error(self.jnt_cmd.position, self.prev_joint_position))
        if max_dev > thresh:
            rospy.logerr("COMMANDING TOO LARGE change in joint position:\n"
                         "    Previous: %s\n"
                         "    Current:  %s" % (self.prev_joint_position, self.jnt_cmd.position))
            self.result.success = False
            return False

        if max(self.jnt_cmd.velocity) > thresh:
            rospy.logerr("COMMANDING TOO LARGE velocity: %s", self.jnt_cmd.velocity)
            self.result.success = False
            return False

        # Send joint command to robot
        self.jnt_cmd.header.stamp = rospy.Time.now()
        self.jnt_des_cmd_pub.publish(self.jnt_cmd)

        self.prev_joint_position = self.jnt_cmd.position

    def _pose_error(self, orientation=True, dim=None):
        if dim is not None:
            return abs(self.pose_error[dim])
        elif orientation:
            return np.linalg.norm(self.pose_error)
        else:
            return np.linalg.norm(self.pose_error[:3])
        
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
        if self.robot_state is None:
            # TODO only for KUKA
            self.num_jnts = len(robot_state.lbr4.joint_state.position)

            # Initialize joint command for virtual robot
            self.virtual_jnt_cmd.position = robot_state.lbr4.joint_state.position
            self.virtual_jnt_cmd.velocity = [0.0]*self.num_jnts
            self.virtual_jnt_cmd.effort   = [0.0]*self.num_jnts
            robot_joint_names = robot_state.lbr4.joint_state.name
            self.virtual_jnt_cmd.name = ["virtual_" + name for name in robot_joint_names]
            self.jnt_names = robot_state.lbr4.joint_state.name

            # Initialize the start point so we can directly utilize move to start functionality
            self._start = JointTrajectoryPoint()
            self._start.positions = robot_state.lbr4.joint_state.position
            rospy.loginfo("Robot state received:\n"
                          "    Num Joints: %d" % self.num_jnts)
        self.robot_state = robot_state

    def _ros_to_python_config(self, ros_config):
        promp_config = ProMPConfig()
        promp_config.state_types = ros_config.state_types
        promp_config.num_bfs     = ros_config.num_bfs
        promp_config.dt          = ros_config.dt
        promp_config.mu_w        = ros_config.mu_w
        promp_config.w_keys      = ros_config.w_keys
        # Unpack sigma_w
        sigma_w = None
        for row in ros_config.sigma_w.rows:
            elements = np.array(row.elements)
            sigma_w = elements if sigma_w is None else np.vstack((sigma_w, elements))
        promp_config.sigma_w = sigma_w
        # Unpack dimension lists
        for row in ros_config.dimensions.rows:
            promp_config.dimensions.append(row.elements)
        return promp_config

    def _ros_to_python_waypoint(self, ros_waypoint):
        waypoint = Waypoint()
        waypoint.values         = ros_waypoint.values
        waypoint.time           = ros_waypoint.time
        waypoint.phase_val      = ros_waypoint.phase_val
        waypoint.sigma          = ros_waypoint.sigma
        waypoint.condition_keys = ros_waypoint.condition_keys
        return waypoint

    def _joint_traj_from_promp_traj(self, promp_traj):
        # find number of points
        key1 = promp_traj.keys()[0]
        key2 = promp_traj[key1].keys()[0]
        num_pts = len(promp_traj[key1][key2])
        j_traj = JointTrajectory()
        for j in range(num_pts):
            j_point = JointTrajectoryPoint()
            for element in promp_traj.keys():
                if element == 'q':
                    for dim in promp_traj[element].keys():
                        j_point.positions.append(promp_traj[element][dim][j])
            j_traj.points.append(j_point)                            
        return j_traj

    def _get_promp_library(self, promp_config, duration=10.0):
        phase_var = LinearPV()
        phase_var.max_time = duration
        phase_var.reset()
        bfs = GaussianLinearBFS(num_bfs=promp_config.num_bfs)
        bfs.max_time = duration
        bfs.reset()

        library = ProMPLibrary(phase_var, bfs)
        promp = ProMP(config=promp_config)
        promp.phase = phase_var
        promp.bfs = bfs
        library.add_primitive(promp.name, promp)
        return library
                    
                
if __name__ == '__main__':
    rospy.init_node("promp_action_server")
    import sys
    argv = rospy.myargv(argv=sys.argv)
    try:
        server = ProMPActionServer(*argv[1:]) if len(argv) > 1 else ProMPActionServer()
        rospy.on_shutdown(server.stop)
        server.start_server()
    except rospy.ROSInterruptException:
        pass
