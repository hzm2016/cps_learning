#!/usr/bin/env python
import os
import sys
import rospy
import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from std_srvs.srv import Trigger, Empty
from policy_action_lib import DMPActionClient, ProMPActionClient
from ll4ma_policy_learning.srv import PolicyExecution, PolicyExecutionResponse
from ll4ma_trajectory_msgs.srv import RobotTrajectory, RobotTrajectoryRequest
from policy_interface import PolicyLearner


class PolicyCommander:

    def __init__(self, robot_name="robot", log_namespace="h5"):
        self.robot_name = robot_name
        self.log_namespace = log_namespace
        rospy.init_node("%s_policy_commander" % self.robot_name)
        rospy.loginfo("Waiting for data service...")
        rospy.wait_for_service(self.log_namespace + "/get_trajectory_data")
        rospy.loginfo("Data service found!")
        self.command_policy_srv = rospy.Service(
            "/%s/policy_commander/command_policy" % self.robot_name,
            PolicyExecution, self.command_policy)
        self.action_clients = {}
        self.action_clients["dmp"] = DMPActionClient(robot_name=self.robot_name, rospy_init=False)
        self.action_clients["promp"] = ProMPActionClient(robot_name=self.robot_name, rospy_init=False)
        self.learner = PolicyLearner()
        self.cached_data = {}

    def command_policy(self, req):
        if req.policy_type == "promp":
            self.command_promp(req)
        elif req.policy_type == "dmp":
            self.command_dmp(req)
        else:
            rospy.loginfo("Unknown policy type: %s" % req.policy_type)
        resp = PolicyExecutionResponse()
        resp.success = True # TODO set based on result of execution
        return resp

    def command_promp(self, req):
        if req.demo_names:
            demo_names = req.demo_names
        else:
            config_file = os.path.expanduser(os.path.join(req.path, req.filename))
            demo_names = []
            with open(config_file, 'r') as f:
                demo_names = f.read().splitlines()

        data = {}
        for demo_name in demo_names:
            if demo_name not in self.cached_data.keys():
                self.cached_data[demo_name] = self.get_trajectory_data(demo_name, req.is_joint_space)
            data[demo_name] = self.cached_data[demo_name]
        promp_configs = [self.learner.learn_promps(data)] # TODO for now assuming 1 coupled ProMP
        self.action_clients["promp"].send_goal(promp_configs, req.num_executions, req.is_joint_space)
                
    def command_dmp(self, req):
        # TODO for now assume only learning from a single demonstration
        demo_name = req.demo_names[0]
        data = self.get_trajectory_data(demo_name, req.is_joint_space)
        if data is not None:
            rospy.loginfo("Commanding DMP policy.")
            dmp_configs = self.learner.learn_dmps(data)
            self.action_clients["dmp"].send_goal(dmp_configs)
        else:
            rospy.logwarn("Trajectory data could not be loaded.")

    def get_trajectory_data(self, name, is_joint_space=True):
        data = {}
        try:
            get_data = rospy.ServiceProxy(self.log_namespace + "/get_trajectory_data", RobotTrajectory)
            req = RobotTrajectoryRequest()
            req.name = name
            req.is_joint_space = is_joint_space
            resp = get_data(req)
        except rospy.ServiceException, e:
            rospy.logwarn("Service request failed: %s" % e)
            return None
        # unpack the data
        if resp.joint_trajectory.points:
            qs = dqs = ddqs = taus = None
            contacts = [] # TODO it's only 1-D binary right now, can make 6-D if needed
            for point in resp.joint_trajectory.points:
                if point.positions:
                    q = np.array(point.positions)[:,None]
                    qs =  q if qs is None else np.hstack((qs, q))
                if point.velocities:
                    dq = np.array(point.velocities)[:,None]
                    dqs = dq if dqs is None else np.hstack((dqs, dq))
                # if point.accelerations:
                #     ddq = np.array(point.accelerations)[:,None]
                #     ddqs = ddq if ddqs is None else np.hstack((ddqs, ddq))
                # if point.efforts:
                #     tau = np.array(point.effort)[:,None]
                #     taus = tau if taus is None else np.hstack((taus, tau))
                contacts.append(point.contact)
            data['q']       = qs
            data['qdot']    = dqs
            # data['ddq']     = ddqs
            # data['tau']     = taus
            data['contact'] = np.array(contacts).reshape((1, len(contacts)))
        elif resp.task_trajectory.points:
            xs = dxs = None
            contacts = [] # TODO it's only 1-D binary right now, can make 6-D if needed
            for point in resp.task_trajectory.points:
                # pose
                x = []
                x.append(point.pose.position.x)
                x.append(point.pose.position.y)
                x.append(point.pose.position.z)
                x.append(point.pose.orientation.x)
                x.append(point.pose.orientation.y)
                x.append(point.pose.orientation.z)
                x.append(point.pose.orientation.w)
                x = np.array(x).reshape((7,1))
                xs = x if xs is None else np.hstack((xs, x))
                # twist
                dx = []
                dx.append(point.twist.linear.x)
                dx.append(point.twist.linear.y)
                dx.append(point.twist.linear.z)
                dx.append(point.twist.angular.x)
                dx.append(point.twist.angular.y)
                dx.append(point.twist.angular.z)
                dx = np.array(dx).reshape((6,1))
                dxs = dx if dxs is None else np.hstack((dxs, dx))
                # binary contact
                contacts.append(point.contact)
            data['x']       = xs
            data['xdot']    = dxs
            data['contact'] = np.array(contacts).reshape((1,len(contacts)))
        else:
            rospy.loginfo("No trajectory data was returned.")
        return data

    def run(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            rate.sleep()

    def _get_goal_poses_from_data(self, data):
        goal_poses = []
        for demo_name in data.keys():
            x = data[demo_name]['x'][:,-1]
            pose = Pose()
            pose.position.x    = x[0]
            pose.position.y    = x[1]
            pose.position.z    = x[2]
            pose.orientation.x = x[3]
            pose.orientation.y = x[4]
            pose.orientation.z = x[5]
            pose.orientation.w = x[6]
            goal_poses.append(pose)
        return goal_poses

    
if __name__ == '__main__':
    argv = rospy.myargv(argv=sys.argv)
    commander = PolicyCommander(*argv[1:]) if len(argv) > 1 else PolicyCommander()
    commander.run()
