#!/usr/bin/env python
import rospy
from ll4ma_policy_learning.msg import Waypoint
from ll4ma_trajectory_util.srv import GetTrajectories, GetTrajectoriesRequest
from ll4ma_policy_learning.srv import (
    LearnProMP,
    LearnProMPRequest,
    ReviewProMPExecution,
    ReviewProMPExecutionRequest
)

get_traj_srv    = "/pandas/get_trajectories"
learn_promp_srv = "/policy_learner/learn_promps"
set_promp_srv   = "/promp_action_server/set_promp"


# === BEGIN service call functions =================================================================

def get_trajs(robot_name, traj_names):
    req = GetTrajectoriesRequest()
    req.robot_name = robot_name
    req.trajectory_names = traj_names
    req.subsample = 0.9
    success = False
    try:
        get_trajs = rospy.ServiceProxy(get_traj_srv, GetTrajectories)
        resp = get_trajs(req)
        success = resp.success
    except rospy.ServiceException as e:
        rospy.logerr("Service request to get trajectories failed: %s" %e)
        return None
    if not success:
        rospy.logerr("Could not get trajectories.")
        return None
    else:
        return resp.joint_trajectories, resp.task_trajectories

def learn_promps(joint_trajectories, task_trajectories):
    # Make sure they're the same length, not the best way to do this but they've been differing by 1
    for j_traj, t_traj in zip(joint_trajectories, task_trajectories):
        length = min(len(j_traj.points), len(t_traj.points))
        j_traj.points = j_traj.points[:length]
        t_traj.points = t_traj.points[:length]
    
    req = LearnProMPRequest()
    req.joint_trajectories = joint_trajectories
    req.task_trajectories = task_trajectories
    success = False
    try:
        learn = rospy.ServiceProxy(learn_promp_srv, LearnProMP)
        resp = learn(req)
        success = resp.success
    except rospy.ServiceException as e:
        rospy.logerr("Service request to learn ProMPs failed: %s" %e)
        return None
    if not success:
        rospy.logerr("Could not learn ProMPs.")
        return None
    else:
        return resp.config

def set_promp(config, waypoints=[], dt=0.1, duration=10.0):
    req = ReviewProMPExecutionRequest()
    req.config = config
    req.waypoints = waypoints
    req.dt = dt
    req.duration = duration
    success = False
    try:
        review = rospy.ServiceProxy(set_promp_srv, ReviewProMPExecution)
        resp = review(req)
        success = resp.success
    except rospy.ServiceException as e:
        rospy.logerr("Service request to set ProMP failed: %s" %e)
        return False
    if not success:
        rospy.logerr("Could not set ProMP.")
        return False
    else:
        rospy.loginfo("Successfully set ProMP.")
        return True

# === END service call functions ===================================================================

    
def parse_args(args):
    # TODO change args as necessary
    import argparse
    parser = argparse.ArgumentParser()
    # def tup(a):
    #     try:
    #         idx, num_demos = map(int, a.split(','))
    #         return idx, num_demos
    #     except:
    #         raise argparse.ArgumentTypeError("Must be idx,num_demos")
    parser.add_argument('-d', nargs='+', dest='demos', type=int)
    parser.add_argument('-g', dest='goal', type=int)
    parser.add_argument('-m', dest='midpoint', type=int)
    parsed_args = parser.parse_args(args)
    return parsed_args

def get_session_config(config_filename):
    import rospkg
    import os
    import yaml
    r = rospkg.RosPack()
    config_file = os.path.join(r.get_path("ll4ma_policy_learning"), "config/" + config_filename)
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    return config

def main():
    rospy.init_node("visualize_promp_session")
    import sys
    import numpy as np

    # Make sure services are running
    rospy.loginfo("Waiting for services...")
    rospy.wait_for_service(get_traj_srv)
    rospy.loginfo("Services are up!")

    # Get whatever options were passed in from command line
    args = parse_args(sys.argv[1:])

    # Get the configuration parameters for this execution
    session_config = get_session_config("table_experiment.yaml")

    # Get trajectory data from data service
    trajectory_names = []
    if args.demos:
        for group in args.demos:
            trajectory_names += session_config[group]['trajectories']
    joint_trajs, task_trajs = get_trajs("lbr4", trajectory_names)
    
    # Learn ProMPs from the data and get the configs
    promp_config = learn_promps(joint_trajs, task_trajs)

    # Get waypoints to add to ProMP config
    waypoints = []
    if args.goal:
        g = session_config[args.goal]['goal']
        wpt = Waypoint()
        wpt.phase_val = 1.0
        wpt.condition_keys = ['x.0', 'x.1', 'x.2']
        wpt.values         = [g['x.0'], g['x.1'], g['x.2']]
        waypoints.append(wpt)

    # waypoints = []
    # wpt = Waypoint()
    # wpt.phase_val = 1.0
    # # wpt.condition_keys = ['q.%d' % idx for idx in range(7)]
    # # wpt.values         = [-1.5790920296410227,
    # #                       -1.1145983759279907,
    # #                       -0.0037967540805405,
    # #                       1.3028029628501825,
    # #                       0.0468973770693989,
    # #                       0.7403954682405864,
    # #                       -1.3632104028216303]
    # wpt.condition_keys = ['x.0', 'x.1', 'x.2']
    # wpt.values = [0.054107919125,  0.0832297702007, 0.74654216]
    # # wpt.values = [-0.0350762773956, -0.78140111403, 0.171366855461]
    # waypoints.append(wpt)
    
    # Send configs to action server for review
    success = set_promp(promp_config, waypoints, duration=10.0, dt=0.1)
    
    
if __name__ == '__main__':
    main()
