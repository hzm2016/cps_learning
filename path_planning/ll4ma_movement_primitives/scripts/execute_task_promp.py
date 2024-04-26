#!/usr/bin/env python
import rospy
from ll4ma_policy_learning.msg import Waypoint
from rospy_service_helper import get_task_trajectories, learn_promps, set_promp

srvs = {
    "get_task_trajs" : "/pandas/get_task_trajectories",
    "learn_promp"    : "/policy_learner/learn_promps",
    "set_promp"      : "/task_promp_action_server/set_promp"
}




def parse_args(args):
    # TODO change args as necessary
    import argparse
    parser = argparse.ArgumentParser()
    def tup(a):
        return map(int, a.split(','))
    parser.add_argument('config_file', type=str)
    parser.add_argument('-d', nargs='+', dest='data_specs', type=tup)
    parsed_args = parser.parse_args(args)
    return parsed_args

def get_session_config(config_rel_path, data_path="~/.rosbags"):
    import os
    import yaml
    config_file = os.path.expanduser(os.path.join(data_path, config_rel_path))
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    return config
    


if __name__ == '__main__':
    rospy.init_node("execute_task_promp")
    import sys
    import numpy as np

    # Make sure services are running
    rospy.loginfo("Waiting for services...")
    for key in srvs.keys():
        rospy.loginfo("    %s" % srvs[key])
        rospy.wait_for_service(srvs[key])
    rospy.loginfo("Services are up!")

    # Get whatever options were passed in from command line
    argv = rospy.myargv(argv=sys.argv)
    args = parse_args(argv[1:])

    # Get the configuration parameters for this execution
    session_config = get_session_config(args.config_file)

    # Get trajectory data from data service
    trajectory_names = []
    if args.data_specs:
        for group, num_demos in args.data_specs:
            trajectory_names += session_config[group]['trajectories'][:num_demos]
    else:
        for group in session_config.keys():
            trajectory_names += session_config[group]['trajectories']
    task_trajs = get_task_trajectories(srvs["get_task_trajs"], "end_effector", trajectory_names)
    
    # Learn ProMPs from the data and get the configs
    promp_config = learn_promps(srvs["learn_promp"], ee_trajs=task_trajs)

    # # Get waypoints to add to ProMP config
    waypoints = []
    # if args.goal:
    #     g = session_config[args.goal]['goal']
    #     wpt = Waypoint()
    #     wpt.phase_val = 1.0
    #     wpt.condition_keys = ['x.0', 'x.1', 'x.2']
    #     wpt.values         = [g['x.0'], g['x.1'], g['x.2']]
    #     waypoints.append(wpt)
    
    # Send configs to action server for review
    success = set_promp(srvs["set_promp"], promp_config, waypoints, duration=10.0, dt=0.1)
