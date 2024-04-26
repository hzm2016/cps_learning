import os
import sys
import yaml
import rospy
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from ll4ma_movement_primitives.promps import ProMP, ProMPConfig
from rospy_service_helper import (
    get_task_trajectories,
    learn_promps
)


srvs = {
    "get_task_trajectories" : "/pandas/get_task_trajectories",
    "learn_promps"          : "/policy_learner/learn_promps"
}




def parse_args(args):
    parser = argparse.ArgumentParser()
    def tup(a):
        return map(int, a.split(','))
    parser.add_argument('filename', type=str, help="Filename of YAML configuration file.")
    parser.add_argument('-p', dest='path', type=str, default=os.environ.get("DATA"),
                        help="Absolute path to data directory (default tries $DATA)")
    parser.add_argument('-d', nargs='+', dest='data_specs', type=tup,
                        help="Tuples specifying GROUP,SUBGROUP,NUM_DEMOS as determined "
                        "by the YAML configuration file")
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help="Display logging messages if enabled")
    parser.add_argument('-w', dest='wishart_factor', type=float, help="Influence of IW prior",
                        default="0.0")
    parsed_args = parser.parse_args(sys.argv[1:])
    return parsed_args

def get_session_config(path, config_filename):
    config_file = os.path.join(path, config_filename)
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    return config

def get_lists(d):
    """From https://stackoverflow.com/questions/51426716"""
    r = [b if isinstance(b, list) else get_lists(b) for b in d.values()]
    return [i for b in r for i in b]

def ros_to_python_config(ros_config):
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




if __name__ == '__main__':
    rospy.init_node("test_wishart_prior")

    args = parse_args(sys.argv)

    def log(msg):
        if args.debug:
            rospy.loginfo(msg)

    log("Waiting for services...")
    for srv in srvs.keys():
        log("    %s" % srvs[srv])
        try:
            rospy.wait_for_service(srvs[srv])
        except rospy.exceptions.ROSInterruptException:
            sys.exit()
    log("Services are up!")

    # Read in the session configuration file
    data_path = os.path.join(os.environ.get("DATA"), "reach_cup")
    session_config = get_session_config(data_path, "reach_cup.yaml")

    # Read in trajectory names from data specifications
    t_names_dict = {}
    if args.data_specs:
        for group, subgroup, num_demos in args.data_specs:
            if group not in t_names_dict.keys():
                t_names_dict[group] = {}
            if subgroup not in t_names_dict[group].keys():
                t_names_dict[group][subgroup] = []
            t_names_dict[group][subgroup] = session_config[group][subgroup]['trajectories'][:num_demos]
    else:
        for group in session_config.keys():
            for subgroup in session_config[group].keys():
                if group not in t_names_dict.keys():
                    t_names_dict[group] = {}
                if subgroup not in t_names_dict[group].keys():
                    t_names_dict[group][subgroup] = []
                t_names_dict[group][subgroup] = session_config[group][subgroup]['trajectories']
    t_names = get_lists(t_names_dict)

    # Retrieve the data for the task trajectories
    t_trajs = get_task_trajectories(srvs["get_task_trajectories"], "end_effector", t_names)

    # Learn ProMPs
    promp_config = ros_to_python_config(learn_promps(srvs["learn_promps"], ee_trajs=t_trajs))
    promp_config.inv_wish_factor = args.wishart_factor
    promp = ProMP(config=promp_config)
    p_trajs = []
    for i in range(10):
        p_traj, dist = promp.generate_trajectory(duration=10.0, dt=0.1)
        p_trajs.append(p_traj)

    # Compute metrics
    sigma_w = promp.get_sigma_w()
    dist_to_sing = np.linalg.norm(sigma_w, ord='fro') / np.linalg.cond(sigma_w, p='fro')
    U, S, V = np.linalg.svd(sigma_w)
    min_sv = np.min(S)
    pos_def = np.all(np.linalg.eigvals(sigma_w) > 0)


    # Create plots
    fig = plt.figure(figsize=(14,9))
    gs = plt.GridSpec(2,2, height_ratios=[4,1])
    gs.update(left=0, right=0.98, top=0.9, bottom=0)
    ax = fig.add_subplot(gs[0,0], projection='3d')

    # Plot the pose trajectories
    for t_traj in t_trajs:
        xs = []; ys = []; zs = []
        for t_point in t_traj.points:
            xs.append(t_point.pose.position.x)
            ys.append(t_point.pose.position.y)
            zs.append(t_point.pose.position.z)
        ax.plot(xs, ys, zs, color='b')

    # Plot the ProMP trajectories
    for p_traj in p_trajs:
        xs = []; ys = []; zs = []
        for j in range(len(p_traj['x'][0])):
            xs.append(p_traj['x'][0][j])
            ys.append(p_traj['x'][1][j])
            zs.append(p_traj['x'][2][j])
        ax.plot(xs, ys, zs, color='r')

    # Visualize covariance
    ax = fig.add_subplot(gs[0,1])
    sns.heatmap(sigma_w)
    plt.axis('equal')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False) 

    ax = fig.add_subplot(gs[1,:])
    plt.axis('off')
    axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
    txt = ("Distance To Singular: %f\n"
           "Smallest Singular Value: %f\n"
           "Positive Definite? %s"% (dist_to_sing, min_sv, pos_def))
    text_box = TextBox(axbox, 'Analysis', initial=txt)

    plt.show()

    
    
    

