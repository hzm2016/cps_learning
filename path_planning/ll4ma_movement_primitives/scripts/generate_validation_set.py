#!/usr/bin/env python
import os
import sys
import yaml
import errno
import rospy
import rospkg
import argparse
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from tqdm import tqdm
from ll4ma_movement_primitives.promps import (
    ActiveLearner, ros_to_python_config, TaskInstance)
from rospy_service_helper import (
    get_task_trajectories, get_joint_trajectories, learn_weights, SRV_NAMES)
from ll4ma_movement_primitives.promps import Optimizer
from ll4ma_movement_primitives.promps import active_learner_util as al_util


def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


if __name__ == '__main__':
    rospy.init_node("experiment_session", anonymous=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trial', dest='trial', type=int, required=True)
    parser.add_argument(
        '-n', '--num_instances', dest='num_instances', type=int, default=10)
    parser.add_argument(
        '-p',
        '--path',
        dest='path',
        type=str,
        default="/media/adam/URLG_HDD/adam")
    parser.add_argument(
        '-e',
        '--experiment',
        dest='experiment',
        type=str,
        default="experiment_1")

    args = parser.parse_args(sys.argv[1:])

    path = os.path.join(args.path, args.experiment, "validation")
    make_dir(path)
    data_filename = os.path.join(
        path, "{}_validation_trial_{}.pkl".format(args.experiment, args.trial))

    # Adding 100 since trials themselves were seeded on the trial nums, so this
    # ensures we're not just getting same set of instances that data was
    # collected on
    np.random.seed(args.trial + 100)

    rospack = rospkg.RosPack()
    config_path = rospack.get_path("ll4ma_policy_learning")
    config_path = os.path.join(config_path, "config")

    config = al_util.load_config(config_path,
                                 "{}.yaml".format(args.experiment))

    data = {}
    for i in range(args.num_instances):
        x = np.random.uniform(config["x_min"], config["x_max"])
        y = np.random.uniform(config["y_min"], config["y_max"])
        z = config["z"]
        theta = np.random.uniform(config["theta_min"], config["theta_max"])
        data["pose_{}".format(i)] = {}
        data["pose_{}".format(i)]["x"] = x
        data["pose_{}".format(i)]["y"] = y
        data["pose_{}".format(i)]["z"] = z
        data["pose_{}".format(i)]["theta"] = theta

    with open(data_filename, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
