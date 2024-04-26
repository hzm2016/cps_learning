#!/usr/bin/env python
import os
import sys
import errno
import rospy
import argparse
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from tqdm import tqdm
from ll4ma_movement_primitives.promps import ActiveLearner, ros_to_python_config, TaskInstance
from rospy_service_helper import (
    get_task_trajectories, get_joint_trajectories, learn_weights, SRV_NAMES)
from ll4ma_movement_primitives.promps import Optimizer


class Experiment:
    """
    Coordinates an experiment session run on the robot.
    """
    def __init__(self, path, experiment, sampling_method, trial, num_instances, num_bfs=6):
        self.path = path
        self.experiment = experiment
        self.sampling_method = sampling_method
        self.trial = trial
        self.num_instances = num_instances
        self.num_bfs = num_bfs

        self.random_instances = None
        self.current_random_idx = 0

        self.selected_instances = []

        # Seeding with the trial number since it will be unique and easily identified
        np.random.seed(self.trial)

        # Setup file paths
        # data_path = os.path.join(path, "{}__grid_data".format(self.experiment))
        # learner_path = os.path.join(data_path, "learners")
        # learner_filename = max(
        #     [f for f in os.listdir(learner_path) if f.endswith(".pkl")])
        learner_filename = os.path.join(path, experiment, "final_active_learner.pkl")

        # This path will be for storing anything this class generates
        self.experiment_path = os.path.join(path, self.experiment, self.sampling_method,
                                            "trial_{}".format(self.trial))

        with open(learner_filename, 'r') as f:
            self.data_collection_learner = pickle.load(f)
        # This one will have instances removed based on selection
        # if self.sampling_method == "max_entropy":
        #     self.discrete_sample_set = self._get_neg_discrete_set(
        #         self.data_collection_learner)
        # else:
        self.discrete_sample_set = deepcopy(self.data_collection_learner.selected_tasks)

        self.optimizer = Optimizer(self.data_collection_learner.table_position,
                                   self.data_collection_learner.table_quaternion, None, None)

        # This one is left intact to compute metrics over
        self.original_sample_set = deepcopy(self.discrete_sample_set)
        if self.num_instances < 1:
            self.num_instances = len(self.discrete_sample_set)

        # Wait for necessary services
        srvs = [SRV_NAMES["get_task_trajs"], SRV_NAMES["get_joint_trajs"], SRV_NAMES["learn_weights"]]
        rospy.loginfo("Waiting for services:")
        for srv in srvs:
            rospy.loginfo("{:4}{s}".format("", s=srv))
            rospy.wait_for_service(srv)
        rospy.loginfo("Services are up!")

        rospy.loginfo(
            "\n\nPATH: {}\nEXPERIMENT: {}\nTRIAL: {}\nSAMPLING METHOD: {}"
            "\nNUM INSTANCES: {}\n".format(self.path, self.experiment,
                                           self.trial, self.sampling_method,
                                           self.num_instances))

    def create_learners(self):
        # Create the path for storing learners
        learner_path = os.path.join(self.experiment_path, "learners")
        self._make_dir(learner_path)

        # learner = ActiveLearner()
        # learner.table_position = self.data_collection_learner.table_position
        # learner.table_quaternion = self.data_collection_learner.table_quaternion
        # learner.config = self.data_collection_learner.config
        # learner._metadata.active_learning_type = self.sampling_method
        # learner._metadata.experiment_type = self.experiment
        # learner._metadata.object_type = self.data_collection_learner._metadata.object_type
        # # Fixing the learner's EE GMM to the one from ALL the data, because we
        # # don't want it shifting all the time for the discrete optimization and
        # # making well-estimated regions poorly estimated just because grasp
        # # points are shifting.
        # learner.ee_gmm = deepcopy(self.data_collection_learner.ee_gmm)

        start = 52
        # TODO hack
        with open(os.path.join(learner_path, "active_learner_%03d.pkl" % start), 'r') as f:
            learner = pickle.load(f)

        learner_idx = 1
        rospy.loginfo("Learning from demos...\n")
        for _ in tqdm(range(start, self.num_instances)):
            # Retrieve next instance based on sampling strategy
            instance = self._get_next_instance(learner)
            self.selected_instances.append(instance)

            if instance.label == 0:
                pp = instance.object_planar_pose
                learner.obj_gmm.add_instance([pp.x, pp.y, pp.theta], False)
                learner.obj_gmm.learn(False)
            else:
                # Get the trajectory data of the associated trajectory
                traj_name = instance.trajectory_name
                task_traj = get_task_trajectories(SRV_NAMES["get_task_trajs"],
                                                  "end_effector_pose_base_frame", [traj_name])[0]
                joint_traj = get_joint_trajectories(SRV_NAMES["get_joint_trajs"], "lbr4",
                                                    [traj_name])[0]

                # Learn ProMP weights and add demo to learner
                w, ros_config = learn_weights(SRV_NAMES["learn_weights"],
                                              ee_trajs=[task_traj],
                                              num_bfs=self.num_bfs)
                config = ros_to_python_config(ros_config)
                config.init_joint_state = joint_traj.points[0].positions
                learner.add_promp_demo(w, config, traj_name, display_msg=False)

            # Save the learner to disk
            learner.write_metadata()
            learner_filename = os.path.join(learner_path, "active_learner_%03d.pkl" % learner_idx)
            with open(learner_filename, 'w') as f:
                pickle.dump(learner, f)

            learner_idx += 1

    def _get_next_instance(self, learner=None):
        if self.sampling_method == "random":
            next_instance = self._select_random_instance()
        elif self.sampling_method == "mahal_cond":
            next_instance = self._select_mahalanobis_instance(learner, True)
        elif self.sampling_method == "mahal_no":
            next_instance = self._select_mahalanobis_instance(learner, False)
        elif self.sampling_method == "max_entropy":
            next_instance = self._select_entropy_instance(learner)
        elif self.sampling_method == "least_confident":
            next_instance = self._select_least_confident_instance(learner)
        elif self.sampling_method == "min_margin":
            next_instance = self._select_min_margin_instance(learner)
        elif self.sampling_method == "kl_cond_ee":
            next_instance = self._select_kl_instance(learner, "kl_cond_ee")
        elif self.sampling_method == "kl_ee_cond":
            next_instance = self._select_kl_instance(learner, "kl_ee_cond")
        elif self.sampling_method == "min_prob_cond":
            next_instance = self._select_min_prob_instance(learner, True)
        elif self.sampling_method == "min_prob_no":
            next_instance = self._select_min_prob_instance(learner, False)
        else:
            rospy.logerr("Unknown sampling method: {}".format(self.sampling_method))
        return next_instance

    def _select_random_instance(self):
        if self.random_instances is None:
            self.random_instances = np.random.choice(
                self.discrete_sample_set, self.num_instances, replace=False)
        next_instance = self.random_instances[self.current_random_idx]
        self.current_random_idx += 1
        return next_instance

    def _select_mahalanobis_instance(self, learner, condition):
        # Have to get a couple random instances first since the EE GMM needs to be fit
        if len(self.selected_instances) < 2:
            next_instance = self._select_random_instance()
        else:
            # Retrieve a Casadi symbolic function for faster computation
            promps = learner.promp_library.get_params()
            ee_gmm = learner.ee_gmm.get_params()
            for i in range(len(ee_gmm['means'])):
                ee_gmm['means'][i][3:] /= np.linalg.norm(ee_gmm['means'][i][3:])
            phi = learner.promp_library.get_phi().T
            mahal_func = self.optimizer.get_mahalanobis_function(phi, promps, ee_gmm)

            next_instance = None
            worst_mahal = 0.0
            for instance in self.discrete_sample_set:
                # mahal = learner.evaluate_mahalanobis(
                #     instance.object_planar_pose, condition)
                pp = instance.object_planar_pose
                mahal = mahal_func([pp.x, pp.y, pp.z, pp.theta])
                if mahal > worst_mahal:
                    worst_mahal = mahal
                    next_instance = instance

        # We don't want to ever resample the same point, since there is only
        # one demo per point and seeing it again won't affect the learning. We
        # can actually get stuck always trying to retrieve the same sample. So,
        # delete it from the candidates.
        self.discrete_sample_set.remove(next_instance)

        return next_instance

    def _select_entropy_instance(self, learner):
        if len(self.selected_instances) < 2:
            next_instance = self._select_random_instance()
        else:
            # Retrieve a Casadi symbolic function for faster computation
            promps = learner.promp_library.get_params()
            ee_gmm = learner.ee_gmm.get_params()
            for i in range(len(ee_gmm['means'])):
                ee_gmm['means'][i][3:] /= np.linalg.norm(ee_gmm['means'][i][3:])
            phi = learner.promp_library.get_phi().T
            entropy_func = self.optimizer.get_entropy_function(phi, promps, ee_gmm)

            next_instance = None
            high_entropy = 0.0
            for instance in self.discrete_sample_set:
                pp = instance.object_planar_pose
                # entropy = learner._evaluate_entropy(planar_pose)
                entropy = entropy_func([pp.x, pp.y, pp.z, pp.theta])
                if entropy >= high_entropy:
                    high_entropy = entropy
                    next_instance = instance

        self.discrete_sample_set.remove(next_instance)
        return next_instance

    def _select_least_confident_instance(self, learner):
        if len(self.selected_instances) < 2:
            next_instance = self._select_random_instance()
        else:
            # Retrieve a Casadi symbolic function for faster computation
            promps = learner.promp_library.get_params()
            ee_gmm = learner.ee_gmm.get_params()
            for i in range(len(ee_gmm['means'])):
                ee_gmm['means'][i][3:] /= np.linalg.norm(ee_gmm['means'][i][3:])
            phi = learner.promp_library.get_phi().T
            lc_func = self.optimizer.get_least_confident_function(phi, promps, ee_gmm)

            next_instance = None
            worst_lc = sys.maxint
            for instance in self.discrete_sample_set:
                pp = instance.object_planar_pose
                # entropy = learner._evaluate_entropy(planar_pose)
                lc = lc_func([pp.x, pp.y, pp.z, pp.theta])

                if lc <= worst_lc:
                    worst_lc = lc
                    next_instance = instance

        self.discrete_sample_set.remove(next_instance)
        return next_instance

    def _select_min_margin_instance(self, learner):
        if len(self.selected_instances) < 2:
            next_instance = self._select_random_instance()
        else:
            next_instance = None
            min_margin = sys.maxint
            for instance in self.discrete_sample_set:
                margin = learner.evaluate_min_margin(instance.object_planar_pose)
                if margin <= min_margin:
                    min_margin = margin
                    next_instance = instance

        self.discrete_sample_set.remove(next_instance)
        return next_instance

    def _select_kl_instance(self, learner, div_type):
        if len(self.selected_instances) < 2:
            next_instance = self._select_random_instance()
        else:
            next_instance = None
            high_value = 0.0
            for instance in self.discrete_sample_set:
                value = learner.evaluate_kl_divergence(instance.object_planar_pose, div_type)
                if value > high_value:
                    high_value = value
                    next_instance = instance
        self.discrete_sample_set.remove(next_instance)
        return next_instance

    def _get_neg_discrete_set(self, learner, width=5):
        discrete_sample_set = deepcopy(self.data_collection_learner.selected_tasks)
        first = deepcopy(discrete_sample_set[0])
        last = deepcopy(discrete_sample_set[-1])
        low_x = first.grid_coords[0]
        high_x = last.grid_coords[0]
        low_y = first.grid_coords[1]
        high_y = last.grid_coords[1]
        x_diff = 0.05714285714300002
        y_diff = 0.055
        thetas = np.linspace(-3.14, 3.14, 8, endpoint=False)

        # Add left infeasible region
        for i in range(low_x - width, high_x + width + 1):
            for j in range(low_y - width, low_y):
                for k, theta in enumerate(thetas):
                    instance = TaskInstance()
                    instance.table_pose = first.table_pose
                    instance.object_planar_pose.x = i * x_diff + first.object_planar_pose.x
                    instance.object_planar_pose.y = j * y_diff + first.object_planar_pose.y
                    instance.object_planar_pose.z = first.object_planar_pose.z
                    instance.object_planar_pose.theta = theta
                    instance.grid_coords = (i, j, k)
                    instance.label = 0
                    discrete_sample_set.append(instance)
        # Add right infeasible region
        for i in range(low_x - width, high_x + width + 1):
            for j in range(high_y + 1, high_y + 1 + width):
                for k, theta in enumerate(thetas):
                    instance = TaskInstance()
                    instance.table_pose = first.table_pose
                    instance.object_planar_pose.x = i * x_diff + first.object_planar_pose.x
                    instance.object_planar_pose.y = j * y_diff + first.object_planar_pose.y
                    instance.object_planar_pose.z = first.object_planar_pose.z
                    instance.object_planar_pose.theta = theta
                    instance.grid_coords = (i, j, k)
                    instance.label = 0
                    discrete_sample_set.append(instance)
        # Add top infeasible region
        for i in range(low_x - width, low_x):
            for j in range(low_y, high_y + 1):
                for k, theta in enumerate(thetas):
                    instance = TaskInstance()
                    instance.table_pose = first.table_pose
                    instance.object_planar_pose.x = i * x_diff + first.object_planar_pose.x
                    instance.object_planar_pose.y = j * y_diff + first.object_planar_pose.y
                    instance.object_planar_pose.z = first.object_planar_pose.z
                    instance.object_planar_pose.theta = theta
                    instance.grid_coords = (i, j, k)
                    instance.label = 0
                    discrete_sample_set.append(instance)
        # Add bottom infeasible region
        for i in range(high_x + 1, high_x + 1 + width):
            for j in range(low_y, high_y + 1):
                for k, theta in enumerate(thetas):
                    instance = TaskInstance()
                    instance.table_pose = first.table_pose
                    instance.object_planar_pose.x = i * x_diff + first.object_planar_pose.x
                    instance.object_planar_pose.y = j * y_diff + first.object_planar_pose.y
                    instance.object_planar_pose.z = first.object_planar_pose.z
                    instance.object_planar_pose.theta = first.object_planar_pose.theta
                    instance.grid_coords = (i, j, k)
                    instance.label = 0
                    discrete_sample_set.append(instance)

        # Update x-y grid indices
        for instance in discrete_sample_set:
            instance.grid_coords = (instance.grid_coords[0] + width,
                                    instance.grid_coords[1] + width,
                                    instance.grid_coords[2])

        # DEBUG by uncommenting this code to get a plot and change the value
        # num_xs = learner.config["num_xs"] + (width * 2)
        # num_ys = learner.config["num_ys"] + (width * 2)
        # num_thetas = learner.config["num_thetas"]
        # grids = [np.ones((num_xs, num_ys)) for _ in range(num_thetas)]
        # for instance in discrete_sample_set:
        #     value = instance.object_planar_pose.y
        #     x, y, theta = instance.grid_coords
        #     grids[theta][x, y] = value

        # fig, axes = plt.subplots(2, 4)
        # fig.set_size_inches(14, 6)
        # for k in range(len(grids)):
        #     idx = int(k > 3)
        #     ax = axes[idx, k % 4]
        #     ax.imshow(grids[k], vmin=-1.0, vmax=1.0, cmap="hot")
        # plt.tight_layout()
        # plt.show()
        # plt.close(fig)
        return discrete_sample_set

    def _make_dir(self, path):
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise


if __name__ == '__main__':
    rospy.init_node("experiment_session", anonymous=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sampling_method', dest='sampling_method', type=str, required=True,
                        choices=["random", "mahal_cond", "max_entropy", "least_confident",
                                 "min_margin"])
    parser.add_argument('-t', '--trial', dest='trial', type=int, required=True)
    parser.add_argument('-n', '--num_instances', dest='num_instances', type=int, default=100)
    parser.add_argument('-p', '--path', dest='path', type=str,
                        default="/media/adam/data_haro/rss_2019")
    parser.add_argument('-e', '--experiment', dest='experiment', type=str, default="experiment_1")

    args = parser.parse_args(sys.argv[1:])
    experiment = Experiment(args.path, args.experiment, args.sampling_method,
                            args.trial, args.num_instances)
    experiment.create_learners()
