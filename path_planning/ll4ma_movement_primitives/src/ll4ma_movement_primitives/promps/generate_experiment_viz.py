#!/usr/bin/env python
import os
import sys
import errno
import argparse
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from tqdm import tqdm
from ll4ma_movement_primitives.promps import (
    ActiveLearner, ros_to_python_config, TaskInstance)
from ll4ma_movement_primitives.promps import Optimizer


class ExperimentVisualization:
    def __init__(self, path, experiment, sampling_method, trial,
                 visualization_method):
        self.path = path
        self.experiment = experiment
        self.sampling_method = sampling_method
        self.trial = trial
        self.visualization_method = visualization_method

        learner_filename = os.path.join(path, experiment,
                                        "final_active_learner.pkl")

        # This path will be for storing anything this class generates
        self.experiment_path = os.path.join(path, self.experiment,
                                            self.sampling_method,
                                            "trial_{}".format(self.trial))

        with open(learner_filename, 'r') as f:
            self.data_collection_learner = pickle.load(f)

        self.optimizer = Optimizer(
            self.data_collection_learner.table_position,
            self.data_collection_learner.table_quaternion, None, None)

        if self.sampling_method == "test":
            self.discrete_sample_set = self._get_neg_discrete_set(
                self.data_collection_learner)
            sys.exit(0)
        else:
            self.discrete_sample_set = deepcopy(
                self.data_collection_learner.selected_tasks)

        # This one is left intact to compute metrics over
        self.original_sample_set = deepcopy(self.discrete_sample_set)

        print "\n*** VISUALIZATION ***"
        print("EXPERIMENT: {}\nSAMPLING METHOD: {}\nTRIAL: {}\n"
              "VISUALIZATION METHOD: {}\n".format(
                  self.experiment, self.sampling_method, self.trial,
                  self.visualization_method))

    def create_metric_grid_viz(self):
        learner_path = os.path.join(self.experiment_path, "learners")
        learner_names = os.listdir(learner_path)
        data_path = os.path.join(self.experiment_path, "viz",
                                 self.visualization_method, "metric_grids",
                                 "data")
        self._make_dir(data_path)
        for i in tqdm(range(len(learner_names))):
            learner_name = learner_names[i]
            filename = os.path.join(learner_path, learner_name)
            with open(filename, 'r') as f:
                learner = pickle.load(f)

            learner_idx = int(learner_name.split('.')[0].split('_')[-1])

            grids = self._compute_metric_grids(
                self.original_sample_set, learner, self.visualization_method)
            if self.visualization_method == "mahal_cond":
                viz_max_val = 30.0
            elif self.visualization_method == "least_confident":
                viz_max_val = 1.0
            else:
                # TODO not sure what to do yet for others
                viz_max_val = 0
            self._plot_metric_grids(grids, self.visualization_method,
                                    learner_idx, viz_max_val)

            grid_data_filename = "{}__{}__{}__{:03d}.pkl".format(
                self.sampling_method, self.trial, self.visualization_method,
                learner_idx)
            with open(os.path.join(data_path, grid_data_filename), 'w') as f:
                pickle.dump(grids, f)

    def create_metric_trend_viz(self, num_trials=10):
        # values_dict = {}
        # for trial_num in range(num_trials):
        #     data_path = os.path.join(
        #         self.path, self.experiment, self.sampling_method,
        #         "trial_{}".format(trial_num), "viz", self.visualization_method,
        #         "metric_grids", "data")
        #     values = []
        #     for data_name in sorted(os.listdir(data_path)):
        #         data_filename = os.path.join(data_path, data_name)
        #         with open(data_filename, 'r') as f:
        #             grids = pickle.load(f)
        #         values.append(max([np.max(g) for g in grids]))
        #     values_dict[trial_num] = values
        # with open(os.path.join(self.path, self.experiment, self.sampling_method, "{}_values.pkl".format(self.sampling_method)), 'w') as f:
        #     pickle.dump(values_dict, f)

        with open(
                os.path.join(self.path, self.experiment, self.sampling_method,
                             "{}_values.pkl".format(self.sampling_method)),
                'r') as f:
            values_dict = pickle.load(f)
        for key in values_dict.keys():
            values = values_dict[key]
            start = 10
            end = 100
            plt.plot(range(start, end), values[start:end], label=key)
            plt.xlim(start, end)
            plt.ylim(0, 19)
        # plt.title(self.sampling_method)
        plt.legend()
        # plt.show()
        save_filename = os.path.join(
            "/home/adam/paper_submissions/promp_active_learning_paper/imgs",
            "{}_trend.pdf".format(self.sampling_method))
        plt.savefig(save_filename)

    def create_metric_grid_image(self, learner_idx):
        learner_name = "active_learner_{:03d}.pkl".format(learner_idx)
        learner_path = os.path.join(self.experiment_path, "learners")
        data_path = os.path.join(self.experiment_path, "viz",
                                 self.visualization_method, "metric_grids",
                                 "data")
        self._make_dir(data_path)
        filename = os.path.join(learner_path, learner_name)
        with open(filename, 'r') as f:
            learner = pickle.load(f)
        grids = self._compute_metric_grids(self.original_sample_set, learner,
                                           self.visualization_method)
        if self.visualization_method == "mahal_cond":
            viz_max_val = 30.0
        elif self.visualization_method == "least_confident":
            viz_max_val = 1.0
        else:
            viz_max_val = None
        grid_data_filename = "{}__{}__{}__{:03d}.pkl".format(
            self.sampling_method, self.trial, self.visualization_method,
            learner_idx)
        with open(os.path.join(data_path, grid_data_filename), 'w') as f:
            pickle.dump(grids, f)

        self._plot_metric_grids(grids, self.visualization_method, learner_idx,
                                viz_max_val)

    def create_single_orient(self, learner_idx, orient, viz_max_val=None):
        grid_path = os.path.join(self.experiment_path, "viz",
                                 self.visualization_method, "metric_grids")
        data_path = os.path.join(grid_path, "data")
        viz_path = os.path.join(grid_path, "imgs")
        grid_data_filename = "{}__{}__{}__{:03d}.pkl".format(
            self.sampling_method, self.trial, self.visualization_method,
            learner_idx)

        with open(os.path.join(data_path, grid_data_filename), 'r') as f:
            grids = pickle.load(f)

        grid = grids[orient]

        max_value = np.max(grid)
        # if viz_max_val is None:
        # viz_max_val = max_value

        fig, ax = plt.subplots()
        ax.imshow(1.0 - grid, vmin=0.0, vmax=max_value, cmap="hot")
        save_filename = "%s_%03d_%d.pdf" % (self.sampling_method, learner_idx,
                                            orient)
        save_path = os.path.join(viz_path, save_filename)

        # plt.suptitle("Max value: {}".format(max_value))
        frame1 = plt.gca()
        frame1.axes.xaxis.set_ticklabels([])
        frame1.axes.yaxis.set_ticklabels([])
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)
        plt.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

    def _compute_metric_grids(self, instances, learner, viz_type):
        num_xs = learner.config["num_xs"]
        num_ys = learner.config["num_ys"]
        num_thetas = learner.config["num_thetas"]
        grids = [np.ones((num_xs, num_ys)) for _ in range(num_thetas)]

        if viz_type == "mahal_cond":
            # print "Building mahalanobis function..."
            # Retrieve a Casadi symbolic function for faster computation
            promps = learner.promp_library.get_params()
            ee_gmm = learner.ee_gmm.get_params()
            for i in range(len(ee_gmm['means'])):
                ee_gmm['means'][i][3:] /= np.linalg.norm(
                    ee_gmm['means'][i][3:])
            phi = learner.promp_library.get_phi().T
            mahal_func = self.optimizer.get_mahalanobis_function(
                phi, promps, ee_gmm)
            # print "Function built."

        for i in tqdm(range(len(instances))):
            instance = instances[i]
            planar_pose = instance.object_planar_pose
            if viz_type == "mahal_cond":
                value = mahal_func([
                    planar_pose.x, planar_pose.y, planar_pose.z,
                    planar_pose.theta
                ])
                # value = learner.evaluate_mahalanobis(
                #     instance.object_planar_pose, condition=True)
            elif viz_type == "mahal_no":
                value = learner.evaluate_mahalanobis(
                    planar_pose, condition=False)
            elif viz_type == "max_entropy":
                # Retrieve a Casadi symbolic function for faster computation
                promps = learner.promp_library.get_params()
                ee_gmm = learner.ee_gmm.get_params()
                for i in range(len(ee_gmm['means'])):
                    ee_gmm['means'][i][3:] /= np.linalg.norm(
                        ee_gmm['means'][i][3:])
                phi = learner.promp_library.get_phi().T
                entropy_func = self.optimizer.get_entropy_function(
                    phi, promps, ee_gmm)
                # value = learner._evaluate_entropy(instance.object_planar_pose)
                pp = instance.object_planar_pose
                value = entropy_func([pp.x, pp.y, pp.z, pp.theta])
            elif viz_type == "least_confident":
                # Retrieve a Casadi symbolic function for faster computation
                promps = learner.promp_library.get_params()
                ee_gmm = learner.ee_gmm.get_params()
                for i in range(len(ee_gmm['means'])):
                    ee_gmm['means'][i][3:] /= np.linalg.norm(
                        ee_gmm['means'][i][3:])
                phi = learner.promp_library.get_phi().T
                lc_func = self.optimizer.get_least_confident_function(
                    phi, promps, ee_gmm)
                # value = learner._evaluate_entropy(instance.object_planar_pose)
                pp = instance.object_planar_pose
                value = lc_func([pp.x, pp.y, pp.z, pp.theta])
            elif viz_type == "min_margin":
                value = learner.evaluate_min_margin(planar_pose)
            elif viz_type == "kl_cond_ee":
                value = learner.evaluate_kl_divergence(planar_pose,
                                                       "kl_cond_ee")
            elif viz_type == "kl_ee_cond":
                value = learner.evaluate_kl_divergence(planar_pose,
                                                       "kl_ee_cond")
            elif viz_type == "min_prob_cond":
                value = learner.evaluate_promp_prob(planar_pose, True)
            elif viz_type == "min_prob_no":
                value = learner.evaluate_promp_prob(planar_pose, False)
            x, y, theta = instance.grid_coords
            grids[theta][x, y] = value
        return grids

    def _plot_metric_grids(self,
                           grids,
                           viz_type,
                           learner_idx,
                           viz_max_val=None):
        viz_path = os.path.join(self.experiment_path, "viz", viz_type,
                                "metric_grids", "imgs")
        self._make_dir(viz_path)
        max_value = max([np.max(grid) for grid in grids])
        if viz_max_val is None:
            viz_max_val = max_value
        fig, axes = plt.subplots(2, 4)
        fig.set_size_inches(16, 8)
        for k in range(len(grids)):
            idx = int(k > 3)
            ax = axes[idx, k % 4]
            ax.imshow(grids[k], vmin=0.0, vmax=viz_max_val, cmap="hot")
        save_filename = "%s_%03d.png" % (self.sampling_method, learner_idx)
        save_path = os.path.join(viz_path, save_filename)
        plt.suptitle("Max value: {}".format(max_value))
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

    def plot_all_grids(self):
        data_path = os.path.join(self.experiment_path, "viz",
                                 self.visualization_method, "metric_grids",
                                 "data")
        viz_path = os.path.join(self.experiment_path, "viz",
                                self.visualization_method, "metric_grids",
                                "imgs")
        self._make_dir(viz_path)
        for learner_idx, filename in enumerate(os.listdir(data_path)):
            with open(os.path.join(data_path, filename), 'r') as f:
                grids = pickle.load(f)
            max_value = max([np.max(grid) for grid in grids])
            # if viz_max_val is None:
            #     viz_max_val = max_value
            viz_max_val = 30
            fig, axes = plt.subplots(2, 4)
            fig.set_size_inches(16, 8)
            for k in range(len(grids)):
                idx = int(k > 3)
                ax = axes[idx, k % 4]
                ax.imshow(grids[k], vmin=0.0, vmax=viz_max_val, cmap="hot")
            save_filename = "%s_%03d.pdf" % (self.sampling_method, learner_idx)
            save_path = os.path.join(viz_path, save_filename)
            plt.suptitle("Max value: {}".format(max_value))
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close(fig)

    def _get_neg_discrete_set(self, learner, width=5):
        discrete_sample_set = deepcopy(
            self.data_collection_learner.selected_tasks)
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
        num_xs = learner.config["num_xs"] + (width * 2)
        num_ys = learner.config["num_ys"] + (width * 2)
        num_thetas = learner.config["num_thetas"]
        grids = [np.ones((num_xs, num_ys)) for _ in range(num_thetas)]
        for instance in discrete_sample_set:
            value = instance.label  #.object_planar_pose.y
            x, y, theta = instance.grid_coords
            grids[theta][x, y] = value

        fig, axes = plt.subplots(2, 4)
        fig.set_size_inches(14, 6)
        for k in range(len(grids)):
            idx = int(k > 3)
            ax = axes[idx, k % 4]
            ax.imshow(grids[k], vmin=0, vmax=1.0, cmap="hot")
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        return discrete_sample_set

    def _make_dir(self, path):
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--sampling_method',
        dest='sampling_method',
        type=str,
        required=True,
        choices=[
            "random", "mahal_cond", "max_entropy", "least_confident",
            "min_margin", "test"
        ])
    parser.add_argument(
        '-v',
        '--viz_method',
        dest='viz_method',
        type=str,
        choices=["mahal_cond", "max_entropy", "least_confident", "min_margin"])
    parser.add_argument('-t', '--trial', dest='trial', type=int)
    parser.add_argument(
        '-p',
        '--path',
        dest='path',
        type=str,
        default="/media/adam/data_haro/rss_2019")
    parser.add_argument(
        '-e',
        '--experiment',
        dest='experiment',
        type=str,
        default="experiment_1")
    parser.add_argument(
        '--visualization_type',
        dest='visualization_type',
        type=str,
        required=True,
        choices=[
            "metric_grid", "metric_trend", "grid_image", "single_orient",
            "test"
        ])
    parser.add_argument('-l', '--learner', dest='learner', type=int)
    parser.add_argument('-k', dest='orient', type=int)

    args = parser.parse_args(sys.argv[1:])

    viz = ExperimentVisualization(args.path, args.experiment,
                                  args.sampling_method, args.trial,
                                  args.viz_method)
    if args.visualization_type == "metric_grid":
        viz.create_metric_grid_viz()
    elif args.visualization_type == "metric_trend":
        viz.create_metric_trend_viz()
    elif args.visualization_type == "grid_image":
        # viz.create_metric_grid_image(args.learner)
        viz.plot_all_grids()
    elif args.visualization_type == "single_orient":
        viz.create_single_orient(args.learner, args.orient)
    else:
        print "Unknown visualization type: {}".format(args.visualization_type)