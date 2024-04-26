#!/usr/bin/env python
import os
import sys
import errno
import numpy as np
import rospy
import cPickle as pickle
from rospy_service_helper import SRV_NAMES
from rospy_service_helper import (get_task_trajectories,
                                  get_joint_trajectories, learn_weights)
from ll4ma_movement_primitives.promps import (ActiveLearner,
                                              ros_to_python_config)


class ActiveLearnerPostProcessor:
    def __init__(self):
        data_abs_path = rospy.get_param("~data_abs_path")
        data_rel_path = rospy.get_param("~data_rel_path")
        self.learner_idx = rospy.get_param("~learner_index")
        self.num_bfs = rospy.get_param("~num_bfs")
        self.regr_alpha = rospy.get_param("~regr_alpha")
        self.session_path = os.path.join(data_abs_path, data_rel_path)

        self.srvs = {
            "get_task_trajs": SRV_NAMES["get_task_trajs"],
            "get_joint_trajs": SRV_NAMES["get_joint_trajs"],
            "learn_weights": SRV_NAMES["learn_weights"]
        }

        # Wait for the services
        rospy.loginfo("Waiting for services I need...")
        for srv in self.srvs.keys():
            rospy.loginfo("    %s" % self.srvs[srv])
            rospy.wait_for_service(self.srvs[srv])
        rospy.loginfo("Services are up!")

    def relearn_active_learner(self):
        rospy.loginfo("Relearning learner {:03d}".format(self.learner_idx))
        original_learner = self.get_original_learner()
        new_learner = ActiveLearner()
        new_learner.ee_gmm = original_learner.ee_gmm
        new_learner.obj_gmm = original_learner.obj_gmm
        new_learner.selected_tasks = original_learner.selected_tasks
        new_learner.config = original_learner.config
        new_learner.table_position = original_learner.table_position
        new_learner.table_quaternion = original_learner.table_quaternion
        new_learner._metadata.experiment_type = original_learner._metadata.experiment_type
        new_learner._metadata.active_learning_type = original_learner._metadata.active_learning_type
        new_learner._metadata.object_type = original_learner._metadata.object_type
        new_learner._metadata.task = original_learner._metadata.task
        new_learner._metadata.robot = original_learner._metadata.robot
        new_learner._metadata.end_effector = original_learner._metadata.end_effector

        trajectory_names = [
            task.trajectory_name for task in new_learner.selected_tasks
        ]
        for trajectory_name in trajectory_names:
            if not trajectory_name:
                continue

            ee_trajs = get_task_trajectories(self.srvs["get_task_trajs"],
                                             "end_effector_pose_base_frame",
                                             [trajectory_name])
            j_traj = get_joint_trajectories(self.srvs["get_joint_trajs"],
                                            "lbr4", [trajectory_name])[0]

            # Learn ProMP weights
            w, ros_config = learn_weights(
                self.srvs["learn_weights"],
                ee_trajs=ee_trajs,
                num_bfs=self.num_bfs,
                regr_alpha=self.regr_alpha)
            config = ros_to_python_config(ros_config)
            config.init_joint_state = j_traj.points[0].positions
            config.num_bfs = self.num_bfs
            config.regr_alpha = self.regr_alpha
            new_learner.add_promp_demo(w, config, data_name=trajectory_name)

        new_learner.write_metadata()
        self.save_learner(new_learner)

    def get_original_learner(self):
        name = "active_learner_{:03d}.pkl".format(self.learner_idx)
        filename = os.path.join(self.session_path, "backup", name)
        rospy.loginfo("Loading learner from: {}".format(filename))
        with open(filename, 'r') as f:
            learner = pickle.load(f)
        return learner

    def save_learner(self, learner, postfix="__"):
        self._make_dir(os.path.join(self.session_path, "learner_copies"))
        name = "active_learner_{:03d}{}.pkl".format(self.learner_idx, postfix)
        filename = os.path.join(self.session_path, "learner_copies", name)
        rospy.loginfo("Saving learner to: {}".format(filename))
        with open(filename, 'w') as f:
            pickle.dump(learner, f)

    def _make_dir(self, new_dir):
        try:
            os.makedirs(new_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise


if __name__ == '__main__':
    rospy.init_node("active_learner_post_processor")
    post_processor = ActiveLearnerPostProcessor()
    post_processor.relearn_active_learner()