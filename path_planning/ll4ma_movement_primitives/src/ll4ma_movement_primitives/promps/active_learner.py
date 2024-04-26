import sys
import time
import random
import rospy
import numpy as np
import torch
from copy import deepcopy
from scipy.stats import multivariate_normal
from torch.distributions.kl import kl_divergence
from geometry_msgs.msg import Pose, PoseArray
from ll4ma_movement_primitives.msg import PlanarPose
from ll4ma_movement_primitives.promps import (ProMPLibrary, GMM, TaskInstance,
                                              ActiveLearnerMetadata, Waypoint,
                                              Optimizer, mahalanobis_distance)
from ll4ma_movement_primitives.srv import GetOptimizationInstance, GetOptimizationInstanceRequest
import active_learner_util as util
from tf import transformations as tf
from rospy_service_helper import visualize_poses

_MAX_ENTROPY = "max_entropy"
_LEAST_CONFIDENT = "least_confident"
_MIN_MARGIN = "min_margin"
_RANDOM = "random"
_MAHALANOBIS = "mahalanobis"
_GRID = "grid"
_VALIDATION = "validation"


class ActiveLearner:
    """
    Main class governing coordination of active learning activities.
    """
    def __init__(self,
                 config_path="",
                 config_filename="",
                 session_path="",
                 active_learning_type=None,
                 experiment_type=None,
                 object_type=None,
                 random_seed=None):
        self.promp_library = ProMPLibrary()
        self.obj_gmm = GMM()
        self.ee_gmm = GMM()
        self.task_candidates = []
        self.selected_tasks = []
        self.zero_lim = 1e-15
        self.num_solve_attempts = 3
        self._metadata = ActiveLearnerMetadata()
        self._metadata.active_learning_type = active_learning_type
        self._metadata.experiment_type = experiment_type
        self._metadata.object_type = object_type
        if random_seed is not None:
            np.random.seed(random_seed)
        if config_path and config_filename and session_path:
            self.config = util.load_config(config_path, config_filename)
            self.img_path = util.setup_img_path(session_path)
            self._set_table_pose()
            self._generate_finite_goal_set()

        self.grid_idx = 0
        self.validation_idx = 0
        self.validation_set = []

        self.visualize_poses()

    def add_instance_to_obj_gmm(self):
        if self.get_latest_instance().label != 0:
            rospy.logwarn("Recent instance was not a failure, doing nothing.")
            return False
        else:
            pp = self.get_latest_instance().object_planar_pose
            # Igoring z because the variance will be super small, we really
            # don't care about the height of the plane, only the pose once the
            # plane is given.
            self.obj_gmm.add_instance([pp.x, pp.y, pp.theta])
            self.obj_gmm.learn()
            return True

    def add_instance_to_ee_gmm(self, pose, display_msg=True):
        self.ee_gmm.add_instance(pose, display_msg)
        self.ee_gmm.learn(display_msg)

    def add_promp_demo(self, w, config, data_name="", display_msg=True):
        """
        Add a new weight space demo to the ProMP library.
        
        Args:
            w (ndarray): Weight vector learned from trajectory data
            config (ProMPConfig): ProMP configuration
            data_name (str): Filename associated with data the demo was
            learned from.
        """
        promp_names, merged = self.promp_library.add_demo(w, config, display_msg)
        # Update the learner metadata
        for promp_name in promp_names:
            if promp_name not in self._metadata.promp_names:
                self._metadata.promp_names.append(promp_name)
                self._metadata.promp_data_names[promp_name] = []
            num_demos = self.promp_library.get_promp(promp_name).num_demos
            self._metadata.promp_num_demos[promp_name] = num_demos
            self._metadata.promp_data_names[promp_name].append(data_name)

        # Resolve the data names for any merged ProMPs
        for promp_name_1, promp_name_2, new_promp in merged:
            for data_name in self._metadata.promp_data_names[promp_name_1]:
                if data_name not in self._metadata.promp_data_names[new_promp]:
                    self._metadata.promp_data_names[new_promp].append(data_name)
            for data_name in self._metadata.promp_data_names[promp_name_2]:
                if data_name not in self._metadata.promp_data_names[new_promp]:
                    self._metadata.promp_data_names[new_promp].append(data_name)

        # Kill the ones removed in the merge
        for removed_1, removed_2, _ in merged:
            del self._metadata.promp_num_demos[removed_1]
            del self._metadata.promp_data_names[removed_1]
            del self._metadata.promp_num_demos[removed_2]
            del self._metadata.promp_data_names[removed_2]
            self._metadata.promp_names.remove(removed_1)
            self._metadata.promp_names.remove(removed_2)

        return promp_names

    def get_promp_names(self):
        """
        Retrieves the names of ProMPs in the ProMP library.
        
        Returns:
            List(str): List of string names of ProMPs
        """
        return self.promp_library.get_promp_names()

    def get_promp(self, promp_name):
        """
        Retreives the ProMP denoted by the specified name.
        
        Args:
            promp_name (str): Name denoting ProMP in Library
        
        Returns:
            ProMP: The ProMP in the library denoted by the specified name.
        """
        return self.promp_library.get_promp(promp_name)

    def generate_task_instance(self):
        learning_type = self._metadata.active_learning_type
        if learning_type == _GRID:
            task_instance = self._generate_grid_instance()
        elif learning_type == _VALIDATION:
            task_instance = self._generate_validation_instance()
        elif self.ready_for_active_learning(display_msg=False):
            if learning_type == _MAX_ENTROPY:
                task_instance = self._generate_max_entropy_instance()
            elif learning_type == _LEAST_CONFIDENT:
                task_instance = self._generate_least_confident_instance()
            elif learning_type == _MIN_MARGIN:
                task_instance = self._generate_min_margin_instance()
            elif learning_type == _RANDOM:
                task_instance = self._generate_random_instance()
            elif learning_type == _MAHALANOBIS:
                task_instance = self._generate_mahalanobis_instance()
            else:
                rospy.logerr("Unknown active learning type: {}".format(learning_type))
        else:
            task_instance = self._generate_random_instance()

        self.selected_tasks.append(task_instance)
        return task_instance

    def ready_for_active_learning(self, display_msg=True):
        promp_ready = self.promp_library.is_ready()
        gmm_ready = self.obj_gmm.is_ready()
        multiple_promps_ready = False
        if self.promp_library.get_num_promps() > 1:
            multiple_promps_ready = True
            for name in self.promp_library.get_promp_names():
                promp = self.promp_library.get_promp(name)
                if promp.num_demos < self.promp_library.min_num_demos:
                    multiple_promps_ready = False

        if display_msg:
            msg = ""
            values = []
            if not promp_ready:
                msg += ("\nProMP not ready for active learning:\n"
                        "    Num ProMPs (1 needed):        {}\n"
                        "    Num Demos ({} needed):         {}\n")
                values.append(self.promp_library.get_num_promps())
                values.append(self.promp_library.min_num_demos)
                values.append(self.promp_library.num_demos)
            if not gmm_ready and not multiple_promps_ready:
                msg += ("GMM not ready for active learning:\n"
                        "    Num GMM Instances ({} needed): {}\n")
                values.append(self.obj_gmm.min_num_instances)
                values.append(self.obj_gmm.get_num_instances())
            if promp_ready and not gmm_ready and not multiple_promps_ready:
                msg += ("ProMPs not ready for active learning:\n"
                        "    Num ProMPs (2 needed):        {}\n"
                        "(Note: can also get negative instances for GMM)\n")
                values.append(self.promp_library.get_num_promps())
            if values:
                rospy.logwarn(msg.format(*values))

        return (promp_ready and gmm_ready) or multiple_promps_ready

    def has_unlabeled_instance(self):
        latest_instance = self.get_latest_instance()
        return self.selected_tasks and not latest_instance.has_label()

    def add_task_instance(self, task_instance):
        self.selected_tasks.append(task_instance)

    def discard_task_instance(self):
        if not self.has_unlabeled_instance():
            rospy.logwarn("No unlabeled task instance exists. Doing nothing.")
            return False
        else:
            self.selected_tasks.pop(-1)
            rospy.loginfo("Unlabeled task instance discarded.")
            return True

    def label_task_instance(self, label):
        if not self.has_unlabeled_instance():
            rospy.logwarn("No unlabeled task instance exists.")
            return False
        else:
            self.selected_tasks[-1].label = label
            rospy.loginfo("Labeled most recent task instance %d" % label)
            return True

    def get_latest_instance(self):
        instance = self.selected_tasks[-1] if self.selected_tasks else None
        return instance

    def set_instance_object_pose(self, pose):
        self.selected_tasks[-1].object_pose = pose

    def set_instance_trajectory_name(self, name):
        self.selected_tasks[-1].trajectory_name = name

    def set_instance_ee_pose_in_obj(self, pose):
        self.selected_tasks[-1].ee_pose_in_obj = pose

    def get_most_likely_promp(self, query_waypoints):
        return self.promp_library.get_most_likely_promp(query_waypoints)

    def get_obj_gmm_params(self):
        return self.obj_gmm.get_params()

    def get_ee_gmm_params(self):
        return self.ee_gmm.get_params()

    def get_promp_traj_params(self):
        return self.promp_library.get_traj_params()

    def get_ee_gmm_instances_in_base(self, obj_pose):
        pose_array = PoseArray()
        base_TF_object = util.tf_from_pose(obj_pose)
        for i in range(self.ee_gmm.task_instances.shape[0]):
            ee_in_obj = self.ee_gmm.task_instances[i, :]
            ee_in_base = util.TF_mu(base_TF_object, ee_in_obj)
            pose = Pose()
            pose.position.x = ee_in_base[0]
            pose.position.y = ee_in_base[1]
            pose.position.z = ee_in_base[2]
            pose.orientation.x = ee_in_base[3]
            pose.orientation.y = ee_in_base[4]
            pose.orientation.z = ee_in_base[5]
            pose.orientation.w = ee_in_base[6]
            pose_array.poses.append(pose)
        return pose_array

    def visualize_region(self):
        util.visualize_region(self)

    def write_metadata(self):
        self._metadata.update_datetime()
        self._metadata.num_demos = self.promp_library.num_demos
        self._metadata.num_promp_samples = self.promp_library.num_demos
        self._metadata.num_promps = self.promp_library.get_num_promps()
        self._metadata.num_obj_gmm_components = self.obj_gmm.get_num_components()
        self._metadata.num_ee_gmm_components = self.ee_gmm.get_num_components()
        self._metadata.num_tot_instances = len(self.task_candidates)
        self._metadata.num_pos_instances = len([i for i in self.selected_tasks if i.label == 1])
        self._metadata.num_neg_instances = len([i for i in self.selected_tasks if i.label == 0])

    def _update_values(self):
        """
        Update the probability and entropy values. Returns maximum entropy 
        value observed.
        """
        max_entropy = 0.0
        for candidate in self.task_candidates:
            ee_pos = candidate.goal_ee_pose.position
            obj_pos = candidate.goal_obj_pose.position
            # TODO these are only x-y position
            promp_query = np.array([ee_pos.x, ee_pos.y])
            gmm_query = np.array([obj_pos.x, obj_pos.y])
            promp_pdf = self.promp_library.pdf(promp_query, self.zero_lim)
            gmm_pdf = self.obj_gmm.pdf(gmm_query, self.zero_lim)
            pos_prob = promp_pdf / (promp_pdf + gmm_pdf)
            neg_prob = gmm_pdf / (promp_pdf + gmm_pdf)
            entropy = -((pos_prob * np.log(pos_prob)) + (neg_prob * np.log(neg_prob)))
            if entropy > max_entropy:
                max_entropy = entropy
            candidate.entropy = entropy
            candidate.pos_prob = pos_prob
            candidate.neg_prob = neg_prob
        return max_entropy

    def _generate_start_instance(self):
        """
        Generate an instance in the center of table for easy start.
        """
        instance = TaskInstance()
        instance.object_planar_pose = PlanarPose(x=0, y=0, z=0, theta=0)
        rospy.loginfo("Generated start instance")
        return instance

    def _generate_random_instance(self):
        x, y, z, theta = self._get_random_planar_pose()
        instance = TaskInstance()
        instance.object_planar_pose = PlanarPose(x=x, y=y, z=z, theta=theta)
        rospy.loginfo("Generated random instance")
        return instance

    def _generate_grid_instance(self):
        """
        Return the next instance from the finite grid (iterated over 
        sequentially in order of creation).
        """
        instance = None
        if not self.finite_goal_set:
            rospy.logwarn("Finite instances have not been generated!")
        elif self.grid_idx == len(self.finite_goal_set):
            rospy.logwarn("You're already done! No more instances to demonstrate.")
        else:
            instance = deepcopy(self.finite_goal_set[self.grid_idx])
            self.grid_idx += 1
        return instance

    def _generate_max_entropy_instance(self):
        rospy.loginfo("Requesting Max-Entropy instance...")
        req = self._create_optimization_request()
        req.active_learning_type = _MAX_ENTROPY
        for _ in range(self.num_solve_attempts):
            x, y, z, theta = self._get_random_planar_pose()
            guess = PlanarPose(x=x, y=y, z=z, theta=theta)
            req.init_guesses.append(guess)
        try:
            get_instance = rospy.ServiceProxy("/optimization/get_instance", GetOptimizationInstance)
            resp = get_instance(req)
        except rospy.ServiceException as e:
            rospy.logerr("Service request to get Max-Entropy instance failed: {}".format(e))
            return None

        max_entropy = -1.0
        selected_instance = None
        for instance in resp.selected_instances:
            entropy = self._evaluate_entropy(instance)
            if entropy > max_entropy:
                selected_instance = instance
                max_entropy = entropy

        chosen = TaskInstance()
        chosen.object_planar_pose = selected_instance
        rospy.logwarn("Generated Max-Entropy instance: Entropy = {}".format(max_entropy))
        return chosen

    def _generate_least_confident_instance(self):
        rospy.loginfo("Requesting Least-Confident instance...")
        req = self._create_optimization_request()
        req.active_learning_type = _LEAST_CONFIDENT

        for _ in range(self.num_solve_attempts):
            x, y, z, theta = self._get_random_planar_pose()
            guess = PlanarPose(x=x, y=y, z=z, theta=theta)
            req.init_guesses.append(guess)

        try:
            get_instance = rospy.ServiceProxy("/optimization/get_instance", GetOptimizationInstance)
            resp = get_instance(req)
        except rospy.ServiceException as e:
            rospy.logerr("Service request to get Least-Confident instance failed: {}".format(e))
            return None

        least_confidence = 1000.0
        selected_instance = None
        for instance in resp.selected_instances:
            confidence = self._evaluate_least_confident(instance)
            if confidence < least_confidence:
                selected_instance = instance
                least_confidence = confidence

        chosen = TaskInstance()
        chosen.object_planar_pose = selected_instance
        rospy.loginfo("Generated Least-Confident instance: Value = {}".format(least_confidence))
        return chosen

    def _generate_mahalanobis_instance(self):
        rospy.loginfo("Requesting Mahalanobis instance...")
        req = self._create_optimization_request()
        req.active_learning_type = _MAHALANOBIS

        for _ in range(self.num_solve_attempts):
            x, y, z, theta = self._get_random_planar_pose()
            guess = PlanarPose(x=x, y=y, z=z, theta=theta)
            req.init_guesses.append(guess)

        try:
            get_instance = rospy.ServiceProxy("/optimization/get_instance", GetOptimizationInstance)
            resp = get_instance(req)
        except rospy.ServiceException as e:
            rospy.logerr("Service request to get Least-Confident instance failed: {}".format(e))
            return None

        worst_mahal = 0.0
        selected_instance = None
        print "\n"
        for instance in resp.selected_instances:
            mahal = self.evaluate_mahalanobis(instance)
            print "INSTANCE", instance, "MAHALANOBIS", mahal
            if mahal > worst_mahal:
                selected_instance = instance
                worst_mahal = mahal
        print "\n"

        chosen = TaskInstance()
        chosen.object_planar_pose = selected_instance
        rospy.loginfo("Generated Mahalanobis instance: Value = {}".format(worst_mahal))
        return chosen

    def _generate_validation_instance(self):
        if not self.validation_set:
            rospy.logerr("Validation set was not generated.")
            return None
        if self.validation_idx == len(self.validation_set):
            rospy.logwarn("You're done!")
            return None
        rospy.loginfo("Generating validation instance {} of {}".format(
            self.validation_idx + 1, len(self.validation_set)))
        chosen = self.validation_set[self.validation_idx]
        self.validation_idx += 1
        return chosen

    def _generate_min_margin_instance(self):
        req = self._create_optimization_request()
        req.active_learning_type = _MIN_MARGIN
        for _ in range(self.num_solve_attempts):
            x, y, z, theta = self._get_random_planar_pose()
            guess = PlanarPose(x=x, y=y, z=z, theta=theta)
            req.init_guesses.append(guess)
        try:
            get_instance = rospy.ServiceProxy("/optimization/get_instance", GetOptimizationInstance)
            resp = get_instance(req)
        except rospy.ServiceException as e:
            rospy.logerr("Service request to get Min-Margin instance failed: {}".format(e))
            return None
        chosen = TaskInstance()
        chosen.object_planar_pose = resp.selected_instance
        rospy.loginfo("Generated Min-Margin instance.")
        return chosen

    def _get_random_planar_pose(self):
        x = np.random.uniform(self.config["x_min"], self.config["x_max"])
        y = np.random.uniform(self.config["y_min"], self.config["y_max"])
        z = self.config["z"]
        theta = np.random.uniform(self.config["theta_min"], self.config["theta_max"])
        return x, y, z, theta

    def _create_optimization_request(self):
        obj_gmm = self.obj_gmm.get_params()
        ee_gmm = self.ee_gmm.get_params()
        # Normalize quaternion
        for i in range(len(ee_gmm['means'])):
            ee_gmm['means'][i][3:] /= np.linalg.norm(ee_gmm['means'][i][3:])
        promps = self.promp_library.get_params()
        phi = self.promp_library.get_phi().T
        req = GetOptimizationInstanceRequest()
        req.obj_gmm = util.serialize_mixture(obj_gmm)
        req.ee_gmm = util.serialize_mixture(ee_gmm)
        req.promps = util.serialize_mixture(promps)
        req.phi = util.serialize_array(phi)
        req.table_pose.position.x = self.table_position[0]
        req.table_pose.position.y = self.table_position[1]
        req.table_pose.position.z = self.table_position[2]
        req.table_pose.orientation.x = self.table_quaternion[0]
        req.table_pose.orientation.y = self.table_quaternion[1]
        req.table_pose.orientation.z = self.table_quaternion[2]
        req.table_pose.orientation.w = self.table_quaternion[3]
        req.lower_bound.x = self.config["x_min"]
        req.lower_bound.y = self.config["y_min"]
        req.lower_bound.z = self.config["z"]
        req.lower_bound.theta = self.config["theta_min"]
        req.upper_bound.x = self.config["x_max"]
        req.upper_bound.y = self.config["y_max"]
        req.upper_bound.z = self.config["z"]
        req.upper_bound.theta = self.config["theta_max"]
        return req

    def _generate_finite_goal_set(self):
        x_min = self.config["x_min"]
        x_max = self.config["x_max"]
        y_min = self.config["y_min"]
        y_max = self.config["y_max"]
        theta_min = self.config["theta_min"]
        theta_max = self.config["theta_max"]
        num_xs = self.config["num_xs"]
        num_ys = self.config["num_ys"]
        num_thetas = self.config["num_thetas"]
        z = self.config["z"]
        table_pose = Pose()
        table_pose.position.x = self.table_position[0]
        table_pose.position.y = self.table_position[1]
        table_pose.position.z = self.table_position[2]
        table_pose.orientation.x = self.table_quaternion[0]
        table_pose.orientation.y = self.table_quaternion[1]
        table_pose.orientation.z = self.table_quaternion[2]
        table_pose.orientation.w = self.table_quaternion[3]

        self.finite_goal_set = []
        xs = np.linspace(x_min, x_max, num_xs)
        ys = np.linspace(y_min, y_max, num_ys)
        thetas = np.linspace(theta_min, theta_max, num_thetas, endpoint=False)

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                for k, theta in enumerate(thetas):
                    task_instance = TaskInstance()
                    task_instance.object_planar_pose = PlanarPose(x=x, y=y, z=z, theta=theta)
                    task_instance.grid_coords = (i, j, k)
                    task_instance.table_pose = table_pose
                    self.finite_goal_set.append(task_instance)

        # TEST with visualization
        rospy.wait_for_service("/visualization/visualize_poses")
        poses = []
        for t in self.finite_goal_set:
            pose = Pose()
            obj_pose = t.object_planar_pose
            q = tf.quaternion_from_euler(0, 0, obj_pose.theta)
            pose.position.x = obj_pose.x
            pose.position.y = obj_pose.y
            pose.position.z = obj_pose.z
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            poses.append(pose)
        visualize_poses("/visualization/visualize_poses", [pose for pose in poses[:1]],
                        base_link="table_center")

    def setup_as_validation(self, config_path, config_filename):
        self._metadata.active_learning_type = _VALIDATION
        self.validation_idx = 0
        config = util.load_config(config_path, config_filename)
        self.validation_set = []
        table_pose = Pose()
        table_pose.position.x = self.table_position[0]
        table_pose.position.y = self.table_position[1]
        table_pose.position.z = self.table_position[2]
        table_pose.orientation.x = self.table_quaternion[0]
        table_pose.orientation.y = self.table_quaternion[1]
        table_pose.orientation.z = self.table_quaternion[2]
        table_pose.orientation.w = self.table_quaternion[3]
        for pose_key in config.keys():
            instance = TaskInstance()
            instance.table_pose = table_pose
            instance.object_planar_pose.x = config[pose_key]["x"]
            instance.object_planar_pose.y = config[pose_key]["y"]
            instance.object_planar_pose.z = config[pose_key]["z"]
            instance.object_planar_pose.theta = config[pose_key]["theta"]
            self.validation_set.append(instance)

    def _evaluate_entropy(self, planar_pose):
        probs = self._get_probabilities(planar_pose)
        entropy = 0
        # print "\n"
        for prob in probs:
            entropy -= prob * np.log(prob)
        # print "ENTROPY", entropy, "\n"
        return entropy  # , probs

    def evaluate_least_confident(self, planar_pose, condition=True):
        """
        Trying to find the instance whose max probability over class labels is
        at a minimum. So once you have probabilities, you return the max of those,
        and that is considered the probability of its best label.
        """
        probs = self._get_probabilities(planar_pose, condition)
        best_label_prob = max(probs)
        return best_label_prob

    def evaluate_min_margin(self, planar_pose, condition=True):
        probs = sorted(self._get_probabilities(planar_pose, condition))
        if len(probs) > 1:
            margin = probs[-1] - probs[-2]  # best minus second best
        else:
            margin = 0
        return margin

    def evaluate_mahalanobis(self, planar_pose, condition=False):
        dists = self._get_mahalanobis_dists(planar_pose, condition)
        best = min(dists)
        return best

    def evaluate_kl_divergence(self, planar_pose, div_type):
        divs = self._get_kl_divergences(planar_pose, div_type)
        best = min(divs)
        return best

    def evaluate_promp_prob(self, planar_pose, condition=False):
        pdfs = self._get_pdfs(planar_pose, condition)
        # Assuming for now there is only the ProMP pdf included and not neg
        return pdfs[0]

    def _evaluate_success(self, planar_pose):
        # probs = self._get_probabilities(planar_pose)
        pdfs = self._get_pdfs(planar_pose)
        pos = sum(pdfs[:-1])
        return pos

    def _get_probabilities(self, planar_pose, condition=True):
        pdfs = self._get_pdfs(planar_pose, condition)
        zero_lim = 1e-15
        pdfs = [max(pdf, zero_lim) for pdf in pdfs]
        total = sum(pdfs)
        probs = [pdf / total for pdf in pdfs]
        return probs

    def _get_pdfs(self, planar_pose, condition=True):
        promp_params = self.promp_library.get_params()
        promp_params = zip(promp_params['weights'], promp_params['means'], promp_params['covs'])
        ee_gmm_params = self.ee_gmm.get_params()
        obj_gmm_params = self.obj_gmm.get_params()
        if obj_gmm_params:
            obj_gmm_params = zip(obj_gmm_params['weights'], obj_gmm_params['means'],
                                 obj_gmm_params['covs'])
        # Normalize quaternion
        for i in range(len(ee_gmm_params['means'])):
            ee_gmm_params['means'][i][3:] /= np.linalg.norm(ee_gmm_params['means'][i][3:])
        ee_gmm_params = zip(ee_gmm_params['weights'], ee_gmm_params['means'], ee_gmm_params['covs'])
        phi = self.promp_library.get_phi().T

        base_TF_object = util.table_planar_to_base_tf(planar_pose.x, planar_pose.y,
                                                      planar_pose.z, planar_pose.theta,
                                                      self.table_position, self.table_quaternion)
        pdfs = []
        for pi_j, mu_j, sigma_j in promp_params:
            # Sum weighted PDF based on EE in obj components
            ee_pdfs = []
            for beta_r, object_mu_ee, object_sigma_ee in ee_gmm_params:
                # Transform EE mean and cov pose in object frame to base frame
                base_mu_ee = util.TF_mu(base_TF_object, object_mu_ee)
                base_sigma_ee = util.TF_sigma(base_TF_object, object_sigma_ee)

                if condition:
                    L_j = np.dot(
                        np.dot(sigma_j, phi),
                        np.linalg.inv(base_sigma_ee +
                                      np.dot(phi.T, np.dot(sigma_j, phi))))
                    mu_new_j = mu_j + np.dot(L_j,
                                             base_mu_ee - np.dot(phi.T, mu_j))
                    sigma_new_j = sigma_j - np.dot(L_j, np.dot(phi.T, sigma_j))
                    mu = np.dot(phi.T, mu_new_j)
                    sigma = np.dot(phi.T, np.dot(sigma_new_j, phi))
                else:
                    mu = np.dot(phi.T, mu_j)
                    sigma = np.dot(phi.T, np.dot(sigma_j, phi))

                g = torch.distributions.MultivariateNormal(
                    loc=torch.from_numpy(mu),
                    covariance_matrix=torch.from_numpy(sigma))
                log_pdf = g.log_prob(base_mu_ee).item()
                pdf = np.exp(log_pdf)

                ee_pdfs.append(beta_r * pdf)

            pdfs.append(pi_j * sum(ee_pdfs))

        if len(obj_gmm_params) > 0:
            obj_pdfs = []
            # Only taking x, y, theta, because we assume they're already
            # all sitting in the same plane, and the only difference
            # in z would come out of sensor noise. Basically, don't let
            # that factor into the probability computations
            query = [planar_pose.x, planar_pose.y, planar_pose.theta]
            for alpha_k, mu_k, sigma_k in obj_gmm_params:

                g = torch.distributions.MultivariateNormal(
                    loc=torch.from_numpy(mu_k),
                    covariance_matrix=torch.from_numpy(sigma_k))
                query = torch.from_numpy(np.array(query))
                log_pdf = g.log_prob(query).item()
                obj_pdf = np.exp(log_pdf)

                # try:
                #     mvn = multivariate_normal(mean=mu_k, cov=sigma_k)
                #     obj_pdf = mvn.pdf(query)
                # except np.linalg.LinAlgError:
                #     obj_pdf = self.zero_lim
                obj_pdfs.append(alpha_k * obj_pdf)
            pdfs.append(sum(obj_pdfs))

        return pdfs

    def _get_mahalanobis_dists(self, planar_pose, condition=True):
        promp_params = self.promp_library.get_params()
        promp_params = zip(promp_params['weights'], promp_params['means'], promp_params['covs'])
        ee_gmm_params = self.ee_gmm.get_params()

        # Normalize quaternion
        for i in range(len(ee_gmm_params['means'])):
            ee_gmm_params['means'][i][3:] /= np.linalg.norm(ee_gmm_params['means'][i][3:])
        ee_gmm_params = zip(ee_gmm_params['weights'], ee_gmm_params['means'], ee_gmm_params['covs'])
        phi = self.promp_library.get_phi().T

        base_TF_object = util.table_planar_to_base_tf(planar_pose.x, planar_pose.y,
                                                      planar_pose.z, planar_pose.theta,
                                                      self.table_position, self.table_quaternion)
        promp_dists = []
        for _, mu_j, sigma_j in promp_params:
            # Sum weighted PDF based on EE in obj components
            waypoint_dists = []
            for _, object_mu_ee, object_sigma_ee in ee_gmm_params:
                # Transform EE mean and cov pose in object frame to base frame
                base_mu_ee = util.TF_mu(base_TF_object, object_mu_ee)
                base_sigma_ee = util.TF_sigma(base_TF_object, object_sigma_ee)

                if condition:
                    L_j = np.dot(
                        np.dot(sigma_j, phi),
                        np.linalg.inv(base_sigma_ee +
                                      np.dot(phi.T, np.dot(sigma_j, phi))))
                    mu_new_j = mu_j + np.dot(L_j,
                                             base_mu_ee - np.dot(phi.T, mu_j))
                    sigma_new_j = sigma_j - np.dot(L_j, np.dot(phi.T, sigma_j))
                    mu = np.dot(phi.T, mu_new_j)
                    sigma = np.dot(phi.T, np.dot(sigma_new_j, phi))
                    dist = mahalanobis_distance(base_mu_ee, mu, sigma)
                else:
                    mu = np.dot(phi.T, mu_j)
                    sigma = np.dot(phi.T, np.dot(sigma_j, phi))
                    dist = mahalanobis_distance(base_mu_ee, mu, sigma)

                waypoint_dists.append(dist)

            promp_dist = min(waypoint_dists)
            promp_dists.append(promp_dist)

        return promp_dists

    def _get_kl_divergences(self, planar_pose, div_type="cond_ee"):
        promp_params = self.promp_library.get_params()
        promp_params = zip(promp_params['weights'], promp_params['means'], promp_params['covs'])
        ee_gmm_params = self.ee_gmm.get_params()

        # Normalize quaternion
        for i in range(len(ee_gmm_params['means'])):
            ee_gmm_params['means'][i][3:] /= np.linalg.norm(ee_gmm_params['means'][i][3:])
        ee_gmm_params = zip(ee_gmm_params['weights'], ee_gmm_params['means'], ee_gmm_params['covs'])
        phi = self.promp_library.get_phi().T

        base_TF_object = util.table_planar_to_base_tf(planar_pose.x, planar_pose.y,
                                                      planar_pose.z, planar_pose.theta,
                                                      self.table_position, self.table_quaternion)
        promp_dists = []
        for _, mu_j, sigma_j in promp_params:
            # Sum weighted PDF based on EE in obj components
            waypoint_dists = []
            for _, object_mu_ee, object_sigma_ee in ee_gmm_params:
                # Transform EE mean and cov pose in object frame to base frame
                base_mu_ee = util.TF_mu(base_TF_object, object_mu_ee)
                base_sigma_ee = util.TF_sigma(base_TF_object, object_sigma_ee)

                L_j = np.dot(
                    np.dot(sigma_j, phi),
                    np.linalg.inv(base_sigma_ee +
                                  np.dot(phi.T, np.dot(sigma_j, phi))))
                mu_new_j = mu_j + np.dot(L_j, base_mu_ee - np.dot(phi.T, mu_j))
                sigma_new_j = sigma_j - np.dot(L_j, np.dot(phi.T, sigma_j))

                mu_orig = np.dot(phi.T, mu_j)
                sigma_orig = np.dot(phi.T, np.dot(sigma_j, phi))
                mu_cond = np.dot(phi.T, mu_new_j)
                sigma_cond = np.dot(phi.T, np.dot(sigma_new_j, phi))

                g_ee = torch.distributions.MultivariateNormal(
                    loc=torch.from_numpy(base_mu_ee),
                    covariance_matrix=torch.from_numpy(base_sigma_ee))
                g_orig = torch.distributions.MultivariateNormal(
                    loc=torch.from_numpy(mu_orig),
                    covariance_matrix=torch.from_numpy(sigma_orig))
                g_cond = torch.distributions.MultivariateNormal(
                    loc=torch.from_numpy(mu_cond),
                    covariance_matrix=torch.from_numpy(sigma_cond))

                if div_type == "kl_cond_ee":
                    dist = kl_divergence(g_cond, g_ee).item()
                elif div_type == "kl_ee_cond":
                    dist = kl_divergence(g_ee, g_cond).item()
                else:
                    rospy.logerr(
                        "Unknown divergence type: {}".format(div_type))
                # print ""
                # print "EE --- ORIG", kl_divergence(g1, g2).item()
                # print "ORIG --- EE", kl_divergence(g2, g1).item()
                # print "EE --- COND", kl_divergence(g1, g3).item()
                # print "COND --- EE", kl_divergence(g3, g1).item()
                # print "ORIG --- COND", kl_divergence(g2, g3).item()
                # print "COND --- ORIG", kl_divergence(g3, g2).item()

                # dist = kl_divergence(g1, g3).item()  # EE COND
                # dist = kl_divergence(g3, g1).item()  # COND EE

                waypoint_dists.append(dist)

            promp_dist = min(waypoint_dists)
            promp_dists.append(promp_dist)

        return promp_dists

    def _set_table_pose(self):
        # TODO account for other experiments
        if self._metadata.experiment_type == "experiment_1":
            pose = self.config["lbr4_base_link__to__table_1_center"]
        elif self._metadata.experiment_type == "experiment_2":
            pose = self.config["lbr4_base_link__to__table_2_center"]
        else:
            rospy.logerr("Unknown experiment type: {}".format(
                self._metadata.experiment_type))
            return None
        self.table_position = [pose["position"]["x"], pose["position"]["y"], pose["position"]["z"]]
        self.table_quaternion = [pose["orientation"]["x"], pose["orientation"]["y"],
                                 pose["orientation"]["z"], pose["orientation"]["w"]]

    def _generate_visualizations(self):
        self._update_values()
        idx = util.increment_file_index(self.img_path, "img.index")
        util.visualize_value(self, idx, "entropy")
        util.visualize_value(self, idx, "pos_prob")
        util.visualize_value(self, idx, "neg_prob")
        util.visualize_gmm(self, idx)
        util.visualize_scatter(self, idx)

    def visualize_poses(self):
        rospy.loginfo("Waiting for pose visualization service...")
        rospy.wait_for_service("/visualization/visualize_poses")
        rospy.loginfo("Service found!")
        poses = []
        for t in self.finite_goal_set:
            pose = Pose()
            obj_pose = t.object_planar_pose
            q = tf.quaternion_from_euler(0, 0, obj_pose.theta)
            pose.position.x = obj_pose.x
            pose.position.y = obj_pose.y
            pose.position.z = obj_pose.z
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            poses.append(pose)
        visualize_poses("/visualization/visualize_poses", poses, base_link="table_1_center")
