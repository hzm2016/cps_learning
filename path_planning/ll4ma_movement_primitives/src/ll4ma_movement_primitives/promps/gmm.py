import rospy
import numpy as np
from tf import transformations as tf
from sklearn.mixture import BayesianGaussianMixture as DPGMM
from scipy.stats import multivariate_normal


class GMM:
    """
    Wrapper class for the sklearn GMM.

    This class is intended to be used in active learning of ProMPs for
    characterizing negative task instances as well as capturing the
    distribution of end-effector poses in a local frame (e.g. the object
    frame).
    """

    def __init__(self, min_num_instances=2, weight_cutoff=1e-2, max_components=10):
        self.gmm = DPGMM()
        self.task_instances = None
        self.min_num_instances = min_num_instances
        self.weight_cutoff = weight_cutoff
        self.max_components = max_components

    def learn(self, display_msg=True):
        if self.get_num_instances() >= self.min_num_instances:
            if display_msg:
                rospy.loginfo("Fitting GMM...")
            # self.gmm.n_components = min(self.task_instances.shape[0],
            #                             self.max_components)

            # TODO trying max of 10 so that the optimization scales a little better
            # self.gmm.n_components = min(10, self.task_instances.shape[0])
            # self.gmm.n_components = self.task_instances.shape[0]
            self.gmm.fit(self.task_instances)
            if display_msg:
                rospy.loginfo("GMM successfully fit with %d instances." %
                              self.task_instances.shape[0])
        else:
            if display_msg:
                rospy.logwarn("Only one instance, waiting for another to fit GMM")

    def pdf(self, query, zero_lim=1e-25):
        if not self.is_fit():
            rospy.logwarn("GMM has not been fit yet.")
            return None

        ws = self.get_weights()
        means = self.get_means()
        covs = self.get_covariances()

        p = 0.0
        for w, mean, cov in zip(ws, means, covs):
            p += w * self._component_pdf(query, mean, cov, zero_lim)
        return max(p, zero_lim)

    def _component_pdf(self, x, mu, sigma, zero_lim=1e-25):
        try:
            mvn = multivariate_normal(mean=mu, cov=sigma)
            return max(mvn.pdf(x), zero_lim)
        except np.linalg.LinAlgError:
            return zero_lim

    def add_instance(self, instance, display_msg=True):
        if isinstance(instance, list):
            instance = np.array(instance).reshape(1, -1)
        if self.task_instances is None:
            self.task_instances = instance.reshape(1, -1)
            if display_msg:
                rospy.loginfo("New instance successfully added.")
        else:
            num_instances, dim = self.task_instances.shape
            instance = instance.reshape(1, -1)
            if instance.shape[1] != dim:
                rospy.logerr("Instance dimensions do not match:\n"
                             "Expected: {}\nActual: {}".format(dim, instance.shape[1]))
            else:
                self.task_instances = np.vstack((self.task_instances, instance))
                if display_msg:
                    rospy.loginfo("New instance successfully added. Num instances: {}".format(
                        num_instances + 1))

    def get_num_instances(self):
        num = 0 if self.task_instances is None else self.task_instances.shape[0]
        return num

    def get_num_components(self):
        return 0 if not self.is_fit() else len(self.get_weights())

    def get_nonzero_indices(self):
        """
        Due to the Dirichlet prior, it won't always use the total number of
        components provided and sets unused ones to have small weights. So
        we only consider effective components to be those with weight values
        above the specified threshold.
        """
        return np.where(self.gmm.weights_ > self.weight_cutoff)

    def get_weights(self):
        if not self.is_fit():
            # rospy.logwarn("GMM has not been learned yet.")
            return []
        idxs = self.get_nonzero_indices()
        return self.gmm.weights_[idxs]

    def get_means(self):
        if not self.is_fit():
            # rospy.logwarn("GMM has not been learned yet.")
            return []
        idxs = self.get_nonzero_indices()
        return self.gmm.means_[idxs]

    def get_covariances(self):
        if not self.is_fit():
            # rospy.logwarn("GMM has not been learned yet.")
            return []
        idxs = self.get_nonzero_indices()
        return self.gmm.covariances_[idxs]

    def get_params(self):
        params = {}
        if not self.is_fit():
            pass
            # rospy.logwarn("GMM has not been learned yet.")
        else:
            params['weights'] = self.get_weights()
            params['means'] = self.get_means()
            params['covs'] = self.get_covariances()
        return params

    def is_fit(self):
        """
        Returns True if GMM model has been fit. Tests if it has the means_
        attribute as that is set iff the model has been fit.
        """
        try:
            _ = self.gmm.means_
            return True
        except AttributeError:
            return False

    def is_ready(self):
        enough_instances = self.get_num_instances() >= self.min_num_instances
        return enough_instances and self.is_fit()
