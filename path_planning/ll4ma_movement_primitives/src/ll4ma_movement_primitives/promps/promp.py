# import rospy
import numpy as np
from copy import copy
from sklearn import linear_model
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal
from src.ll4ma_movement_primitives.promps import (
    ProMPConfig, mahalanobis_distance, reject_outliers, damped_pinv)
from src.ll4ma_movement_primitives.phase_variables import LinearPV
from src.ll4ma_movement_primitives.basis_functions import GaussianLinearBFS


class ProMP:  
    """
    Implements a Probabilistic Movement Primitive (ProMP).

    See [1] for technical details.

    [1] Paraschos, A., Daniel, C., Peters, J., & Neumann, G. (2018). 
        Using probabilistic movement primitives in robotics. 
        Autonomous Robots, 42(3), 529-551.  
    """
    
    def __init__(self, phase=None, config=ProMPConfig()):
        self.num_bfs = config.num_bfs
        self.regr_alpha = config.regr_alpha
        self.mu_w_mle = config.mu_w_mle
        self.sigma_w_mle = config.sigma_w_mle
        self.sigma_w_prior = config.sigma_w_prior
        self.inv_wish_factor = config.inv_wish_factor
        self.dist_threshold = config.dist_threshold
        self.name = config.name  
        self.state_types = config.state_types
        self.dimensions = config.dimensions
        self.w_keys = config.w_keys
        self.num_demos = int(config.num_demos)
        self.init_joint_state = config.init_joint_state
        self.fixed_mahalanobis_threshold = config.fixed_mahalanobis_threshold
        self.num_dims = len([d for dim_list in self.dimensions for d in dim_list])
        self.phase = LinearPV() if phase is None else phase
        self.bfs = GaussianLinearBFS(num_bfs=self.num_bfs)
        self.w = None  # Will be incrementally stacked vertically

        if not np.any(self.sigma_w_prior):
            self.sigma_w_prior = np.eye(self.num_dims * self.num_bfs) * 0.001

    def learn_from_demos(self, demos, dt=1.0, duration=None, regr=None):
        for demo in demos:
            w = self.learn_weights(demo, dt, duration, regr)
            self.w = w if self.w is None else np.vstack((self.w, w))
            self.num_demos += 1
        # MLE for weight distribution (sample mean and covariance)
        self.update_w_mle()  

    def update_w_mle(self, display_msg=True):
        """
        Computes the sample mean and covariance given the current weights learned 
        from demos. If we only have one demo, set the mean as the sole weight 
        vector and use the provided prior on the covariance.
        """
        if self.num_demos == 1:
            self.mu_w_mle = np.copy(self.w)
            if display_msg:  
                # rospy.loginfo("Updated weights MLE over 1 demo (set demo as mean)")
                print("Updated weights MLE over 1 demo (set demo as mean)")
        else:
            # Sample mean
            self.mu_w_mle = np.mean(self.w, axis=0).reshape(1, -1)
            # Cache the current covariance for prior in MAP estimation
            # self.sigma_w_prior = self.get_sigma_w()

            # Sample covariance
            self.sigma_w_mle = np.zeros((self.num_dims * self.num_bfs, self.num_dims * self.num_bfs))

            for i in range(self.num_demos):
                sample_w = self.w[i, :].reshape(1, -1)
                self.sigma_w_mle += np.outer(sample_w - self.mu_w_mle, sample_w - self.mu_w_mle)
            self.sigma_w_mle /= self.num_demos
            if display_msg:
                # rospy.loginfo("Updated weights MLE over {} demos.".format(self.num_demos))
                print("Updated weights MLE over {} demos.".format(self.num_demos))

    def learn_weights(self, demo, dt=1.0, duration=None, regr=None):
        """
        Learn weight vector from provided demo using linear regression.
        
        Args:
            demo (ndarray): Demonstration to learn weights from
            dt (float, optional): Defaults to 1.0. Timestep separating data 
            points in demonstration.
            duration (float, optional): Defaults to None. Duration of demo.
            regr (linear_model, optional): Defaults to None. Regressor 
            implementing the linear_model interface from sklearn.
        
        Returns:
            ndarray: Learned weight vector
        """
        if regr is None:
            regr = linear_model.Ridge(self.regr_alpha, fit_intercept=False)
        duration = max(demo.shape) / self.num_dims
        # Generate rollouts of phase variable and basis functions
        self.phase.max_time = duration
        self.bfs.max_time = duration
        self.phase.reset()
        self.bfs.reset()
        xs = self.phase.get_rollout(dt)
        phi, _ = self.get_block_phi(xs)
        # Compute weights using regression
        regr.fit(phi.T, demo)
        w = regr.coef_.flatten()
        return w

    def add_demo(self, w, display_msg=True):
        """
        Add demo as learned weight vector. Updates the MLEs for the
        distribution mean and covariance with the new sample.
        
        Args:
            w (ndarray): Weight vector learned from demo to add to ProMP
        """
        self.w = w if self.w is None else np.vstack((self.w, w))
        self.num_demos += 1
        self.update_w_mle(display_msg)

    def generate_trajectory(self, dt=0.1, duration=20.0, mean=False, waypoints=[]):
        """
        Generates a rollout trajectory from current distribution parameters.

        Args:
            dt (float, optional): Defaults to 0.1. Timestep between consecutive
            trajectory points
            duration (float, optional): Defaults to 20.0. Duration of the 
            generated trajectory
            mean (bool, optional): Defaults to False. Mean trajectory is
            generated if true, otherwise a random weight sample is used
            waypoints (list, optional): Defaults to []. Waypoints to be
            conditioned on (of type Waypoint)
        
        Returns:
            traj (dict): Hierarchical dictionary with State Type > Dim > 
            ndarray of trajectory structure.
        """
        self.phase.max_time = duration
        self.bfs.max_time = duration
        self.phase.reset()
        self.bfs.reset()

        xs = self.phase.get_rollout(dt)
        phi, _ = self.get_block_phi(xs)
        if mean:
            sample_w = self.get_mu_w()
        else:
            if waypoints:
                mu_w, sigma_w = self.get_w_params_conditioned(waypoints)
            else:
                mu_w = self.get_mu_w()
                sigma_w = self.get_sigma_w()
            sample_w = np.random.multivariate_normal(mu_w, sigma_w)

        traj_vec = np.dot(phi.T, sample_w)
        traj_components = np.split(traj_vec, self.num_dims)
        traj = {}
        for i, state_type in enumerate(self.state_types):
            traj[state_type] = {}
            for dim in self.dimensions[i]:
                traj[state_type][dim] = traj_components.pop(0)
        return traj

    def get_w_matrix(self):
        """
        Returns the current matrix of accumulated weight vectors from learned
        demonstrations.
        
        Returns:
            ndarray: Matrix of weight vectors, size (num_instances, dims*num_bfs)
        """
        return np.copy(self.w)

    def get_mu_w(self):
        """
        Returns the current MLE for the mean of the weight vectors learned from
        demonstrations.
        
        Returns:
            ndarray: Mean weight vector
        """
        return np.copy(self.mu_w_mle).flatten()

    def get_sigma_w(self, delta=0.9):
        """
        Computes the covariance of the weight vectors as a convex combination
        of the current Maximum Likelihood Estimate (MLE) and the previous 
        estimate. This follows the approach of "Context-Driven Movement 
        Primitive Adaptation" - Wilbers et al. ICRA 2017.

            delta (float, optional): Defaults to 0.9. Trade-off parameter
            governing the convex combination of prior and current estimates.
            Must be in range [0,1], where 0 is full current MLE and 1 is full
            prior estimate.
        
        Returns:
            ndarray: The computed convex combination of covariance estimates.
        """
        # TODO do you want to keep it like this as function of demos?
        delta = np.exp(-self.num_demos)
        if self.sigma_w_mle is None:
            sigma_w = np.copy(self.sigma_w_prior)
        else:
            # TODO the prior right now is just the original. Did try carry over
            # like Wilbers paper but original prior should be sufficient.
            prior = delta * np.copy(self.sigma_w_prior)
            mle = (1.0 - delta) * np.copy(self.sigma_w_mle)
            sigma_w = prior + mle
        return sigma_w

    def get_w_params_conditioned(self, waypoints=[]):
        """
        Returns the distribution parameters (mean and cov) of the posterior 
        after conditioning on the provided waypoints. Uses the closed form
        Gaussian updates from [1].

        Args:
            waypoints (list(Waypoint), optional): Defaults to []. List of
            Waypoints the distribution is to be conditioned on.
        
        Returns:
            mu_w (ndarray): Posterior mean after conditioning
            sigma_w (ndarray): Posterior cov after conditioning
        """
        mu_w = self.get_mu_w()
        sigma_w = self.get_sigma_w()
        for waypoint in waypoints:
            block_phi, _ = self.get_block_phi(waypoint.phase_val, waypoint.condition_keys)
            sigma_w_phi = np.dot(sigma_w, block_phi.T)  # (ND,D)
            phi_sigma_w = np.dot(block_phi, sigma_w)  # (D,ND)
            gen_inv = np.linalg.pinv(waypoint.sigma + np.dot(phi_sigma_w, block_phi.T))  # (D,D)
            L = np.dot(sigma_w_phi, gen_inv)  # See Eq. 27 of [1]

            # Read out the waypoint values into their right place in the vector
            if not self.w_keys:
                # rospy.logerr("No weight keys. Something isn't right.")
                print("No weight keys. Something isn't right.")
                return None
            wpt_values = []
            for w_key in self.w_keys:
                if w_key in waypoint.condition_keys:
                    idx = waypoint.condition_keys.index(w_key)
                    value = waypoint.values[idx]
                    wpt_values.append(value)
                else:
                    wpt_values.append(0.0)

            # Update mean and covariance for weights
            mu_w += np.dot(L, wpt_values - np.dot(block_phi, mu_w))  # Eq. 25 of [1]
            sigma_w -= np.dot(L, phi_sigma_w)  # Eq. 26 of [1]
        return mu_w, sigma_w

    def get_traj_params(self, x=1.0, w_keys=[], waypoints=[]):
        """
        Computes the trajectory distribution in state space by mapping through
        weight space with the current weight space distribution parameters.
        Distribution parameters are computed for a specified timestep (as a 
        value of the phase variable).

        Args:
            x (float, optional): Defaults to 1.0. Phase variable value at which
            to compute the distribution parameters.
            w_keys (list, optional): Defaults to []. List of index keys for the
            weight space vectors (allows taking subset of weight space)
            waypoints (list, optional): Defaults to []. List of any Waypoints
            to be conditoned on.
        
        Returns:
            mu_traj (ndarray): Mean in state space
            sigma_traj (ndarray): Cov in state space
        """
        phi, keys = self.get_block_phi(x, condition_keys=w_keys)
        if waypoints:
            mu_w, sigma_w = self.get_w_params_conditioned(waypoints)
        else:
            mu_w = self.get_mu_w()
            sigma_w = self.get_sigma_w()
        mu_traj = np.dot(phi, mu_w)
        sigma_traj = np.dot(phi, np.dot(sigma_w, phi.T))
        idxs = [keys.index(w_key) for w_key in w_keys]
        return mu_traj[idxs], sigma_traj[idxs, :][:, idxs]

    def compute_kl_divergence(self, waypoints=[]):
        # mu_1, sigma_1 = self.get_w_dist_conditioned(waypoints)
        # mu_2 = self.get_mu_w()
        # sigma_2 = self.get_sigma_w()
        mu_1 = self.get_mu_w()
        sigma_1 = self.get_sigma_w()
        mu_2, sigma_2 = self.get_w_params_conditioned(waypoints)
        s1, logdet_1 = np.linalg.slogdet(sigma_1)
        s2, logdet_2 = np.linalg.slogdet(sigma_2)
        logdet_1 *= s1
        logdet_2 *= s2
        # sigma_2_inv = np.linalg.inv(sigma_2)
        sigma_2_inv = damped_pinv(sigma_2)
        t1 = logdet_2 - logdet_1
        t2 = len(mu_1)
        # print "SIGMA INV", sigma_2_inv
        t3 = np.trace(np.dot(sigma_2_inv, sigma_1))
        t4 = np.dot((mu_2 - mu_1).T, np.dot(sigma_2_inv, mu_2 - mu_1))
        # print "T1", t1
        # print "T2", t2
        # print "T3", t3
        # print "T4", t4
        kl = 0.5 * (t1 - t2 + t3 + t4)
        return kl

    def get_config(self):
        """
        Returns the ProMPConfig containing copies of all the data elements
        on this instance.
        
        Returns:
            ProMPConfig: Configuration element for this instance
        """
        config = ProMPConfig()
        config.num_bfs = np.copy(self.num_bfs)
        config.regr_alpha = np.copy(self.regr_alpha)
        config.mu_w_mle = np.copy(self.mu_w_mle)
        config.sigma_w_mle = np.copy(self.sigma_w_mle)
        config.sigma_w_prior = np.copy(self.sigma_w_prior)
        config.inv_wish_factor = np.copy(self.inv_wish_factor)
        config.dist_threshold = np.copy(self.dist_threshold)
        config.name = np.copy(self.name)
        config.state_types = np.copy(self.state_types)
        config.dimensions = np.copy(self.dimensions)
        config.w_keys = np.copy(self.w_keys)
        config.num_demos = np.copy(self.num_demos)
        config.num_dims = np.copy(self.num_dims)
        config.init_joint_state = np.copy(self.init_joint_state)
        return config

    def get_block_phi(self, x, condition_keys=[]):
        # Full rollout if phase series passed in, otherwise it's just one value
        if isinstance(x, np.ndarray):
            bfs = self.bfs.get_rollout(x)
        else:
            bfs = self.bfs.get_value(x)

        bfs_list = []
        keys = []
        for i, state_type in enumerate(self.state_types):
            for dim in self.dimensions[i]:
                dim = int(dim)
                key_s = "%s.%s" % (state_type, dim)
                keep_bfs = not condition_keys or key_s in condition_keys
                bfs_s = bfs if keep_bfs else np.zeros(bfs.shape)
                bfs_list.append(bfs_s)
                keys.append(key_s)
        block_phi = block_diag(*bfs_list)
        return block_phi, keys

    def should_add_demo(self, w, use_fixed_threshold=True, display_msg=True):
        # Update Mahalanobis distance threshold that determines if we should add this demo
        if use_fixed_threshold:
            threshold = self.fixed_mahalanobis_threshold
        else:
            # TODO why is this being set on instance?
            self.compute_mahalanobis_threshold()
            threshold = self.dist_threshold
        mu_w = self.get_mu_w()
        sigma_w = self.get_sigma_w()

        dist = mahalanobis_distance(w, mu_w, sigma_w)

        if dist > threshold:  
            if display_msg:  
                # rospy.logwarn("Demonstration is too different:\n"
                #               "   Mahalanobis Distance: %f\n"
                #               "              Threshold: %f" % (dist, threshold))
                print("Demonstration is too different:\n"
                              "   Mahalanobis Distance: %f\n"
                              "              Threshold: %f" % (dist, threshold))
            return False
        else:
            if display_msg:
                # rospy.loginfo("Demonstration can be incorporated!\n"
                #               "   Mahalanobis Distance: %f\n"
                #               "              Threshold: %f" % (dist, threshold))
                print("Demonstration can be incorporated!\n"
                              "   Mahalanobis Distance: %f\n"
                              "              Threshold: %f" % (dist, threshold))
            return True

    def compute_mahalanobis_threshold(self, num_samples=200):
        """
        Compute Mahalanobis distance threshold in weight space by generating a 
        bunch of samples.
        """
        # rospy.loginfo("Computing Mahalanobis distance threshold...")
        dists = []
        mu_w = self.get_mu_w()
        sigma_w = self.get_sigma_w()

        for _ in range(num_samples):
            w = np.random.multivariate_normal(mu_w, sigma_w)
            dist = mahalanobis_distance(w, mu_w, sigma_w)
            dists.append(dist)

        dists = np.array(dists)
        dists_no_outliers = reject_outliers(dists)
        self.dist_threshold = np.max(dists_no_outliers)
        # rospy.loginfo("Threshold computed: " + str(self.dist_threshold))

    def pdf(self, query, x, zero_lim=1e-50, w_keys=[], waypoints=[]):
        """
        Compute the PDF of query point. Truncates at zero_lim to allow for 
        meaningful uncertainty sampling computation in active learning.

        Returns both PDF value and Mahalanobis distance for similarity computations.
        """
        mu, sigma = self.get_traj_params(x=x, w_keys=w_keys, waypoints=waypoints)
        mahalanobis = mahalanobis_distance(query, mu, sigma)
        try:
            mvn = multivariate_normal(mean=mu, cov=sigma)
            pdf = mvn.pdf(query)
            pdf = max(pdf, zero_lim)
            return pdf, mahalanobis
        except np.linalg.LinAlgError:
            return zero_lim, mahalanobis
