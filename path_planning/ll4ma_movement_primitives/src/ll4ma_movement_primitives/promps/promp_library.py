#!/usr/bin/env python
import os
import sys
import numpy as np
import rospy
import random
import math
import torch
from torch.distributions.kl import kl_divergence
from random import shuffle
from copy import copy, deepcopy
from geometry_msgs.msg import Pose
from ll4ma_movement_primitives.promps import ProMP, mahalanobis_distance
from scipy.stats import multivariate_normal
from rospy_service_helper import visualize_poses


class ProMPLibrary:
    """
    Implements a library of ProMPs.

    Intended for use in active learning of ProMPs, as it uses similarity
    measures to automatically determine which ProMP in the library a new
    demonstration should be added to.
    """
    
    def __init__(self):
        self.promps = {}
        self.demo_counts = {}
        self.num_demos = 0
        self.promp_idx = 0  # For assigning unique keys to created ProMPs
        self.min_num_demos = 1
        self.use_cov_transfer = rospy.get_param("/use_transfer_cov", False)
        self.use_merging = rospy.get_param("/use_promp_merging", False)

    def add_demo(self, w, config=None, display_msg=True):
        """
        Add demo to the specified ProMP if key is provided, otherwise use a 
        similarity measure based on mahalanobis distance in weight space to 
        determine which ProMP the demo can be incorporated into. For the  first 
        sample, use should provide at least 3 demos so at least one ProMP exists 
        with a decent MLE for transfer and probability computation.
        """
        merged = []
        if self.num_demos < self.min_num_demos and self.get_num_promps() == 1:
            promp_key = self._add_demo_to_first_promp(w, display_msg)
            promp_keys = [promp_key]
        else:
            promp_keys = self._add_demo_to_similar_promp(w, display_msg)

        # If necessary, create new ProMP or handle merging
        if len(promp_keys) == 0:
            promp_key = self._add_demo_to_new_promp(w, config, display_msg)
            promp_keys = [promp_key]
        elif self.use_merging and len(promp_keys) > 1:
            new_promps, merged = self._merge_promps(promp_keys)
            # Get rid of the removed ones from the library
            for promp_key_1, promp_key_2, _ in merged:
                del self.promps[promp_key_1]
                del self.promps[promp_key_2]
                promp_keys.remove(promp_key_1)
                promp_keys.remove(promp_key_2)
            promp_keys += new_promps

        # Update logging info
        for promp_key in promp_keys:
            self.demo_counts[promp_key] = self.get_promp(promp_key).num_demos
            self.num_demos += 1

        return promp_keys, merged

    def get_promp(self, promp_key):
        promp = None
        if promp_key not in self.promps.keys():
            rospy.loginfo("Unknown ProMP: %s" % promp_key)
        else:
            promp = self.promps[promp_key]
        return promp

    def get_most_likely_promp(self, query_waypoints):
        """
        Returns the ProMP most capable (i.e. with highest probability of success)
        to achieve task.
        
        Args:
            query_waypoints (list): List of potential Waypoints to condition on
        
        Returns:
            promp (ProMP): ProMP to achieve the task
            waypoint (Waypoint): Best waypoint ProMP should be conditioned on
        """
        if not self.is_ready():
            rospy.logwarn("Not ready to make likelihood determinations.")
            return None

        least_value = sys.maxint
        best_promp_name = ""
        best_waypoint = None
        best_waypoint_idx = -1
        for promp_key in self.get_promp_names():
            promp = self.get_promp(promp_key)
            for i, waypoint in enumerate(query_waypoints):

                # mu_orig, sigma_orig = promp.get_traj_params(
                #     w_keys=waypoint.condition_keys)
                mu_cond, sigma_cond = promp.get_traj_params(
                    w_keys=waypoint.condition_keys, waypoints=[waypoint])

                # g_ee = torch.distributions.MultivariateNormal(
                #     loc=torch.from_numpy(waypoint.values),
                #     covariance_matrix=torch.from_numpy(waypoint.sigma))
                # g_orig = torch.distributions.MultivariateNormal(
                #     loc=torch.from_numpy(mu_orig),
                #     covariance_matrix=torch.from_numpy(sigma_orig))
                # g_cond = torch.distributions.MultivariateNormal(
                #     loc=torch.from_numpy(mu_cond),
                #     covariance_matrix=torch.from_numpy(sigma_cond))

                # value = kl_divergence(g_cond, g_ee).item()
                # value = kl_divergence(g_ee, g_cond).item()
                # print "VALUE", value

                value = mahalanobis_distance(waypoint.values, mu_cond,
                                             sigma_cond)

                # value = mahalanobis_distance(waypoint.values, mu_orig,
                #                              sigma_orig)

                if value < least_value:
                    least_value = value
                    best_promp_name = promp_key
                    best_waypoint = waypoint
                    best_waypoint_idx = i
        rospy.loginfo(
            "Chose ProMP '{}' waypoint {} of {} with value {}".format(
                best_promp_name, best_waypoint_idx, len(query_waypoints),
                least_value))

        return best_promp_name, best_waypoint

    def get_promp_names(self):
        return self.promps.keys()

    def get_num_promps(self):
        return len(self.promps.keys())

    def get_params(self):
        params = {"weights": [], "means": [], "covs": []}
        for promp_name in self.get_promp_names():
            promp = self.get_promp(promp_name)
            params["weights"].append(1.0 / self.get_num_promps())
            params["means"].append(promp.get_mu_w())
            params["covs"].append(promp.get_sigma_w())
        return params

    def get_traj_params(self):
        params = {'means': [], 'covs': []}
        for name in self.get_promp_names():
            promp = self.get_promp(name)
            mu, sigma = promp.get_traj_params()
            params['means'].append(mu)
            params['covs'].append(sigma)
        return params

    def get_phi(self, x=1.0):
        phi, _ = self.get_promp(self.get_promp_names()[0]).get_block_phi(x)
        return phi

    def is_ready(self):
        enough_promps = self.get_num_promps() > 0
        enough_demos = self.num_demos >= self.min_num_demos
        return enough_promps and enough_demos

    def pdf(self, query, zero_lim=1e-25, w_keys=[]):
        """
        Computes PDF of query point w.r.t. ProMP library by computing the PDF 
        for each ProMP and weighting them according to similarity based on
        Mahalanobis distance between query and each ProMP distribution.
        """
        promp_pdf = 0.0
        pdfs = []
        dists = []
        for promp_key in self.get_promp_names():
            pdf, dist = self.get_promp(promp_key).pdf(
                query, x=1.0, zero_lim=zero_lim, w_keys=w_keys)
            pdfs.append(pdf)
            dists.append(dist)
        total_dist = sum(dists)
        for pdf, dist in zip(pdfs, dists):
            promp_pdf += (1.0 - (dist / total_dist)) * pdf
        return max(promp_pdf, zero_lim)

    def _add_demo_to_first_promp(self, w, display_msg=True):
        """
        Returns name of ProMP it was added to for logging purpose.
        """
        promp_key = self.promps.keys()[0]
        promp = self.promps[promp_key]
        # Calling this just to show values at console:
        promp.should_add_demo(w)
        promp.add_demo(w, display_msg)
        return promp_key

    def _add_demo_to_similar_promp(self, w, display_msg=True):
        # TODO trying add to only one (most similar). Was getting a lot of
        # overlap between ProMPs which might be messing with the active learning
        promp_keys = []
        best_value = sys.maxint
        best_promp = None
        best_promp_key = None
        for promp_key in self.promps.keys():
            if display_msg:
                rospy.loginfo("Testing demo similarity for '%s'" % promp_key)
            promp = self.promps[promp_key]
            mu_w = promp.get_mu_w()
            sigma_w = promp.get_sigma_w()
            dist = mahalanobis_distance(w, mu_w, sigma_w)
            if dist < promp.fixed_mahalanobis_threshold and dist < best_value:
                best_value = dist
                best_promp = promp
                best_promp_key = promp_key
            # if promp.should_add_demo(w, display_msg=display_msg):
            #     if display_msg:
            #         rospy.loginfo(
            #             "Adding new demonstration to existing ProMP '{}'".
            #             format(promp_key))
            #     promp.add_demo(w, display_msg)
            #     promp_keys.append(promp_key)
        if best_promp:
            if display_msg:
                rospy.loginfo(
                    "Adding new demonstration to existing ProMP '{}'".format(
                        best_promp_key))
            promp_keys = [best_promp_key]
            best_promp.add_demo(w, display_msg)
        return promp_keys

    def _add_demo_to_new_promp(self, w, config, display_msg=True):
        promp_key = "promp_%d" % self.promp_idx
        self.promp_idx += 1
        if display_msg:
            rospy.loginfo("Creating new ProMP '%s' from a single demonstration"
                          % promp_key)

        # Do covariance transfer if enabled
        num_promps = self.get_num_promps()
        if self.use_cov_transfer and num_promps == 1:
            rospy.loginfo("Transferring cov from one other ProMP")
            other_promp = self.get_promp(self.get_promp_names()[0])
            config.sigma_w_prior = other_promp.get_sigma_w()
        elif self.use_cov_transfer and num_promps > 1:
            config.sigma_w_prior = np.zeros((len(w), len(w)))
            # Compute convex combination of covariances where coefficients
            # are determined by mahalanobis distance between demo and
            # each ProMP (normalized over all such values)
            promp_names = self.get_promp_names()
            dists = []
            sigma_ws = []
            for promp_name in promp_names:
                promp = self.get_promp(promp_name)
                # Only transfer if it has more than just a prior
                if promp.num_demos > 1:
                    mu_w = promp.get_mu_w()
                    sigma_w = promp.get_sigma_w()
                    dist = mahalanobis_distance(w, mu_w, sigma_w)
                    dists.append(dist)
                    sigma_ws.append(sigma_w)
            total = sum(dists)
            for i, dist in enumerate(dists):
                coef = 1.0 - (dist / total)
                config.sigma_w_prior += coef * sigma_ws[i]
            rospy.loginfo("Transferred cov from {} ProMPs".format(len(dists)))

        promp = ProMP(config=config)
        promp.add_demo(w, display_msg)
        self.promps[promp_key] = promp
        self.demo_counts[promp_key] = 0
        return promp_key

    def _merge_promps(self, promp_keys):
        keys = promp_keys[:]
        new_promps = []
        merged = []
        # TODO assuming fixed threshold for now
        threshold = self.get_promp(promp_keys[0]).fixed_mahalanobis_threshold

        performed_merge = False
        while len(keys) > 1:
            sims = []
            for key in keys:
                other_keys = [k for k in keys if k != key]
                for other_key in other_keys:
                    # Check if KEY ProMP can be merged into OTHER_KEY ProMP
                    promp = self.get_promp(key)
                    other_promp = self.get_promp(other_key)
                    mu = promp.get_mu_w()
                    other_mu = other_promp.get_mu_w()
                    other_sigma = other_promp.get_sigma_w()
                    dist = mahalanobis_distance(mu, other_mu, other_sigma)
                    sims.append((key, other_key, dist))
            sims.sort(key=lambda x: x[-1])  # Descending by sim value
            best = sims[0]
            if best[-1] > threshold:
                break

            # Merge FIRST into SECOND
            merged_promp_key = self._merge_promp_pair(best[0], best[1])
            merged.append((best[0], best[1], merged_promp_key))
            if best[0] in new_promps:
                new_promps.remove(best[0])
            if best[1] in new_promps:
                new_promps.remove(best[1])
            keys.append(merged_promp_key)
            keys.remove(best[0])
            keys.remove(best[1])
            new_promps.append(merged_promp_key)
            performed_merge = True

        if performed_merge:
            rospy.loginfo("MERGE was performed")
        else:
            rospy.logwarn("NO MERGE OCCURRED")
        return new_promps, merged

    def _merge_promp_pair(self, key_1, key_2):
        # TODO this is a bit of a hack, should probably have a way of doing
        # this with knowledge of which trajectories the learned weights come
        # from. Alas, deadlines...
        promp_1 = self.get_promp(key_1)
        promp_2 = self.get_promp(key_2)
        new_promp_key = "promp_%d" % self.promp_idx
        self.promp_idx += 1
        new_promp = deepcopy(promp_1)
        for i in range(promp_2.w.shape[0]):
            w = promp_2.w[i, :]
            can_add = True
            for j in range(promp_1.w.shape[0]):
                all_close = np.allclose(w, promp_1.w[j, :])
                can_add = can_add and not all_close
            if can_add:
                new_promp.add_demo(w)
        self.promps[new_promp_key] = new_promp
        rospy.loginfo("Merged '{}' and '{}' to make new ProMP '{}'".format(
            key_1, key_2, new_promp_key))
        return new_promp_key
