import os
import sys
import errno
import casadi as ca
import numpy as np
import cPickle as pickle
from copy import deepcopy
from ll4ma_movement_primitives.util import casadi_util as ca_util

_OBJ_POSE_DOF = 4
_POSE_3D_DOF = 7
_WEIGHT_DOF = 10


class Optimizer:
    def __init__(self, table_pos, table_quat, lbx, ubx):
        """
        Optimizer using CasADi for continuous optimization.

        Args:
            table_pos (list): Position of table with respect to base
            table_quat (list): Quaternion of table with respect to base
            lbx (list): Lower bounds on planar pose values
            ubx (list): Upper bounds on planar pose values
        """
        self.table_pos = table_pos
        self.table_quat = table_quat
        self.lbx = lbx
        self.ubx = ubx
        self.zero_lim = 1e-4

        # TEMPORARY
        self.mahalanobis_test_func = None

        # Setup PDF over object pose
        self.obj_pose = ca.SX.sym('obj_pose', 3)
        self.obj_mu = ca.SX.sym('obj_mu', 3)
        self.obj_sigma = ca.SX.sym('obj_sigma', 3, 3)
        self.obj_pdf = ca_util.gaussian_pdf(self.obj_pose, self.obj_mu,
                                            self.obj_sigma, self.zero_lim)
        self.obj_gaussian = ca.Function('obj_gaussian', [self.obj_pose, self.obj_mu, self.obj_sigma],
                                        [self.obj_pdf], ['obj_pose', 'obj_mu', 'obj_sigma'],
                                        ['obj_pdf'])

        # Setup Gaussian PDF over ProMP pose
        self.promp_pose = ca.SX.sym('promp_pose', _POSE_3D_DOF)
        self.promp_mu = ca.SX.sym('promp_mu', _POSE_3D_DOF)
        self.promp_sigma = ca.SX.sym('promp_sigma', _POSE_3D_DOF, _POSE_3D_DOF)
        self.promp_pdf = ca_util.gaussian_pdf(self.promp_pose, self.promp_mu, self.promp_sigma,
                                              self.zero_lim)
        self.promp_gaussian = ca.Function('promp_gaussian',
                                          [self.promp_pose, self.promp_mu, self.promp_sigma],
                                          [self.promp_pdf], ['promp_pose', 'promp_mu', 'promp_sigma'],
                                          ['promp_pdf'])

        self.mahal_val = ca_util.mahalanobis_distance(
            self.promp_pose, self.promp_mu, self.promp_sigma)
        self.mahal_func = ca.Function('mahal_distance',
                                      [self.promp_pose, self.promp_mu, self.promp_sigma],
                                      [self.mahal_val], ['promp_pose', 'promp_mu', 'promp_sigma'],
                                      ['mahal_dist'])

    def max_entropy_objective(self, obj_pose, phi, promps, ee_gmm, obj_gmm=None):
        probs = self._get_probabilities(obj_pose, phi, promps, ee_gmm, obj_gmm)
        num_terms = probs.shape[0]

        # Make sure none of them are behaving badly around zero
        for i in range(num_terms):
            probs[i] = ca.fmax(1e-15, probs[i])

        neg_entropies = ca.SX.sym("neg_entropies", num_terms)
        for i in range(num_terms):
            neg_entropies[i] = probs[i] * ca.log(probs[i])
        neg_entropy = ca.sum1(neg_entropies)
        return neg_entropy

    def least_confident_objective(self, obj_pose, phi, promps, ee_gmm, obj_gmm=None):
        """
        Minimize confidence. Want the instance whose most likely label has the
        lowest probability over all instances, so the objective function per pose
        is the max over label probabilities.
        """
        probs = self._get_probabilities(obj_pose, phi, promps, ee_gmm, obj_gmm)
        max_prob = ca.mmax(probs)
        return max_prob

    def mahalanobis_objective(self, obj_pose, phi, promps, ee_gmm, obj_gmm=None, use_approx_min=True):
        """
        Compute mahalanobis distance between resulting EE pose before and after
        conditioning. Want the one that is closest before conditioning and also
        changing the least after conditioning.
        """
        dists = self._get_mahalanobis_dists(obj_pose, phi, promps, ee_gmm)
        # We have minimum distances over waypoints for each ProMP, so take the
        # min over this to get the min over all ProMPs, this would be the best
        # ProMP/Waypoint for this instance.
        if use_approx_min:
            dist = self._continuous_min_approx(dists)
        else:
            dist = ca.mmin(dists)

        # But now we want to know the worst over all instances, so negate it.
        dist *= -1.0
        return dist

    # def min_margin_objective(self, obj_pose, phi, gmm_obj, promps, gmm_ee):
    #     """
    #     Minimize the margin between the most likely label and the second most
    #     likely label, i.e. the difference between them.
    #     """
    #     probs = self._get_probabilities(obj_pose, phi, gmm_obj, promps, gmm_ee)
    #     # Convert list of probabilities to casadi symbolic matrix
    #     ca_probs = ca.SX.sym("ca_probs", len(probs))
    #     for i in range(len(probs)):
    #         ca_probs[i] = probs[i]
    #     # Get the max probability
    #     max_prob = ca.mmax(ca_probs)
    #     # With this initialization, guarantee not to be max, unless the max
    #     # value has duplicates, in which case it really doesn't matter since
    #     # the difference is zero either way:
    #     second = ca.fmin(probs[0], probs[1])
    #     for prob in probs:
    #         maybe = ca.if_else(ca.gt(max_prob, prob), prob, second)
    #         second = ca.if_else(ca.gt(maybe, second), maybe, second)
    #     return max_prob - second

    def get_max_entropy_instance(self, init_guesses, phi, promps, ee_gmm, obj_gmm=None):
        obj_pose = ca.SX.sym("obj_pose", _OBJ_POSE_DOF)
        print "Getting objective function..."
        objective = self.max_entropy_objective(obj_pose, phi, promps, ee_gmm, obj_gmm)
        print "Setting up optimization problem..."
        opts = {"ipopt": {"max_iter": 500}}
        problem = {'x': obj_pose, 'f': objective}
        solver = ca.nlpsol("nlp", "ipopt", problem, opts)
        print "Solving the optimization..."
        solns = []
        for init_guess in init_guesses:
            soln = solver(lbx=self.lbx, ubx=self.ubx, x0=init_guess)['x'].full().flatten()
            solns.append(soln)
        return solns

    def get_least_confident_instance(self, init_guesses, phi, promps, ee_gmm, obj_gmm=None):
        obj_pose = ca.SX.sym("obj_pose", _OBJ_POSE_DOF)
        print "Getting objective function..."
        objective = self.least_confident_objective(obj_pose, phi, promps, ee_gmm, obj_gmm)
        print "Setting up optimization problem..."
        opts = {"ipopt": {"max_iter": 500}}
        problem = {'x': obj_pose, 'f': objective}
        solver = ca.nlpsol("nlp", "ipopt", problem, opts)
        print "Solving the optimization..."
        solns = []
        for init_guess in init_guesses:
            soln = solver(lbx=self.lbx, ubx=self.ubx, x0=init_guess)['x'].full().flatten()
            solns.append(soln)
        return solns

    def get_mahalanobis_instance(self, init_guesses, phi, promps, ee_gmm, obj_gmm=None):
        obj_pose = ca.SX.sym("obj_pose", _OBJ_POSE_DOF)
        print "Getting objective function..."
        objective = self.mahalanobis_objective(obj_pose, phi, promps, ee_gmm, obj_gmm)
        print "Setting up optimization problem..."
        opts = {
            "ipopt": {
                "max_iter": 5000,
                "derivative_test": "first-order",
                "hessian_approximation": "limited-memory"
            }
        }
        problem = {'x': obj_pose, 'f': objective}
        solver = ca.nlpsol("nlp", "ipopt", problem, opts)
        print "Solving the optimization..."
        solns = []
        for init_guess in init_guesses:
            soln = solver(lbx=self.lbx, ubx=self.ubx, x0=init_guess)['x'].full().flatten()
            solns.append(soln)
        return solns

    # def get_min_margin_instance(self, phi, gmm_obj, promps, gmm_ee):
    #     ca_pose = ca.SX.sym("obj_pose", _OBJ_POSE_DOF)
    #     min_margin = self.min_margin_objective(ca_pose, phi, gmm_obj, promps,
    #                                            gmm_ee)
    #     problem = {'x': ca_pose, 'f': min_margin}
    #     solver = ca.nlpsol("nlp", "ipopt", problem)
    #     soln = solver(lbx=self.lbx, ubx=self.ubx)['x'].full().flatten()
    #     return soln

    def evaluate_entropy(self, obj_pose, phi, promps, ee_gmm, obj_gmm=None):
        sym_pose = ca.SX.sym("sym_pose", _OBJ_POSE_DOF)
        objective = self.max_entropy_objective(sym_pose, phi, promps, ee_gmm, obj_gmm)
        F = ca.Function('f', [sym_pose], [objective], ['sym_pose'], ['neg_entropy'])
        val = F(sym_pose=obj_pose)['neg_entropy'].full().flatten()[0]
        val = -val  # Negative entropy was computed, return entropy
        print "OPTIMIZER ENTROPY", val
        return val

    def evaluate_pdfs(self, obj_pose, phi, promps, ee_gmm, obj_gmm=None):
        sym_pose = ca.SX.sym("sym_pose", _OBJ_POSE_DOF)
        pdfs = self._get_pdfs(sym_pose, phi, promps, ee_gmm, obj_gmm)
        for i in range(pdfs.shape[0]):
            pdf = pdfs[i]
            pdf_func = ca.Function("pdf", [sym_pose], [pdf], ["sym_pose"], ["output"])
            pdf_val = pdf_func(sym_pose=obj_pose)["output"].full().flatten()[0]
            print "OPTIMIZER PDF", pdf_val

    def evaluate_mahalanobis(self, obj_pose, phi, promps, ee_gmm, obj_gmm=None):
        """
        TODO this is set up to evaluate offline by caching the function
        """
        if self.mahalanobis_test_func is None:
            sym_pose = ca.SX.sym("sym_pose", _OBJ_POSE_DOF)
            objective = self.mahalanobis_objective(sym_pose, phi, promps, ee_gmm, obj_gmm)
            self.mahalanobis_test_func = ca.Function('f', [sym_pose], [objective],
                                                     ['sym_pose'], ['result'])
        val = self.mahalanobis_test_func(sym_pose=obj_pose)['result'].full().flatten()[0]
        # Negating since we were optimizing negation
        val = -val
        return val

    def get_mahalanobis_function(self, phi, promps, ee_gmm, obj_gmm=None):
        """
        This is intended for external use in order to get a symbolic function
        that can be computed more quickly. It negates the usual value since
        in CasADi we want to run minimization, but this function should return
        the real value.
        """
        sym_pose = ca.SX.sym("sym_pose", _OBJ_POSE_DOF)
        # Negating so the function makes sense in and of itself
        objective = -self.mahalanobis_objective(sym_pose, phi, promps, ee_gmm, obj_gmm)
        function = ca.Function('f', [sym_pose], [objective], ['planar_pose'], ['result'])
        return function

    def get_entropy_function(self, phi, promps, ee_gmm, obj_gmm=None):
        """
        For external use for faster computation.
        """
        sym_pose = ca.SX.sym("sym_pose", _OBJ_POSE_DOF)
        # Negating since the objective is negative entropy (being minimized)
        objective = -self.max_entropy_objective(sym_pose, phi, promps, ee_gmm, obj_gmm)
        function = ca.Function('f', [sym_pose], [objective], ['planar_pose'], ['result'])
        return function

    def get_least_confident_function(self, phi, promps, ee_gmm, obj_gmm=None):
        """
        For external use for faster computation.
        """
        sym_pose = ca.SX.sym("sym_pose", _OBJ_POSE_DOF)
        # Negating since the objective is negative entropy (being minimized)
        objective = self.least_confident_objective(sym_pose, phi, promps, ee_gmm, obj_gmm)
        function = ca.Function('f', [sym_pose], [objective], ['planar_pose'], ['result'])
        return function

    # def evaluate_least_confident(self, obj_pose, phi, gmm_obj, promps, gmm_ee):
    #     ca_pose = ca.SX.sym('pose', _OBJ_POSE_DOF)
    #     least_confident = self.least_confident_objective(
    #         ca_pose, phi, gmm_obj, promps, gmm_ee)
    #     F = ca.Function('f', [ca_pose], [least_confident], ['obj_pose'],
    #                     ['least_confident'])
    #     val = F(obj_pose=obj_pose)['least_confident'].full().flatten()[0]
    #     return val

    # def evaluate_min_margin(self, obj_pose, phi, gmm_obj, promps, gmm_ee):
    #     ca_pose = ca.SX.sym('pose', _OBJ_POSE_DOF)
    #     min_margin = self.min_margin_objective(ca_pose, phi, gmm_obj, promps,
    #                                            gmm_ee)
    #     F = ca.Function('f', [ca_pose], [min_margin], ['obj_pose'],
    #                     ['min_margin'])
    #     val = F(obj_pose=obj_pose)['min_margin'].full().flatten()[0]
    #     return val

    # def evaluate_promp_pdf(self, obj_pose, phi, promps, gmm_ee):
    #     ca_pose = ca.SX.sym('pose', _OBJ_POSE_DOF)
    #     pdfs = self.promp_pdf_objective(ca_pose, phi, promps, gmm_ee)
    #     F = ca.Function('f', [ca_pose], [pdfs], ['obj_pose'], ['pdfs'])
    #     vals = F(obj_pose=obj_pose)['pdfs']
    #     # TODO this probably won't be an intelligible type, in the end want a
    #     # list of values but not sure what will be the function output
    #     return vals

    # def evaluate_gmm_pdf(self, obj_pose, gmm_obj):
    #     ca_pose = ca.SX.sym('pose', _OBJ_POSE_DOF)
    #     pdf = self.gmm_pdf_objective(ca_pose, gmm_obj)
    #     F = ca.Function('f', [ca_pose], [pdf], ['obj_pose'], ['pdf'])
    #     val = F(obj_pose=obj_pose)['pdf'].full().flatten()[0]
    #     return val

    def _get_probabilities(self, obj_pose, phi, promps, ee_gmm, obj_gmm):
        pdfs = self._get_pdfs(obj_pose, phi, promps, ee_gmm, obj_gmm)
        total = ca.sum1(pdfs)
        num_probs = pdfs.shape[0]
        probs = ca.SX.sym("probs", num_probs)
        for i in range(num_probs):
            probs[i] = pdfs[i] / total
        return probs

    def _get_pdfs(self, obj_pose, phi, promps, ee_gmm, obj_gmm):
        promp_params = zip(promps['weights'], promps['means'], promps['covs'])
        ee_gmm_params = zip(ee_gmm['weights'], ee_gmm['means'], ee_gmm['covs'])
        num_promps = len(promp_params)
        num_ee_gmm = len(ee_gmm_params)
        if obj_gmm:
            obj_gmm_params = zip(obj_gmm['weights'], obj_gmm['means'], obj_gmm['covs'])
            num_obj_gmm = len(obj_gmm_params)
        else:
            num_obj_gmm = 0

        # Set the transform of object with respect to base frame
        base_TF_object = ca_util.table_planar_to_base_tf(obj_pose[0], obj_pose[1], obj_pose[2],
                                                         obj_pose[3], self.table_pos, self.table_quat)
        if num_obj_gmm > 0:
            pdfs = ca.SX.sym("pdfs", num_promps + 1)
        else:
            pdfs = ca.SX.sym("pdfs", num_promps)

        # Build the ProMP PDFs
        for j, (pi_j, mu_j, sigma_j) in enumerate(promp_params):
            # Sum weighted PDF based on EE in obj components
            ee_pdfs = ca.SX.sym("ee_pdfs_{}".format(j), num_ee_gmm)
            for r, (beta_r, object_mu_ee, object_sigma_ee) in enumerate(ee_gmm_params):
                # Transform EE mean and cov pose in object frame to base frame
                base_mu_ee = ca_util.TF_mu(base_TF_object, object_mu_ee, "{}_{}".format(str(j),
                                                                                        str(r)))
                base_sigma_ee = ca_util.TF_sigma(base_TF_object, object_sigma_ee)

                # Condition ProMP distributions on transformed point
                L_j = ca.mtimes(ca.mtimes(sigma_j, phi),
                                ca.inv(base_sigma_ee +
                                       ca.mtimes(phi.T, ca.mtimes(sigma_j, phi))))
                mu_new_j = mu_j + ca.mtimes(L_j, base_mu_ee - ca.mtimes(phi.T, mu_j))
                sigma_new_j = sigma_j - ca.mtimes(L_j, ca.mtimes(phi.T, sigma_j))

                # Compute the PDF value for this ProMP
                mu = ca.mtimes(phi.T, mu_new_j)
                sigma = ca.mtimes(phi.T, ca.mtimes(sigma_new_j, phi))
                # mu = ca.mtimes(phi.T, mu_j)
                # sigma = ca.mtimes(phi.T, ca.mtimes(sigma_j, phi))
                ee_pdf = self.promp_gaussian(base_mu_ee, mu, sigma)

                # ee_pdf = ca.fmax(self.zero_lim, ee_pdf)
                ee_pdf *= beta_r
                ee_pdfs[r] = ee_pdf

            promp_pdf = ca.sum1(ee_pdfs)
            promp_pdf *= pi_j
            # promp_pdf = ca.fmax(self.zero_lim, promp_pdf)
            pdfs[j] = promp_pdf

        # Object GMM PDF
        if num_obj_gmm > 0:
            obj_pdfs = ca.SX.sym("obj_gmm_pdf", num_obj_gmm)
            for k, (alpha_k, mu_k, sigma_k) in enumerate(obj_gmm_params):
                # Ignoring z, because it's constant once the plane is given,
                # so it really shouldn't factor into probability computations
                query = ca.SX.sym("obj_gmm_query", 3)
                query[0] = obj_pose[0]
                query[1] = obj_pose[1]
                query[2] = obj_pose[3]
                # query = ca.SX([obj_pose[0], obj_pose[1], obj_pose[3]])
                obj_pdf = self.obj_gaussian(query, mu_k, sigma_k)
                obj_pdf *= alpha_k
                obj_pdfs[k] = obj_pdf
            obj_gmm_pdf = ca.sum1(obj_pdfs)
            obj_gmm_pdf = ca.fmax(self.zero_lim, obj_gmm_pdf)
            pdfs[-1] = obj_gmm_pdf

        return pdfs

    def _get_mahalanobis_dists(self, obj_pose, phi, promps, ee_gmm, condition=True,
                               use_approx_min=True):
        promp_params = zip(promps['weights'], promps['means'], promps['covs'])
        ee_gmm_params = zip(ee_gmm['weights'], ee_gmm['means'], ee_gmm['covs'])
        num_promps = len(promp_params)
        num_ee_gmm = len(ee_gmm_params)

        # Set the transform of object with respect to base frame
        base_TF_object = ca_util.table_planar_to_base_tf(obj_pose[0], obj_pose[1], obj_pose[2],
                                                         obj_pose[3], self.table_pos, self.table_quat)

        promp_dists = ca.SX.sym("mahal_dists", num_promps)

        # Build the ProMP PDFs
        for j, (_, mu_j, sigma_j) in enumerate(promp_params):
            # Sum weighted PDF based on EE in obj components
            waypoint_dists = ca.SX.sym("waypoint_dists_{}".format(j), num_ee_gmm)
            for r, (_, object_mu_ee, object_sigma_ee) in enumerate(ee_gmm_params):
                # Transform EE mean and cov pose in object frame to base frame
                base_mu_ee = ca_util.TF_mu(base_TF_object, object_mu_ee, "{}_{}".format(str(j),
                                                                                        str(r)))
                base_sigma_ee = ca_util.TF_sigma(base_TF_object, object_sigma_ee)

                if condition:
                    L_j = ca.mtimes(ca.mtimes(sigma_j, phi),
                                    ca.inv(base_sigma_ee +
                                           ca.mtimes(phi.T, ca.mtimes(sigma_j, phi))))
                    mu_new_j = mu_j + ca.mtimes(L_j, base_mu_ee - ca.mtimes(phi.T, mu_j))
                    sigma_new_j = sigma_j - ca.mtimes(L_j, ca.mtimes(phi.T, sigma_j))
                    mu = ca.mtimes(phi.T, mu_new_j)
                    sigma = ca.mtimes(phi.T, ca.mtimes(sigma_new_j, phi))
                else:
                    mu = ca.mtimes(phi.T, mu_j)
                    sigma = ca.mtimes(phi.T, ca.mtimes(sigma_j, phi))

                waypoint_dists[r] = self.mahal_func(base_mu_ee, mu, sigma)

            if use_approx_min:
                pass
            else:
                promp_dist = ca.mmin(waypoint_dists)
            promp_dists[j] = promp_dist

        return promp_dists

    def _continuous_min_approx(self, vec, rho=5000):
        """
        Smooth differentiable approximation to the min function.

        See: https://mathoverflow.net/questions/35191
        """
        s = ca.sum1(ca.expm(-rho * vec))
        value = -(1.0 / rho) * ca.log(s)
        return value
