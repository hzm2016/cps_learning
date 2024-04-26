#!/usr/bin/env python
import time
import rospy
import numpy as np
from ll4ma_movement_primitives.promps import active_learner_util as al_util
from ll4ma_movement_primitives.promps import Optimizer
from ll4ma_policy_learning.srv import GetOptimizationInstance, GetOptimizationInstanceResponse
from ll4ma_movement_primitives.promps import _MAX_ENTROPY, _LEAST_CONFIDENT, _MIN_MARGIN, _MAHALANOBIS
from ll4ma_policy_learning.msg import PlanarPose


class OptimizerNode:
    """
    Service provider for CasADi optimization functionality.
    """

    def __init__(self):
        self.entropy_srv = rospy.Service("/optimization/get_instance", GetOptimizationInstance,
                                         self.get_instance)
        self.rate = rospy.Rate(100)

    def run(self):
        rospy.loginfo("Ready to receive optimization requests.")
        while not rospy.is_shutdown():
            self.rate.sleep()

    def get_instance(self, req):
        if req.active_learning_type == _MAX_ENTROPY:
            return self._get_max_entropy_instance(req)
        elif req.active_learning_type == _LEAST_CONFIDENT:
            return self._get_least_confident_instance(req)
        elif req.active_learning_type == _MIN_MARGIN:
            return self._get_min_margin_instance(req)
        elif req.active_learning_type == _MAHALANOBIS:
            return self._get_mahalanobis_instance(req)
        else:
            rospy.logerr("Unknown active learning type: {}".format(req.active_learning_type))
            return GetOptimizationInstanceResponse(success=False)

    def _get_max_entropy_instance(self, request):
        rospy.loginfo("Generating Max-Entropy instance...")
        # Setup the optimizer
        optimizer = self._get_optimizer(request)
        # Unpack the parameters
        obj_gmm = al_util.deserialize_mixture(request.obj_gmm)
        ee_gmm = al_util.deserialize_mixture(request.ee_gmm)
        promps = al_util.deserialize_mixture(request.promps)
        phi = al_util.deserialize_array(request.phi)
        init_guesses = []
        for guess in request.init_guesses:
            init_guess = [guess.x, guess.y, guess.z, guess.theta]
            init_guesses.append(init_guess)

        # Run optimization
        instances = optimizer.get_max_entropy_instance(init_guesses, phi, promps, ee_gmm, obj_gmm)

        resp = GetOptimizationInstanceResponse()
        for instance in instances:
            selected = PlanarPose(x=instance[0], y=instance[1], z=instance[2], theta=instance[3])
            resp.selected_instances.append(selected)
        resp.success = True

        return resp

    def _get_least_confident_instance(self, request):
        rospy.loginfo("Generating Least-Confident instance...")
        # Setup the optimizer
        optimizer = self._get_optimizer(request)
        # Unpack the parameters
        obj_gmm = al_util.deserialize_mixture(request.obj_gmm)
        ee_gmm = al_util.deserialize_mixture(request.ee_gmm)
        promps = al_util.deserialize_mixture(request.promps)
        phi = al_util.deserialize_array(request.phi)
        init_guesses = []
        for guess in request.init_guesses:
            init_guess = [guess.x, guess.y, guess.z, guess.theta]
            init_guesses.append(init_guess)

        # Run optimization
        instances = optimizer.get_least_confident_instance(init_guesses, phi, promps, ee_gmm, obj_gmm)

        resp = GetOptimizationInstanceResponse()
        for instance in instances:
            selected = PlanarPose(x=instance[0], y=instance[1], z=instance[2], theta=instance[3])
            resp.selected_instances.append(selected)
        resp.success = True

        return resp

    def _get_mahalanobis_instance(self, request):
        rospy.loginfo("Generating Mahalanobis instance...")
        # Setup the optimizer
        optimizer = self._get_optimizer(request)
        # Unpack the parameters
        obj_gmm = al_util.deserialize_mixture(request.obj_gmm)
        ee_gmm = al_util.deserialize_mixture(request.ee_gmm)
        promps = al_util.deserialize_mixture(request.promps)
        phi = al_util.deserialize_array(request.phi)
        init_guesses = []
        for guess in request.init_guesses:
            init_guess = [guess.x, guess.y, guess.z, guess.theta]
            init_guesses.append(init_guess)

        # Run optimization
        instances = optimizer.get_mahalanobis_instance(init_guesses, phi, promps, ee_gmm, obj_gmm)

        resp = GetOptimizationInstanceResponse()
        for instance in instances:
            selected = PlanarPose(x=instance[0], y=instance[1], z=instance[2], theta=instance[3])
            resp.selected_instances.append(selected)
        resp.success = True

        return resp

    # def _get_min_margin_instance(self, request):
    #     # Setup the optimizer
    #     optimizer = self._get_optimizer(request)
    #     # Unpack the parameters
    #     obj_gmm = al_util.deserialize_mixture(request.obj_gmm)
    #     ee_gmm = al_util.deserialize_mixture(request.ee_gmm)
    #     promps = al_util.deserialize_mixture(request.promps)
    #     phi = al_util.deserialize_array(request.phi)
    #     # Run optimization
    #     instance = optimizer.get_min_margin_instance(phi, obj_gmm, promps,
    #                                                  ee_gmm)
    #     min_margin = optimizer.evaluate_min_margin(instance, phi, obj_gmm,
    #                                                promps, ee_gmm)

    #     resp = GetOptimizationInstanceResponse()
    #     resp.selected_instance.x = instance[0]
    #     resp.selected_instance.y = instance[1]
    #     resp.selected_instance.z = instance[2]
    #     resp.selected_instance.theta = instance[3]
    #     resp.min_margin = min_margin
    #     resp.success = True
    #     return resp

    def _get_optimizer(self, request):
        p = request.table_pose.position
        q = request.table_pose.orientation
        l = request.lower_bound
        u = request.upper_bound
        table_pos = [p.x, p.y, p.z]
        table_quat = [q.x, q.y, q.z, q.w]
        lbx = [l.x, l.y, l.z, l.theta]
        ubx = [u.x, u.y, u.z, u.theta]
        optimizer = Optimizer(table_pos, table_quat, lbx, ubx)
        return optimizer


if __name__ == '__main__':
    rospy.init_node('optimizer_node')
    node = OptimizerNode()
    node.run()
