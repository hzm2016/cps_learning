#!/usr/bin/env python
import rospy
import numpy as np
from tf import transformations as tf
from ll4ma_movement_primitives.promps import ProMP, ProMPConfig
from ll4ma_policy_learning.msg import Row, ProMPConfigROS
from ll4ma_policy_learning.srv import LearnProMP, LearnProMPResponse
from ll4ma_movement_primitives.promps import python_to_ros_config


class PolicyLearner:
    """
    Service provider for learning ProMPs.
    """
    
    def __init__(self):
        self.rate = rospy.Rate(100)
        self.promp_srv = rospy.Service("/policy_learner/learn_promps", LearnProMP, self.learn_promps)
        self.w_srv = rospy.Service("/policy_learner/learn_weights", LearnProMP, self.learn_weights)

    def run(self):
        rospy.loginfo("Policy learning services are available.")
        while not rospy.is_shutdown():
            self.rate.sleep()

    # === BEGIN Service functions ==================================================================

    def learn_promps(self, req):
        """
        Learns ProMP configurations given a collection of demos.
        """
        resp = LearnProMPResponse()
        data = self._get_data_trajs(req.joint_trajectories, req.ee_trajectories,
                                    req.object_trajectories)
        config, demos = self._get_config_from_data(data)
        if req.num_bfs > 0:
            config.num_bfs = req.num_bfs
        promp = ProMP(config=config)
        promp.learn_from_demos(demos)
        config = promp.get_config()
        if len(config.state_types) > 1:
            config.name = ""
            for i, state_type in enumerate(config.state_types):
                config.name += state_type
                if i < len(config.state_types) - 1:
                    config.name += "_"
        else:
            config.name = config.state_types[0]
        resp.config = python_to_ros_config(config)
        resp.success = True
        return resp

    def learn_weights(self, req):
        """
        Learns weights for basis functions given a collection of demos.
        """
        resp = LearnProMPResponse()
        data = self._get_data_trajs(req.joint_trajectories, req.ee_trajectories,
                                    req.object_trajectories)
        config, demos = self._get_config_from_data(data)
        config.num_bfs = req.num_bfs
        config.regr_alpha = req.regr_alpha
        promp = ProMP(config=config)
        resp.w = promp.learn_weights(demos[0]).flatten().tolist()
        resp.config = python_to_ros_config(config)
        resp.success = True
        return resp

    # === END Service functions ====================================================================

    def _get_data_trajs(self, j_trajs, ee_trajs, obj_trajs):
        """
        Reads out trajectory data from ROS messages and populates a dictionary.
        """
        num_trajectories = max(len(j_trajs), len(ee_trajs), len(obj_trajs))
        data = {}
        for i in range(num_trajectories):
            data_i = {}
            trajectory_name = None
            # Add joint trajectory data
            if j_trajs:
                qs = dqs = xs = obj_xs = None
                for j_point in j_trajs[i].points:
                    q = np.array(j_point.positions)[:, None]
                    qs = q if qs is None else np.hstack((qs, q))
                    dq = np.array(j_point.velocities)[:, None]
                    dqs = dq if dqs is None else np.hstack((dqs, dq))
                data_i['q'] = qs
                data_i['qdot'] = dqs
            # Add end-effector trajectory
            if ee_trajs:
                xs = None
                for ee_point in ee_trajs[i].points:
                    x = []
                    x.append(ee_point.pose.position.x)
                    x.append(ee_point.pose.position.y)
                    x.append(ee_point.pose.position.z)
                    x.append(ee_point.pose.orientation.x)
                    x.append(ee_point.pose.orientation.y)
                    x.append(ee_point.pose.orientation.z)
                    x.append(ee_point.pose.orientation.w)
                    x = np.array(x).reshape((7, 1))
                    xs = x if xs is None else np.hstack((xs, x))
                data_i['x'] = xs

                trajectory_name = ee_trajs[i].trajectory_name
            # Add object trajectory
            if obj_trajs:
                obj_xs = None
                for obj_point in obj_trajs[i].points:
                    obj_x = []
                    obj_x.append(obj_point.pose.position.x)
                    obj_x.append(obj_point.pose.position.y)
                    obj_x.append(obj_point.pose.position.z)
                    obj_x.append(obj_point.pose.orientation.x)
                    obj_x.append(obj_point.pose.orientation.y)
                    obj_x.append(obj_point.pose.orientation.z)
                    obj_x.append(obj_point.pose.orientation.w)
                    obj_x = np.array(obj_x).reshape((7, 1))
                    obj_xs = obj_x if obj_xs is None else np.hstack((obj_xs, obj_x))
                trajectory_name = obj_trajs[i].trajectory_name
            data[trajectory_name] = data_i
        return data

    def _get_config_from_data(self, data):
        """
        Initializes a ProMP configuration given the provided settings.
        """
        config = ProMPConfig()
        demos = []
        for demo_name in data.keys():
            ds = None
            config.state_types = []
            config.dimensions = []
            for state_type in data[demo_name].keys():
                dimensions = []
                for dim in range(data[demo_name][state_type].shape[0]):
                    w_key = "{}.{}".format(state_type, dim)
                    dimensions.append(dim)
                    d = data[demo_name][state_type][dim][:, None]
                    ds = d if ds is None else np.vstack((ds, d))
                    config.num_dims += 1
                    config.w_keys.append(w_key)
                config.state_types.append(state_type)
                config.dimensions.append(dimensions)
            demos.append(ds)
        return config, demos


if __name__ == '__main__':
    rospy.init_node("policy_learner")
    learner = PolicyLearner()
    learner.run()
