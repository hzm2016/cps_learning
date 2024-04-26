import sys
import roslib
import rospy
import actionlib
from sensor_msgs.msg import JointState
from ll4ma_policy_learning.msg import DMPConfiguration, DMPWeightVector

# TODO this is the pared down version used in RSS 2018 paper. It includes some peculiarities to the
# phase DMPs (making contact, in contact, etc.). It's untested in this form and needs to be cleaned
# up and incorporated better into this package.


class PhaseDMPActionClient:

    def __init__(self, rospy_init=True):
        if rospy_init:
            rospy.init_node("trajectory_action_client")
        self.robot_name = rospy.get_param("/robot_name", "lbr4")
        self.use_constraint_frame = rospy.get_param("/%s/use_constraint_frame"
                                                    % self.robot_name, False)
        self.ns = "/%s/execute_trajectory" % self.robot_name
        self.client = actionlib.SimpleActionClient(self.ns, OpSpaceTrajectoryAction)
        rospy.loginfo("[TrajectoryActionClient] Trying to connect with action server...")
        server_running = self.client.wait_for_server(timeout=rospy.Duration(10.0))
        if server_running:
            rospy.loginfo("[TrajectoryActionClient] Connected!")
        else:
            rospy.logwarn("[TrajectoryActionClient] You're probably connected to action server.")
            # TODO make a better check if this is important, for some reason when this is launched
            # with Gazebo it says it times out waiting, but does so immediately without actually
            # waiting, even though it is in fact connected. I think ROS is making more of a check
            # than just connection, so look into ROS source code for action server to check.

    def set_dmp_goal(self, trajectory, phase_params, phase_attrs, phase_types):
        rospy.loginfo("[TrajectoryActionClient] Setting DMP goal...")
        self.goal = OpSpaceTrajectoryGoal()
        self.goal.trajectory.dmp_configs = self._get_dmp_configs(phase_params, phase_attrs,
                                                                 phase_types)
        self.goal.trajectory.commanded_trajectory = trajectory
        rospy.loginfo("[TrajectoryActionClient] DMP goal set.")
        return True
        
    def send_goal(self, timeout=1000.0):
        rospy.loginfo("[TrajectoryActionClient] Sending goal to %s action server..." % self.ns)
        self.goal.trajectory.header.stamp = rospy.Time.now()
        self.client.send_goal(self.goal)
        rospy.loginfo("[TrajectoryActionClient] Goal sent successfully.")
        self.wait_for_result(timeout)

    def wait_for_result(self, timeout=1000.0):
        rospy.loginfo("[TrajectoryActionClient] Waiting for result...")
        success = self.client.wait_for_result(timeout=rospy.Duration(timeout))
        if success:
            rospy.loginfo("[TrajectoryActionClient] Result received: %s" % self.get_result())
        else:
            rospy.logwarn("[TrajectoryActionClient] Timed out waiting for result.")
        
    def _get_dmp_configs(self, phase_params, phase_attrs, phase_types):
        dmp_configs = []
        for phase_key in phase_params.keys():
            elements = phase_params[phase_key]
            for element_key in elements.keys():
                dims = elements[element_key]
                for dim_key in dims.keys():
                    params = dims[dim_key]
                    dmp_config = DMPConfiguration()
                    dmp_config.phase_name = str(phase_key)
                    dmp_config.phase_type = str(phase_types[phase_key])
                    dmp_config.element_name = str(element_key)
                    dmp_config.dimension = str(dim_key)
                    if dim_key == 'rot':
                        for i in range(3):
                            dmp_config.weight_vectors.append(
                                DMPWeightVector(params['w'][i,:].tolist()))
                    else:
                        dmp_config.weight_vectors = [DMPWeightVector(params['w'][:].tolist())]
                    dmp_config.tau = params['tau']
                    dmp_config.alpha = params['alpha']
                    dmp_config.beta = params['beta']
                    dmp_config.gamma = params['gamma']
                    dmp_config.alpha_c = params['alpha_c']
                    dmp_config.alpha_nc = params['alpha_nc']
                    dmp_config.alpha_p = params['alpha_p']
                    dmp_config.alpha_f = params['alpha_f']
                    # need to account for quaternion init/goal for CDMP and floats for normal DMP
                    if dim_key == 'rot':
                        dmp_config.init = params['init']
                        dmp_config.goal = params['goal']
                    else:
                        dmp_config.init = [params['init']]
                        dmp_config.goal = [params['goal']]
                    dmp_config.num_bfs = int(params['num_bfs'])
                    dmp_config.dt = params['dt']
                    # TODO this is super hardcoded to get force goals for making_contact phase
                    if (phase_key == 'phase_1' 
                        and 'force_goal' in phase_attrs['phase_1'].keys()
                        and dim_key in ['0', '1', '2']):
                        dmp_config.force_goal = phase_attrs['phase_1']['force_goal'][int(dim_key)]
                        dmp_config.make_contact_cf = phase_attrs['phase_1']['make_contact_cf']
                        dmp_config.make_contact_mag = phase_attrs['phase_1']['make_contact_mag']
                    dmp_configs.append(dmp_config)
        return dmp_configs
