"""
DMP transformation system for quaternions with contact feedback.

RSS 2018 submission (under review).

"""
import numpy as np
from sklearn import linear_model.Ridge
from overrides import overrides
from ll4ma_movement_primitives.dmps import QuaternionTFSystem
from ll4ma_movement_primitives.util import quaternion


class ContactQuaternionTFSystem(QuaternionTFSystem):
    """
    Contact-aware extension of the Quaternion transformation system for DMPs.

    See [1] for technical details.

    [1] Conkey, A., & Hermans, T. (2018). Learning Task Constraints 
        from Demonstration for Hybrid Force/Position Control. 
        arXiv preprint arXiv:1811.03026.
    """
    
    def __init__(self, dmp_config, demos=[]):
        super(ContactQuaternionTFSystem, self).__init__(dmp_config, demos)
        self.alpha_c  = dmp_config.alpha_c
        self.alpha_nc = dmp_config.alpha_nc
        self.alpha_p  = dmp_config.alpha_p
        self.alpha_f  = dmp_config.alpha_f
        self.current_goal = self.goal

    @overrides
    def get_value(self, x, bfs, dt, pose_error=None, wrench_mag=None, in_contact=0):

        # TODO change in goal here is not quite working, need to investigate a little more. For now
        # just ignore orientation goal in the goal convergence determination

        # modulate goal if necessary
        # if in_contact:
        #     g_diff = quaternion.err(self.q, self.current_goal)
        #     g_diff = np.insert(g_diff, 3, 0.0)
        #     g_diff = quaternion.prod(g_diff, self.current_goal)
        #     g_diff *= self.alpha_c
        #     self.current_goal = quaternion.prod(g_diff, self.current_goal)
        # else:
        #     pass
            
        feedback = 0.0
        if pose_error is not None:
            feedback += self.alpha_p * pose_error
        if wrench_mag is not None:
            feedback += self.alpha_f * wrench_mag

        self.fb_scaling = 1.0 / (1.0 + float(in_contact) * feedback)

        # compute learned forcing function value
        f = x * np.dot(self.w, bfs) / np.sum(bfs, axis=0)
        f = f[:,None] # 3 x 1
        
        # update the system
        self.d_omega = (
            (self.alpha / self.tau**2) * (
                quaternion.err(self.current_goal, self.q) * self.fb_scaling # TODO is this right?
                - (quaternion.err(self.current_goal, self.init) * x) + f
            )
            - (self.beta / self.tau) * self.omega
        )

        # integrate
        self.omega += self.d_omega * dt    
        self.omega *= self.fb_scaling
        self.q = quaternion.integ(self.q, self.omega, dt)
        
        return self.q, self.omega, self.d_omega

    @overrides
    def get_current_state(self):
        # TODO making these consistent with normal transformation system, but may want to rename
        # both to something that makes sense generally
        state = {}
        state['y'] = self.q
        state['dy'] = self.omega
        state['ddy'] = self.d_omega
        state['goal'] = self.current_goal
        state['fb'] = self.fb_scaling
        return state
    
