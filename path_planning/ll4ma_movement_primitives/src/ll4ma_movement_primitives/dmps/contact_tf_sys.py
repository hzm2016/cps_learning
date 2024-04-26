import rospy
import numpy as np
from sklearn import linear_model
from ll4ma_movement_primitives.dmps import TransformationSystem


class ContactTFSystem(TransformationSystem):
    """
    Contact-aware DMP transformation system. 

    Allows for shifting the DMP goal and halting based on feedback
    of contact interaction forces. See [1] for technical details.

    [1] Conkey, A., & Hermans, T. (2018). Learning Task Constraints 
        from Demonstration for Hybrid Force/Position Control. 
        arXiv preprint arXiv:1811.03026.
    """

    def __init__(self, dmp_config, demos=[]):
        super(ContactTFSystem, self).__init__(dmp_config, demos)
        self.alpha_c      = dmp_config.alpha_c
        self.alpha_nc     = dmp_config.alpha_nc
        self.alpha_p      = dmp_config.alpha_p
        self.alpha_f      = dmp_config.alpha_f
        self.current_goal = self.goal

    @overrides
    def get_value(self, x, bfs, dt, pose_error=None, wrench_mag=None, in_contact=0):
        # modulate the goal if necessary
        if in_contact:
            self.current_goal += self.alpha_c * (self.y - self.current_goal)
        else:
            self.current_goal += self.alpha_nc * (self.goal - self.current_goal)

        feedback = 0.0
        if pose_error is not None:
            feedback += self.alpha_p * pose_error
        if wrench_mag is not None:
            feedback += self.alpha_f * wrench_mag

        self.fb_scaling = 1.0 / (1.0 + float(in_contact) * feedback)
            
        # compute learned forcing function value
        f = x * np.dot(self.w, bfs) / np.sum(bfs, axis=0)

        # update the system
        self.ddy = (
            (self.alpha / self.tau**2) * (
                (self.current_goal - self.y) * self.fb_scaling # TODO is this right
                - (self.current_goal - self.init) * x + f
            )
            - (self.beta / self.tau) * self.dy
        )

        # integrate
        self.dy += self.ddy * dt
        self.dy *= self.fb_scaling
        self.y += self.dy * dt
        
        return self.y, self.dy, self.ddy

    @overrides
    def get_current_state(self):
        state = {}
        state['y'] = self.y
        state['dy'] = self.dy
        state['ddy'] = self.ddy
        state['goal'] = self.current_goal
        state['fb'] = self.fb_scaling
        return state

