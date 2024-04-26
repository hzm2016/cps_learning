import rospy
import numpy as np
from sklearn.linear_model import Ridge


class TransformationSystem(object):
    """
    Implements the transformation system component of a DMP. See [1]
    for technical details.

    [1] Pastor, P., Hoffmann, H., Asfour, T., & Schaal, S. (2009, May). 
        Learning and generalization of motor skills by learning from 
        demonstration. In 2009 IEEE International Conference on Robotics 
        and Automation (pp. 763-768). IEEE.
    """
    
    def __init__(self, dmp_config, demos=[]):
        self.demos        = demos
        self.init         = dmp_config.init
        self.goal         = dmp_config.goal
        self.current_goal = dmp_config.goal
        self.tau          = dmp_config.tau
        self.alpha        = dmp_config.alpha
        self.beta         = dmp_config.beta
        self.regr_alpha   = dmp_config.regr_alpha
        self.w            = dmp_config.w
        self.reset()

    def learn(self, xs_list, bfs_list, regr=None):
        regr = Ridge(alpha=self.regr_alpha, fit_intercept=False) if regr is None else regr
        f = None
        f_target = None
        for i, ys in enumerate(self.demos):
            dt = 1.0 / len(ys)
            dys = self._diff(ys, dt)            
            # TODO getting accelerations through differentiation is no good on real data,
            # velocities are too noisy so you get huge accelerations which causes the
            # DMP to go unstable and oscillate.
            ddys = self._diff(dys, dt) 
            # ddys = np.zeros(dys.shape)
            init = ys[0]
            goal = ys[-1]
            f_current = xs_list[i] * bfs_list[i] / np.sum(bfs_list[i], axis=0)
            f_target_current = (((ddys + self.beta * dys) / self.alpha)
                                - (goal - ys)
                                + (goal - init) * xs_list[i])            
            f = f_current if f is None else np.hstack((f, f_current))
            f_target = f_target_current if f_target is None else np.hstack((f_target,
                                                                            f_target_current))
        regr.fit(f.T, f_target)
        self.w = regr.coef_
        return self.w
     
    def get_value(self, x, bfs, dt):
        # compute learned forcing function value
        f = x * np.dot(self.w, bfs) / np.sum(bfs, axis=0)

        # update the system
        self.ddy = (
            (self.alpha / self.tau**2) * ((self.goal - self.y) - (self.goal - self.init) * x + f)
            - (self.beta / self.tau) * self.dy
        )

        # integrate
        self.dy += self.ddy * dt
        self.y += self.dy * dt
        
        return self.y, self.dy, self.ddy

    def get_current_state(self):
        state = {}
        state['y'] = self.y
        state['dy'] = self.dy
        state['ddy'] = self.ddy
        state['goal'] = self.goal
        return state
    
    def get_rollout(self, xs, bfs, dt):
        self.reset()
        if self.w is None:
            self.learn(xs, bfs, dt)
        ys = []; dys = []; ddys = []
        for i in range(len(xs)):
            y, dy, ddy = self.get_value(xs[i], bfs[:,i], dt)
            ys.append(y)
            dys.append(dy)
            ddys.append(ddy)
        return np.array(ys), np.array(dys), np.array(ddys)

    def reset(self):
        self.y = self.init
        self.dy = 0.0
        self.ddy = 0.0
       
    def _diff(self, ys, dt):
        dys = np.diff(ys) / dt 
        return np.insert(dys, 0, 0.0)
