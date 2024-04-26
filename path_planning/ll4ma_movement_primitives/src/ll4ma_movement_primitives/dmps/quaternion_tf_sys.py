import numpy as np
from sklearn.linear_model import Ridge
from overrides import overrides
from ll4ma_movement_primitives.dmps import TransformationSystem
from ll4ma_movement_primitives.util import quaternion


class QuaternionTFSystem(TransformationSystem):
    """
    Implements a DMP transformation system for orientation in Cartesian space.
    
    Orientation must be handled differently since there is no singularity-free 
    representation of orientation in 3 parameters, and the dimensions in a 
    representation using more than 3 are not independent of each other.
    
    This implementation uses quaternions and is based on [1].
    
    [1] Ude, A., Nemec, B., Petric, T., & Morimoto, J. (2014, May). 
        Orientation in cartesian space dynamic movement primitives. 
        In 2014 IEEE International Conference on Robotics and 
        Automation (ICRA) (pp. 2997-3004). IEEE.
    """
    
    def __init__(self, dmp_config, demos=[]):
        super(QuaternionTFSystem, self).__init__(dmp_config, demos)

    @overrides
    def learn(self, xs_list, bfs_list, regr=None):
        regr = Ridge(alpha=self.regr_alpha, fit_intercept=False) if regr is None else regr
        num_bfs = bfs_list[0].shape[0]
        f = np.array([], dtype=np.int64).reshape(num_bfs, 0)
        f_t = np.array([], dtype=np.int64).reshape(3, 0)
        for i, data in enumerate(self.demos):
            # assuming demos recorded orientation, angular velocity, and angular acceleration
            qs = data['q']
            omegas = data['omega']
            d_omegas = data['d_omega']
            dt = 1.0 / qs.shape[1]
            init = qs[:,0]
            goal = qs[:,-1][:,None]
            f_curr = xs_list[i] * bfs_list[i] / np.sum(bfs_list[i], axis=0)
            f_t_curr = (((d_omegas + self.beta * omegas) / self.alpha)
                        - quaternion.err(goal, qs)
                        + quaternion.err(goal, init) * xs_list[i])
            f = np.hstack((f, f_curr))
            f_t = np.hstack((f_t, f_t_curr))
        regr.fit(f.T, f_t.T)
        self.w = regr.coef_
        return self.w

    @overrides
    def get_value(self, x, bfs, dt, pose_error=None, wrench_mag=None, in_contact=0):
        # compute learned forcing function value
        f = x * np.dot(self.w, bfs) / np.sum(bfs, axis=0)
        f = f[:,None] # 3 x 1
        
        # update the system
        print "Q", self.y
        print "GOAL", self.current_goal
        
        self.d_omega = (
            (self.alpha / self.tau**2) * (
                quaternion.err(self.current_goal, self.q) * self.fb_scaling
                - (quaternion.err(self.current_goal, self.init) * x)
                + f
            )
            - (self.beta / self.tau) * self.omega
        )

        # integrate
        self.omega += self.d_omega * dt    
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
        state['goal'] = self.goal
        return state

    @overrides
    def get_rollout(self, xs, bfs, dt):
        self.reset()
        if self.w is None:
            self.learn(xs, bfs, dt)
        qs = np.array([], dtype=np.int64).reshape(4,0)
        omegas = np.array([], dtype=np.int64).reshape(3,0)
        d_omegas = np.array([], dtype=np.int64).reshape(3,0)
        for i in range(len(xs)):
            q, omega, d_omega = self.get_value(xs[i], bfs[:,i], dt)
            qs = np.hstack((qs, q))
            omegas = np.hstack((omegas, omega))
            d_omegas = np.hstack((d_omegas, d_omega))
        return qs, omegas, d_omegas

    @overrides
    def reset(self):
        self.q = self.init
        self.omega = np.zeros(3)[:,None]
        self.d_omega = np.zeros(3)[:,None]
    
