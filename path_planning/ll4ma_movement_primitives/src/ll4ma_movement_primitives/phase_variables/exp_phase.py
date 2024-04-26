import numpy as np
import matplotlib.pyplot as plt
from overrides import overrides
from src.ll4ma_movement_primitives.phase_variables import PhaseVariable


class ExponentialPV(PhaseVariable):
    """
    Implements an exponential phase variable.

    Phase value initializes to 1.0 and gradually converges to zero 
    as time goes to infinity. See [1] and [2] for mathematical details.

    [1] Ijspeert, A. J., Nakanishi, J., Hoffmann, H., Pastor, P., 
        & Schaal, S. (2013). Dynamical movement primitives: 
        learning attractor models for motor behaviors. Neural 
        computation, 25(2), 328-373.
    [2] Ude, A., Nemec, B., Petric, T., & Morimoto, J. (2014, May). 
        Orientation in cartesian space dynamic movement primitives. 
        In 2014 IEEE International Conference on Robotics and 
        Automation (ICRA) (pp. 2997-3004). IEEE.
    """
    
    def __init__(self, gamma=2.0, tau=1.0):
        self.gamma = gamma
        self.tau = tau
        self.reset()

    @overrides
    def get_value(self, dt):
        value = self.x
        self.x += self.x * dt * (-self.gamma / self.tau)
        return value

    @overrides
    def get_rollout(self, dt=None, num_pts=None, end_pad=0):
        xs = []
        if dt is None and num_pts is None:
            print("Must provide either a timestep or number of rollout points.")
            return None  

        if dt is None:
            dt = self.tau / num_pts
        if num_pts is None:
            num_pts = self.tau / dt
        ts = np.linspace(0.0, self.tau, num_pts + end_pad)
        for t in ts:
            xs.append(self.get_value(dt))
        return np.array(xs)

    def get_rollout_list(self, trajs, duration=None):
        xs_list = []
        for traj in trajs:
            xs = []
            if duration is None:
                duration = max(traj.shape)
            dt = self.tau / duration
            for t in np.linspace(0.0, self.tau, duration):
                xs.append(self.get_value(dt))
            xs_list.append(np.array(xs))
        return xs_list

    @overrides
    def reset(self):
        self.x = 1.0
