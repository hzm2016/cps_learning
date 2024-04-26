import numpy as np
from overrides import overrides
from src.ll4ma_movement_primitives.phase_variables import PhaseVariable


class LinearPV(PhaseVariable):
    """
    Implements a linear phase variable.

    The phase value initializes to zero and linearly increases to 1.0. 
    The value is clipped at 1.0. See [1] for mathematical details.

    [1] Paraschos, A., Daniel, C., Peters, J., & Neumann, G. (2018). 
        Using probabilistic movement primitives in robotics. 
        Autonomous Robots, 42(3), 529-551.
    """

    def __init__(self, max_time=1.0):
        self.max_time = max_time
        self.reset()

    @overrides
    def get_value(self, dt):
        value = self.x
        self.x += (dt / self.max_time)
        self.x = min(self.x, 1.0) # saturate at 1
        return value

    @overrides
    def get_rollout(self, dt, end_pad=0):
        self.reset()
        xs = []
        time = np.linspace(0.0, self.max_time + (end_pad * dt),
                           int(round((self.max_time / dt) + end_pad)))
        for t in time:
            xs.append(self.get_value(dt))
        return np.array(xs)

    @overrides
    def reset(self):
        self.x = 0.0

    def get_max_val(self):
        return 1.0
