import numpy as np
from overrides import overrides
from ll4ma_movement_primitives.basis_functions import BasisFunctionSystem


class GaussianLinearBFS(BasisFunctionSystem):
    """
    Implements squared exponential bases functions spaced uniformly.

    Intended for use with a linear phase variable. See [1] and [2]
    for technical details.

    [1] Ijspeert, A. J., Nakanishi, J., Hoffmann, H., Pastor, P., 
        & Schaal, S. (2013). Dynamical movement primitives: 
        learning attractor models for motor behaviors. Neural 
        computation, 25(2), 328-373.
    [2] Paraschos, A., Daniel, C., Peters, J., & Neumann, G. (2018). 
        Using probabilistic movement primitives in robotics. 
        Autonomous Robots, 42(3), 529-551.
    """
    
    def __init__(self, num_bfs=10, max_time=1.0):
        super(GaussianLinearBFS, self).__init__(num_bfs)
        self.max_time = max_time
        self.reset()

    @overrides
    def get_value(self, x):
        bfs = np.exp(-self.widths * (x * self.max_time - self.centers)**2)
        return bfs / np.sum(bfs, axis=0)

    @overrides
    def reset(self):
        self.centers = np.linspace(0.0, self.max_time, self.num_bfs)
        hs = []
        for i in range(self.num_bfs - 1):
            hs.append(1.0 / (self.centers[i+1] - self.centers[i])**2)
        hs.append(hs[-1])
        self.widths = np.array(hs)
