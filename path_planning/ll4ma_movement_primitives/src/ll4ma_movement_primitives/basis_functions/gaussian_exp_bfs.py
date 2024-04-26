import numpy as np
from overrides import overrides
from ll4ma_movement_primitives.basis_functions import BasisFunctionSystem


class GaussianExponentialBFS(BasisFunctionSystem):
    """
    Implements squared exponential basis functions spaced uniformly in
    exponential space. 

    Intended for use with an exponential phase variable. See [1] and [2] 
    for technical details.

    [1] Ijspeert, A. J., Nakanishi, J., Hoffmann, H., Pastor, P., 
        & Schaal, S. (2013). Dynamical movement primitives: 
        learning attractor models for motor behaviors. Neural 
        computation, 25(2), 328-373.
    [2] Ude, A., Nemec, B., Petric, T., & Morimoto, J. (2014, May). 
        Orientation in cartesian space dynamic movement primitives. 
        In 2014 IEEE International Conference on Robotics and 
        Automation (ICRA) (pp. 2997-3004). IEEE.
    """
    
    def __init__(self, num_bfs=10, gamma=2.0, tau=1.0):
        super(GaussianExponentialBFS, self).__init__(num_bfs)
        self.gamma = gamma
        self.tau = tau
        self.reset()

    @overrides
    def get_value(self, x):
        return np.exp(-self.widths * (x - self.centers)**2)

    @overrides
    def reset(self):
        # based on Ude et al. 2014 "Orientation in Cartesian Space Dynamic Movement Primitives"
        self.centers = np.exp(-self.gamma * np.arange(self.num_bfs) / (float(self.num_bfs - 1)))
        hs = []
        for i in range(self.num_bfs - 1):
            hs.append(1.0 / (self.centers[i+1] - self.centers[i])**2)
        hs.append(hs[-1])
        self.widths = np.array(hs)
