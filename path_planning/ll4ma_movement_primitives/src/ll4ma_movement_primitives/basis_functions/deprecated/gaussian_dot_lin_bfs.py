import numpy as np
from overrides import overrides
from ll4ma_movement_primitives.basis_functions import BasisFunctionSystem


class GaussianDotLinearBFS(BasisFunctionSystem):

    def __init__(self, num_bfs=10, max_time=1.0):
        super(GaussianDotLinearBFS, self).__init__(num_bfs)
        self.max_time = max_time
        self.reset()

    @overrides
    def get_value(self, x):
        bfs = np.exp(-self.widths * (x * self.max_time - self.centers)**2)
        sum_bfs = np.sum(bfs, axis=0)
        d_bfs = -0.25 * self.widths * self.max_time * (self.max_time * x - self.centers) * bfs
        return d_bfs

    @overrides
    def reset(self):
        self.centers = np.linspace(0.0, self.max_time, self.num_bfs)
        hs = []
        for i in range(self.num_bfs - 1):
            hs.append(1.0 / (self.centers[i+1] - self.centers[i])**2)
        hs.append(hs[-1])
        self.widths = np.array(hs)
