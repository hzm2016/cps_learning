import numpy as np


class BasisFunctionSystem(object):
    """
    Interface class for basis function systems.
    """
    
    def __init__(self, num_bfs=10):
        self.num_bfs = num_bfs
        
    def get_value(self, x):
        """
        Returns the basis function system value given the value of the 
        phase variable.
        """
        raise NotImplementedError

    def get_rollout(self, xs):
        """
        Returns a full rollout of the basis function values given a sequence
        of phase variable values.
        """
        bfs = []
        for x in xs:
            bfs.append(self.get_value(x))
        return np.array(bfs).T
    
    def get_rollout_list(self, xs_list):
        """
        Returns basis function system rollouts in batch.
        """
        bfs_list = []
        for xs in xs_list:
            bfs = self.get_rollout(xs)
            bfs_list.append(bfs)
        return bfs_list

    def reset(self):
        """
        Resets the basis function system to its nominal starting state.
        """
        raise NotImplementedError
