import numpy as np


class PhaseVariable(object):
    """
    Interface for phase variables.

    Implementations of phase variables should extend this class and
    implement the specified functions.
    """

    def __init__(self):
        self.x = None
        self.tau = 1.0

    def get_value(self, dt):
        """
        Increment the current phase value according to the provided
        timestep increment and return the current phase value.
        """
        raise NotImplementedError
        
    def get_current_state(self):
        """
        Returns the current state of the canonical system.

        By default, only the current phase variable is returned in
        a dictionary. This can be extended by adding additional
        values to the dictionary.
        """
        state = {}
        state['x'] = self.x
        return state

    def get_rollout(self, dt=None, num_pts=None, end_pad=0):
        """
        Returns a full rollout of the phase variable.
        """
        raise NotImplementedError
                
    def reset(self):
        """
        Resets the phase variable to its nominal starting state.
        """
        raise NotImplementedError
