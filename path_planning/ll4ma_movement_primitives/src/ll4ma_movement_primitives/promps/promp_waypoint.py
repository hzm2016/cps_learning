import numpy as np

class Waypoint:

    def __init__(self, t=0.0, x=None, values=[], condition_keys=[], sigma=None):
        """
        Waypoint for conditioning ProMP on desired state.

        t              - Timestep for conditioning (should only be provided if phase value is not)
        x              - Phase variable value for conditioning
        values         - State values to condition the ProMP on
        sigma          - Covariance associated with the observation noise on the waypoint
        condition_keys - State identifier keys associated with values (e.g. ['q.0', 'q.1'])
        """
        self.time           = t
        self.phase_val      = x
        self.values         = values
        self.condition_keys = condition_keys
        self.sigma          = sigma if sigma is not None else np.zeros((len(self.values),
                                                                        len(self.values)))

    def __str__(self):
        rep = (
            "\nTime           : {}\n"
            "Phase Value    : {}\n"
            "Values         : {}\n"
            "Condition Keys : {}\n"
            "Sigma          : {}\n".format(self.time, self.phase_val, self.values,
                                           self.condition_keys, self.sigma)
        )
        return rep
