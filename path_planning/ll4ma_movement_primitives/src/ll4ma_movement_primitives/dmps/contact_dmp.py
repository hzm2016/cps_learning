import sys
import rospy
import numpy as np
from overrides import overrides
from matplotlib import pyplot as plt
from ll4ma_movement_primitives.dmp import DMP, ContactTFSystem
from ll4ma_movement_primitives.phase_variables import ContactExponentialPV
from ll4ma_movement_primitives.basis_functions import GaussianExponentialBFS


class ContactDMP(DMP):
    """
    Contact-aware extension of DMPs.

    See [1] for technical details.

    [1] Conkey, A., & Hermans, T. (2018). Learning Task Constraints 
        from Demonstration for Hybrid Force/Position Control. 
        arXiv preprint arXiv:1811.03026.
    """

    def __init__(self, dmp_config, demos=[], phase=None, name="dmp"):
        super(ContactDMP, self).__init__(dmp_config, demos, phase, name)        

        # read in values from config
        self.alpha_c  = dmp_config.alpha_c
        self.alpha_nc = dmp_config.alpha_nc
        self.alpha_f  = dmp_config.alpha_f

        self.phase = ContactExponentialPV(self.gamma, self.tau, self.alpha_f)
        self.ts = ContactTFSystem(dmp_config, self.demos)

        self.reset()

    @overrides
    def get_values(self, x=None, dt=None, pose_error=None, wrench_mag=None, in_contact=0):
        # usually x will be passed in if multi-dim DMP, since you won't want to poll CS multiple times
        if x is None:
            if dt is None:
                dt = self.dt
            x = self.phase.get_value(dt)
        bfs = self.bfs.get_value(x)
        vals = self.ts.get_value(x=x, bfs=bfs, dt=dt, pose_error=pose_error, wrench_mag=wrench_mag,
                                in_contact=in_contact)
        return vals

    @overrides
    def get_params(self):
        params = super(ContactDMP, self).get_params()
        params['alpha_c']  = self.alpha_c
        params['alpha_nc'] = self.alpha_nc
        params['alpha_p']  = self.alpha_p
        params['alpha_f']  = self.alpha_f
        return params

    @overrides
    def reset(self):
        super(ContactDMP, self).reset()
        self.cs.alpha_f = self.alpha_f
        self.cs.reset()
        self.ts.alpha_c = self.alpha_c
        self.ts.alpha_nc = self.alpha_nc
        self.ts.alpha_f = self.alpha_f
        self.ts.reset()

    @staticmethod
    def print_params(params, name=None, msg=None, num_spaces=4):
        if msg is None:
            if name is not None:
                msg = "\nDMP parameters for '%s':" % name
            else:
                msg = "\nDMP parameters:"
        np.set_printoptions(precision=3)
        indent = ' ' * num_spaces
        # TODO fix how you handle spacing here so it is variable
        print msg
        print "%sinit = %s"       % (indent, str(params['init']))
        print "%sgoal = %s"       % (indent, str(params['goal']))
        print "%snum_bfs = %.3f"  % (indent, params['num_bfs'])
        print "%snum_pts = %.3f"  % (indent, params['num_pts'])
        print "%sdt = %.5f"       % (indent, params['dt'])
        print "%stau = %.3f"      % (indent, params['tau'])
        print "%salpha = %.3f"    % (indent, params['alpha'])
        print "%sbeta = %.3f"     % (indent, params['beta'])
        print "%sgamma = %.3f"    % (indent, params['gamma'])
        print "%salpha_c = %.3f"  % (indent, params['alpha_c'])
        print "%salpha_nc = %.3f" % (indent, params['alpha_nc'])
        print "%salpha_p = %.3f"  % (indent, params['alpha_p'])
        print "%salpha_f = %.3f"  % (indent, params['alpha_f'])
        # TODO fix how weights are printed so it looks nicer
        print "%sweights = %s\n"  % (indent, str(params['w']))
