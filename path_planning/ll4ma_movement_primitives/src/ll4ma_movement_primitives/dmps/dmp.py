import sys
import rospy
import numpy as np
from matplotlib import pyplot as plt
from ll4ma_movement_primitives.phase_variables import ExponentialPV
from ll4ma_movement_primitives.basis_functions import GaussianExponentialBFS
from ll4ma_movement_primitives.dmps import TransformationSystem


class DMP(object):
    """
    Implements a Dynamic Movement primitive.

    This class unifies the various components of a DMP, including
    the phase variable, basis function system, and transformation
    system. This serves as the main interface for using the DMP
    policy.
    """

    def __init__(self, dmp_config, demos=[], phase=None):
        if isinstance(demos, np.ndarray) or isinstance(demos, dict):
            demos = [demos] # DMP expects list of demonstrations
        self.demos = demos

        # read in values from config
        self.init       = dmp_config.init
        self.goal       = dmp_config.goal
        self.num_bfs    = dmp_config.num_bfs
        self.dt         = dmp_config.dt
        self.w          = dmp_config.w
        self.alpha      = dmp_config.alpha
        self.beta       = dmp_config.beta
        self.tau        = dmp_config.tau
        self.gamma      = dmp_config.gamma
        self.regr_alpha = dmp_config.regr_alpha
        self.name       = dmp_config.name
        self.state_type = dmp_config.state_type
        self.dimension  = dmp_config.dimension

        self.phase = phase
        if self.phase is None:
            self.phase = ExponentialPV(gamma=self.gamma, tau=self.tau)
        self.ts = TransformationSystem(dmp_config, self.demos)
        self.bfs = GaussianExponentialBFS(num_bfs=self.num_bfs)
        self.num_pts = None

        self.reset()

    def learn(self):
        xs_list = self.phase.get_rollout_list(self.demos)
        bfs_list = self.bfs.get_rollout_list(xs_list)            
        self.w = self.ts.learn(xs_list, bfs_list)
        self.ts.reset()

    def get_values(self, x=None, dt=None):
        # usually x will be passed in if multi-dim DMP, since you won't want to poll CS multiple times
        if x is None:
            if dt is None:
                dt = self.dt
            x = self.cs.get_value(dt)
        bfs = self.bfs.get_value(x)
        vals = self.ts.get_value(x=x, bfs=bfs, dt=dt)
        return vals

    def get_current_state(self):
        ts_state = self.ts.get_current_state()
        phase_state = self.phase.get_current_state()
        state = {}
        for key in ts_state.keys():
            state[key] = ts_state[key]
        for key in phase_state.keys():
            state[key] = phase_state[key]
        return state
            
    def get_rollout(self):
        xs = self.get_phase_rollout()
        bfs = self.get_bfs_rollout()
        ys, dys, ddys = self.ts.get_rollout(xs, bfs, self.dt)
        return ys, dys, ddys

    def get_bfs_rollout(self):
        xs = self.get_phase_rollout()
        bfs = self.bfs.get_rollout(xs)
        return bfs

    def get_phase_rollout(self):
        self.reset()
        xs = self.phase.get_rollout(self.dt, self.num_pts)
        return xs

    def get_params(self):
        params = {}
        params['tau']     = self.tau
        params['alpha']   = self.alpha
        params['beta']    = self.beta
        params['gamma']   = self.gamma
        params['init']    = self.init
        params['goal']    = self.goal
        params['w']       = self.w
        params['num_bfs'] = self.num_bfs
        params['dt']      = self.dt
        return params

    def get_current_goal(self):
        return self.ts.current_goal

    def set_current_goal(self, goal):
        self.ts.current_goal = goal

    def set_init(self, init):
        self.init = init
        self.ts.init = init

    def reset(self):
        # set default values if not provided
        if self.init is None:
            self.init = self.demos[0][0]
        if self.goal is None:
            # set default goal to first trajectory end
            self.goal = self.demos[0][-1]
        if self.beta is None:
            self.beta = 2.0 * np.sqrt(self.alpha)
        # if self.num_pts is None:
        #     # only need num_pts if demo provided, otherwise it's online execution
        #     if self.demos:
        #         self.num_pts = len(self.demos[0])
        if self.dt is None:
            self.dt = self.tau / self.num_pts

        # phase variable
        self.phase.tau     = self.tau
        self.phase.gamma   = self.gamma
        self.phase.reset()

        # transformation system
        self.ts.tau          = self.tau
        self.ts.alpha        = self.alpha
        self.ts.beta         = self.beta
        self.ts.init         = self.init
        self.ts.goal         = self.goal
        self.ts.current_goal = self.goal
        self.ts.w            = self.w
        self.ts.regr_alpha   = self.regr_alpha
        self.ts.reset()

        # basis function system
        self.bfs.tau = self.tau
        self.bfs.gamma = self.gamma
        self.bfs.reset()
        
        if self.w is None:
            self.learn()

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
        print "%sdt = %.5f"       % (indent, params['dt'])
        print "%stau = %.3f"      % (indent, params['tau'])
        print "%salpha = %.3f"    % (indent, params['alpha'])
        print "%sbeta = %.3f"     % (indent, params['beta'])
        print "%sgamma = %.3f"    % (indent, params['gamma'])
        print "%sweights = %s\n"  % (indent, str(params['w']))
