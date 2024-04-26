import numpy as np
from overrides import overrides
from matplotlib import pyplot as plt
from ll4ma_movement_primitives.dmps import DMP, QuaternionTFSystem

# TODO this class is currently broken, need to update based on refactors of DMP but
# it was proving more difficult than expected. Need to step through and maybe restructure
# things a bit to make this inheritance worthwhile.


class QuaternionDMP(DMP):

    def __init__(self, dmp_config, demos=[], phase=None):
        super(QuaternionDMP, self).__init__(dmp_config, demos, phase)
        self.ts = QuaternionTFSystem(dmp_config, demos)
        self.learn()
        # TODO probably need to reset

    @overrides
    def learn(self):
        xs_list = self.phase.get_rollout_list(self.demos, duration=self.demos[0]['q'].shape[1])
        bfs_list = self.bfs.get_rollout_list(xs_list)            
        self.w = self.ts.learn(xs_list, bfs_list)
        self.ts.reset()
        
    @overrides
    def reset(self):
        # Set default init to first trajectory start
        if self.init is None:
            self.init = self.demos[0]['q'][:,0]
        # Set default goal to first trajectory end
        if self.goal is None:
            self.goal = self.demos[0]['q'][:,-1]
        if self.num_pts is None:
            # Only need num_pts if demo provided, otherwise it's online execution
            if self.demos:
                self.num_pts = self.demos[0]['q'].shape[1]

        if self.beta is None:
            self.beta = 2.0 * np.sqrt(self.alpha)
        if self.dt is None:
            self.dt = self.tau / self.num_pts

        # Phase variable
        self.phase.tau     = self.tau
        self.phase.gamma   = self.gamma
        self.phase.reset()

        # Transformation system
        self.ts.tau          = self.tau
        self.ts.alpha        = self.alpha
        self.ts.beta         = self.beta
        self.ts.init         = self.init
        self.ts.goal         = self.goal
        self.ts.current_goal = self.goal
        self.ts.w            = self.w
        self.ts.regr_alpha   = self.regr_alpha
        self.ts.reset()

        # Basis function system
        self.bfs.tau = self.tau
        self.bfs.gamma = self.gamma
        self.bfs.reset()
        
        if self.w is None:
            self.learn()

