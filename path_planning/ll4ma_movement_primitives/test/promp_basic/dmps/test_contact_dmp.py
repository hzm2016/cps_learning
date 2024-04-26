import unittest
import numpy as np
import matplotlib.pyplot as plt
import pyquaternion as pyquat
from copy import copy
from policy_learning.dmps import DynamicMovementPrimitive
from policy_learning.util import quaternion


# TODO this file won't run yet, need to update for latest refactoring


class TestDMP(unittest.TestCase):

    def test_force_buildup(self):
        tau = 2*np.pi
        dt = tau / 2000
        time = np.linspace(0, tau, 2000)
        traj = np.hstack((np.sin(np.linspace(0.0, tau, 1000)), np.zeros(1000)))
        dmp = DynamicMovementPrimitive(traj, num_bfs=10, gamma=2.0, alpha=1000.0,
                                       alpha_p=1000.0, alpha_f=1000.0)
        dmp.tau = tau
        dmp.reset()

        ys = []
        wrench_mag = 0.0
        for t in time:
            if t > 0.2 * max(time) and t < 0.3 * max(time):
                wrench_mag += 0.01
                x = dmp.cs.get_value(dt, wrench_mag=wrench_mag, in_contact=1)
            else:
                x = dmp.cs.get_value(dt)
            bfs = dmp.bfs.get_value(x)
            y = dmp.get_value(x=x, dt=dt)
            ys.append(y)
            
        plt.figure(figsize=(12,10))
        plt.suptitle("Force Buildup", fontsize=18, fontweight='bold')
        plt.plot(time, traj, lw=3.0)
        plt.plot(time, ys, lw=3.0)
        plt.show()
 

if __name__ == '__main__':
    unittest.main()
