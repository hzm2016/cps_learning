import unittest
import numpy as np
import matplotlib.pyplot as plt
import pyquaternion as pyquat
from copy import copy
from ll4ma_movement_primitives.dmps import DMP, DMPConfig


class TestDMP(unittest.TestCase):

    def test_trajectory_learning(self):
        tau = 2.0*np.pi
        dt = tau / 2000
        num_bfs = 10
        time = np.linspace(0, tau, 2000)
        t = np.linspace(0.0, tau, 1000)
        demo = np.hstack((np.sin(t) / (1.0 + np.exp(-t))**2, np.zeros(1000)))
        dmp_config = DMPConfig()
        # Mess with alpha and beta to get better goal convergence
        dmp_config.alpha = 100
        dmp_config.beta = 20
        dmp_config.dt = dt
        dmp_config.tau = tau
        dmp = DMP(dmp_config, [demo])
        
        ys, dys, ddys = dmp.get_rollout()
            
        plt.figure(figsize=(12,10))
        plt.plot(time, demo, lw=3.0, label='demo')
        plt.plot(time[:len(ys)], ys, lw=3.0, label='learned')
        plt.suptitle("DMP from Demo", fontsize=18, fontweight='bold')
        plt.xlim(0.0, tau)
        plt.legend()
        plt.show()
 

if __name__ == '__main__':
    unittest.main()
