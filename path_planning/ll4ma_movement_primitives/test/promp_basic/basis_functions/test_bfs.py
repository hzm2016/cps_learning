import matplotlib
matplotlib.use('Agg')  

import unittest 
import numpy as np 
import matplotlib.pyplot as plt 
from src.ll4ma_movement_primitives.phase_variables import LinearPV, ExponentialPV  
from src.ll4ma_movement_primitives.basis_functions import GaussianLinearBFS, GaussianExponentialBFS


class TestBFS(unittest.TestCase):  

    def test_gaussian_lin_bfs(self):
        num_pts = 1000
        max_time = 1.0
        dt = max_time / num_pts
        time = np.linspace(0.0, max_time, num_pts)  
        phase = LinearPV(max_time) 
        xs = phase.get_rollout(dt)   
        bfs = GaussianLinearBFS(max_time=max_time)  
        vals = bfs.get_rollout(xs)
        vals = (vals.T / np.sum(vals, axis=1)).T
        plt.figure(figsize=(12,10))
        plt.suptitle("Gaussian Basis Functions - Linear Spacing", fontsize=18, fontweight='bold')
        for j in range(vals.shape[0]):
            plt.plot(time, vals[j,:], lw=3.0)
        # plt.show()
        plt.savefig("gaussian_lin.svg")
        
    def test_gaussian_exp_bfs(self):
        num_pts = 1000
        tau = 1.0
        dt = tau / num_pts
        time = np.linspace(0.0, tau, num_pts)
        phase = ExponentialPV(tau=tau)
        xs = phase.get_rollout(dt)
        bfs = GaussianExponentialBFS(tau=tau)
        vals = bfs.get_rollout(xs)
        plt.figure(figsize=(12,10))
        plt.suptitle("Gaussian Basis Functions - Exponential Spacing", fontsize=18, fontweight='bold')
        for j in range(vals.shape[0]):
            plt.plot(time, vals[j,:], lw=5.0)
        # plt.show()
        plt.savefig("gaussian_exp.svg")
        
if __name__ == '__main__':
    unittest.main()
