import unittest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.ll4ma_movement_primitives.promps import ProMP, ProMPConfig, ProMPLibrary, Waypoint
from src.ll4ma_movement_primitives.phase_variables import LinearPV  
from src.ll4ma_movement_primitives.basis_functions import GaussianLinearBFS  


class TestProMP(unittest.TestCase):

    def test_single_dof_conditioning(self):
        sns.set_palette("deep")

        num_demos = 500
        dt = 0.01
        duration = 1.0
        duration += dt # to get time points to line up correctly
        t = np.arange(0.0, duration, dt)
        noise = 0.03
        A = np.array([.2, .2, .01, -.05])
        X = np.vstack( (np.sin(5*t), t**2, t, np.ones((1,len(t))) ))
        
        Y = np.zeros((num_demos, len(t)))
        demos = []
        for traj in range(0, num_demos):
            sample = np.dot(A + noise * np.random.randn(1,4), X)[0]
            demos.append(sample)
            
        ds = None
        for d in demos:
            ds = d if ds is None else np.vstack((ds, d))
        upper = np.max(ds, axis=0)
        lower = np.min(ds, axis=0)
        plt.figure(figsize=(14,12))
        plt.xlim(0.0, 1.0)
        plt.subplot(221)
        plt.fill_between(t, lower, upper, facecolor=sns.xkcd_rgb['goldenrod'], alpha=0.4, lw=0.0)
        
        config = ProMPConfig()
        config.num_bfs = 10
        config.alpha = 0.001
        config.state_types = ['q']
        config.dimensions = [[1]]
        config.w_keys = ["q.1"]
        
        p = ProMP(config=config)
        p.learn_from_demos(demos)

        # No waypoints
        gens = []
        for i in range(500):
            gens.append(p.generate_trajectory(dt, duration))
        gs = None
        for g in gens:
            q = g['q'][1]
            gs = q if gs is None else np.vstack((gs, q))
        g_upper = np.max(gs, axis=0)
        g_lower = np.min(gs, axis=0)
        plt.title("No Conditioning", fontweight='bold', fontsize=22)
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.fill_between(t, g_lower, g_upper, facecolor=sns.xkcd_rgb['red'], alpha=0.4, lw=0.0)
        
        # Start at zero
        waypoints = [Waypoint(x=0.0, values=[0.0], condition_keys=["q.1"],
                              sigma=np.eye(1)*0.0001)]
        
        gens = []
        for i in range(500):
            gens.append(p.generate_trajectory(dt, duration, waypoints=waypoints))
        gs = None
        for g in gens:
            q = g['q'][1]
            gs = q if gs is None else np.vstack((gs, q))
        g_upper = np.max(gs, axis=0)
        g_lower = np.min(gs, axis=0)
        plt.subplot(222)
        plt.title("Condition Start=(0.0, 0.0)", fontweight='bold', fontsize=22)
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.fill_between(t, lower, upper, facecolor=sns.xkcd_rgb['goldenrod'], alpha=0.4, lw=0.0)
        plt.fill_between(t, g_lower, g_upper, facecolor=sns.xkcd_rgb['red'], alpha=0.4, lw=0.0)
        
        # End at non-zero
        waypoints.append(Waypoint(x=1.0, values=[-0.2], condition_keys=["q.1"]))
        gens = []
        for i in range(500):
            gens.append(p.generate_trajectory(dt, duration, waypoints=waypoints))
        gs = None
        for g in gens:
            q = g['q'][1]
            gs = q if gs is None else np.vstack((gs, q))
        g_upper = np.max(gs, axis=0)
        g_lower = np.min(gs, axis=0)
        plt.subplot(223)
        plt.title("Condition End=(1.0, -0.2)", fontweight='bold', fontsize=22)
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.fill_between(t, lower, upper, facecolor=sns.xkcd_rgb['goldenrod'], alpha=0.4, lw=0.0)
        plt.fill_between(t, g_lower, g_upper, facecolor=sns.xkcd_rgb['red'], alpha=0.4, lw=0.0)
        
        # Mid waypoint
        waypoints.append(Waypoint(x=0.5, values=[0.1], condition_keys=["q.1"]))
        gens = []
        for i in range(500):
            gens.append(p.generate_trajectory(dt, duration, waypoints=waypoints))
        gs = None
        for g in gens:
            q = g['q'][1]
            gs = q if gs is None else np.vstack((gs, q))
        g_upper = np.max(gs, axis=0)
        g_lower = np.min(gs, axis=0)
        plt.subplot(224)
        plt.title("Condition Waypoint=(0.5, 0.1)", fontweight='bold', fontsize=22)
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.fill_between(t, lower, upper, facecolor=sns.xkcd_rgb['goldenrod'], alpha=0.4, lw=0.0)
        plt.fill_between(t, g_lower, g_upper, facecolor=sns.xkcd_rgb['red'], alpha=0.4, lw=0.0)
        
        plt.subplots_adjust(left=0.07, bottom=0.05, right=0.93, top=0.95, 
                            wspace=0.2, hspace=0.2)

        fig = plt.gcf()
        fig.canvas.set_window_title("Conditioning")
        plt.savefig("conditioning.png", format="png")
        plt.show()
        
        
if __name__ == '__main__':
    unittest.main()
