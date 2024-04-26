import matplotlib
matplotlib.use('Agg')
import unittest
import numpy as np
import matplotlib.pyplot as plt
from ll4ma_movement_primitives.phase_variables import LinearPV, ExponentialPV


class TestPhase(unittest.TestCase):

    def test_exp_phase(self):
        duration = 10.0
        num_pts = 1000
        dt = duration / num_pts
        time = np.linspace(0.0, duration + 10.0, num_pts + 2000)
        phase = ExponentialPV(tau=duration)
        xs = []
        for t in time:
            xs.append(phase.get_value(dt))
        plt.figure(figsize=(12,10))
        plt.suptitle("Exponential Phase Variable", fontsize=18, fontweight='bold')
        plt.plot(time, xs, lw=5.0)
        plt.savefig("exp_phase.svg")
        # plt.show()

    def test_lin_phase(self):
        pv = LinearPV(max_time=5.0)
        dt = 0.01
        pad = 100
        xs = pv.get_rollout(dt, pad)
        plt.figure(figsize=(12,10))
        plt.plot(xs, lw=3.0)
        plt.ylim(0.0, 1.2)
        plt.suptitle("Linear Phase Variable", fontsize=18, fontweight='bold')
        # plt.show()


    # def test_cs_error_buildup(self):
    #     duration = 10.0
    #     num_pts = 1000
    #     dt = duration / num_pts
    #     time = np.linspace(0.0, duration, num_pts)
    #     cs = CanonicalSystem(tau=duration, alpha_p=100.0)
    #     pose_error = 0.0
    #     xs = []
    #     new_time = np.hstack((time, np.linspace(duration, duration + 3.0, 300)))
    #     for t in new_time:
    #         if t > 1.0 and t < 3.0:
    #             xs.append(cs.get_value(dt, pose_error, in_contact=1))
    #             pose_error += 0.01
    #         elif t >= 3.0:
    #             xs.append(cs.get_value(dt, pose_error, in_contact=1))
    #             pose_error -= 0.01
    #             pose_error = max(pose_error, 0.0)
    #         else:
    #             xs.append(cs.get_value(dt))
    #     plt.figure(figsize=(12,10))
    #     plt.title("Pose Error", fontsize=18, fontweight='bold')
    #     plt.ylim(0.0, 1.0)
    #     plt.plot(new_time, xs, lw=3.0)
    #     plt.show()
        

if __name__ == '__main__':
    unittest.main()
