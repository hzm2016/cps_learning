import unittest
import numpy as np
import matplotlib.pyplot as plt
import pyquaternion as pyquat
from copy import copy
from ll4ma_movement_primitives.dmps import QuaternionDMP, DMPConfig
from ll4ma_movement_primitives.util import quaternion


# TODO this file won't run yet, need to update for latest refactoring


class TestQuaternionDMP(unittest.TestCase):
 
    def test_quaternion_dmp(self):
        tau = 10.0
        num_pts = 1000
        dt = tau / num_pts
        time = np.linspace(0.0, tau, 2.0 * num_pts)
        q0 = pyquat.Quaternion(0.3717, -0.4993, -0.6162, 0.4825) # w, x, y, z
        g0 = pyquat.Quaternion(0.2471, 0.1797, 0.3182, -0.8974) # w, x, y, z
        q_seq = pyquat.Quaternion.intermediates(q0, g0, num_pts - 2, include_endpoints=True)
        # convert to numpy arrays
        q_seq = np.array([q.q for q in q_seq]).T
        # repeat goal for some duration
        q_seq = np.hstack((q_seq, np.tile(q_seq[:,-1][:,None], num_pts)))
        # put into (x, y, z, w) format
        qs = copy(q_seq)
        qs[-1,:] = q_seq[0,:]
        qs[:3,:] = q_seq[1:,:]

        qs_shift = copy(qs)
        qs_shift = np.delete(qs_shift, -1, 1)
        qs_shift = np.hstack((qs_shift[:,0][:,None], qs_shift))
        omegas = 2.0 * quaternion.prod(qs_shift, quaternion.conj(qs)) / dt
        omegas = omegas[:3,:] # get rid of scalar part, implicitly considered zero
        d_omegas = np.hstack((np.zeros(3)[:,None], np.diff(omegas) / dt))
        for i in range(3):
            for j in range(d_omegas.shape[1]):
                if d_omegas[i,j] > 0.1:
                    d_omegas[i,j] = 0.1

        demo = {}
        demo['q'] = qs
        demo['omega'] = omegas
        demo['d_omega'] = np.zeros((3,omegas.shape[1])) # get huge instantaneous accel if you int omega

        alpha = 50.0
        beta = alpha / 4.0
        gamma = 2.0

        dmp_config = DMPConfig()
        dmp_config.alpha = 50.0
        dmp_config.beta = dmp_config.alpha / 4.0
        dmp_config.gamma = 2.0
        dmp_config.dt = dt
        dmp_config.tau = tau
        dmp_config.w = [] # TODO hack to get around bad inheritance structure

        dmp = QuaternionDMP(dmp_config, [demo])
        
        qs_learned = None
        omegas_learned = None
        for t in time:
            x = dmp.phase.get_value(dt)
            bfs = dmp.bfs.get_value(x)
            q, omega, d_omega = dmp.ts.get_value(x, bfs, dt)
            qs_learned = q if qs_learned is None else np.hstack((qs_learned, q))
            omegas_learned = omega if omegas_learned is None else np.hstack((omegas_learned, omega))

        plt.figure(figsize=(12,12))
        plt.suptitle("Cartesian DMP", fontsize=18, fontweight='bold')
        plt.subplot(211)

        # plt.plot(time, qs[0,:], lw=4.0, ls='--', color='g')
        # plt.plot(time, qs[1,:], lw=4.0, ls='--', color='r')
        # plt.plot(time, qs[2,:], lw=4.0, ls='--', color='c')
        # plt.plot(time, qs[3,:], lw=4.0, ls='--', color='b')
            
        plt.plot(time, qs_learned[0,:], lw=3.0, color='g')
        plt.plot(time, qs_learned[1,:], lw=3.0, color='r')
        plt.plot(time, qs_learned[2,:], lw=3.0, color='c')
        plt.plot(time, qs_learned[3,:], lw=3.0, color='b')
        plt.ylabel('Quaternion', fontsize=16)
        plt.subplot(212)
        plt.plot(time, omegas_learned[0,:], lw=3.0, color='b')
        plt.plot(time, omegas_learned[1,:], lw=3.0, color='g')
        plt.plot(time, omegas_learned[2,:], lw=3.0, color='r')
        plt.ylabel('Angular Velocity', fontsize=16)
        plt.xlabel('Time (Sec)', fontsize=16)
        plt.show()


if __name__ == '__main__':
    unittest.main()
