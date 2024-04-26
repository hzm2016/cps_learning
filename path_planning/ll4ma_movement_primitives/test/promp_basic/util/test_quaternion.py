import unittest
import numpy as np
from ll4ma_movement_primitives.util import quaternion


class TestQuaternion(unittest.TestCase):

    def setUp(self):
        self.q1 =      np.array([-0.09754239,  0.08603414,  0.94800085,  0.29047891])[:,None]
        self.q2 =      np.array([ 0.84682529, -0.04538183, -0.11360851, -0.51761040])[:,None]
        self.q3 =      np.array([ 0.22507510,  0.96668174,  0.09069538, -0.08149817])[:,None]
        self.q4 =      np.array([ 0.24204358, -0.69017305,  0.28571261, -0.61922886])[:,None]
        self.q1_conj = np.array([ 0.09754239, -0.08603414, -0.94800085,  0.29047891])[:,None]
        self.q2_conj = np.array([-0.84682529,  0.04538183,  0.11360851, -0.5176104 ])[:,None]
        self.q1_q2 =   np.array([ 0.32972164,  0.73399482, -0.59212521,  0.04385181])[:,None]        
        self.q1_q3 =   np.array([-0.83528312,  0.49600706, -0.16457183, -0.17086607])[:,None]
        self.q2_q1 =   np.array([ 0.26322605, -0.84942407, -0.45526675,  0.04385181])[:,None]
        self.q2_q4 =   np.array([-0.74103862,  0.11589562, -0.65100976,  0.11668875])[:,None]
        self.q3_q1 =   np.array([ 0.98194131,  0.05157101,  0.06274135, -0.17086607])[:,None]

    def test_prod_left_vec(self):
        q2q3 = np.hstack((self.q2, self.q3))
        q1_q2q3 = np.hstack((self.q1_q2, self.q1_q3))
        self.assertTrue(np.allclose(q1_q2q3, quaternion.prod(self.q1, q2q3)))

    def test_prod_right_vec(self):
        q2q3 = np.hstack((self.q2, self.q3))
        q2q3_q1 = np.hstack((self.q2_q1, self.q3_q1))
        self.assertTrue(np.allclose(q2q3_q1, quaternion.prod(q2q3, self.q1)))

    def test_prod_both_vec(self):
        self.assertTrue(np.allclose(self.q1_q2, quaternion.prod(self.q1, self.q2)))

    def test_prod_both_arr(self):
        q1q2 = np.hstack((self.q1, self.q2))
        q3q4 = np.hstack((self.q3, self.q4))
        q1q2_q3q4 = np.hstack((self.q1_q3, self.q2_q4))
        self.assertTrue(np.allclose(q1q2_q3q4, quaternion.prod(q1q2, q3q4)))

    def test_conj_vec(self):
        self.assertTrue(np.allclose(self.q1_conj, quaternion.conj(self.q1)))

    def test_conj_arr(self):
        q1q2 = np.hstack((self.q1, self.q2))
        q1cq2c = np.hstack((self.q1_conj, self.q2_conj))
        self.assertTrue(np.allclose(q1cq2c, quaternion.conj(q1q2)))

    def test_log_vec_nonzero(self):
        v = self.q1[:3]
        w = self.q1[-1]
        logq1 = np.arccos(w) * v / np.linalg.norm(v)
        self.assertTrue(np.allclose(logq1, quaternion.log(self.q1)))

    def test_log_vec_zero(self):
        v = np.zeros(4)
        v[-1] = 1
        self.assertTrue(np.allclose(np.zeros(3), quaternion.log(v)))

    def test_log_arr_nonzero(self):
        q1q2 = np.hstack((self.q1, self.q2))
        v1 = self.q1[:3]
        v2 = self.q2[:3]
        w1 = self.q1[-1]
        w2 = self.q2[-1]
        logq1 = np.arccos(w1) * v1 / np.linalg.norm(v1)
        logq2 = np.arccos(w2) * v2 / np.linalg.norm(v2)
        logq1logq2 = np.hstack((logq1, logq2))
        self.assertTrue(np.allclose(logq1logq2, quaternion.log(q1q2)))
        
    def test_log_arr_zero(self):
        a = np.zeros((4, 2))
        a[-1,:] = np.ones((1,2))
        self.assertTrue(np.allclose(np.zeros((3,2)), quaternion.log(a)))

    def test_log_arr_mixed(self):
        v1 = self.q1[:3]
        w1 = self.q1[-1]
        q2 = np.zeros((4,1))
        q2[-1] = 1
        q1q2 = np.hstack((self.q1, q2))
        logq1 = np.arccos(w1) * v1 / np.linalg.norm(v1)
        logq1logq2 = np.hstack((logq1, q2[:3]))
        self.assertTrue(np.allclose(logq1logq2, quaternion.log(q1q2)))

    def test_exp_nonzero(self):
        v = self.q1[:3]
        e = np.zeros((4,1))
        e[-1] = np.cos(np.linalg.norm(v))
        e[:3] = np.sin(np.linalg.norm(v)) * v / np.linalg.norm(v)
        self.assertTrue(np.allclose(e, quaternion.exp(v)))

    def test_exp_zero(self):
        v = np.zeros((3,1))
        e = np.zeros((4,1))
        e[-1] = 1
        self.assertTrue(np.allclose(e, quaternion.exp(v)))


if __name__ == '__main__':
    unittest.main()
