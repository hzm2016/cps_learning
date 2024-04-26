"""
This is a utility class for standard operations on quaternions.
"""

import numpy as np
from copy import copy


def prod(q1, q2):
    """
    Computes the product of two quaternions.

    This function defers the actual product operation to a helper function,
    and primarily manages the different cases of whether the inputs are
    single quaternions or arrays of quaternions.
    """
    if q1.ndim == 1:
        q1 = q1[:,None] # make column vector
    if q2.ndim == 1:
        q2 = q2[:,None] # make column vector

    if q1.shape == q2.shape and q1.shape == (4,1):
        # product of two individual quaternions
        p = _prod(q1, q2)
    elif q1.shape[1] == 1:
        # q1 is single quaternion and q2 is array of quaternions
        p = copy(q2)
        for i in range(q2.shape[1]):
            p[:,i] = np.squeeze(_prod(q1, q2[:,i][:,None]))
    elif q2.shape[1] == 1:
        # q2 is single quaternion and q1 is array of quaternions
        p = copy(q1)
        for i in range(q1.shape[1]):
            p[:,i] = np.squeeze(_prod(q1[:,i][:,None], q2))
    else:
        # both are arrays
        if q1.shape != q2.shape:
            print "Arrays must be of equal size when doing batch product."
            p = None
        else:
            p = copy(q1)
            for i in range(q1.shape[1]):
                p[:,i] = np.squeeze(_prod(q1[:,i][:,None], q2[:,i][:,None]))
    return p
    
def conj(q):
    """
    Computes quaternion conjugation.
    """
    if q.ndim == 1:
        q = q[:,None] # make column vector
    q_conj = np.zeros(q.shape)
    q_conj[-1,:] = q[-1,:]
    q_conj[:3,:] = -q[:3,:]
    return q_conj
    
def log(q):
    """
    Computes the logarithm of a quaternion.
    """
    if q.ndim == 1:
        q = q[:,None] # make column vector
    l = np.zeros((3, q.shape[1]))
    for i in range(q.shape[1]):
        if not np.array_equal(q[:3,i], np.zeros(3)):
            # There seem to be floating point errors sometimes pushing the scalar value outside
            # the range [-1,1] for which arccos is undefined. Need to saturate to range [-1,1].
            if abs(q[-1,i]) > 1.0:
                q[-1,i] = np.sign(q[-1,i]) # will be 1.0 or -1.0, which is the desired saturation
            l[:3,i] = np.arccos(q[-1,i]) * q[:3,i] / np.linalg.norm(q[:3,i])
    return l

def exp(v):
    """
    Computes quaternion exponentiation.
    """
    if v.ndim == 1:
        v = v[:,None] # make column vector
    q = np.zeros((4,1))
    q[-1] = 1
    if not np.array_equal(v, np.zeros((3,1))):
        norm_v = np.linalg.norm(v)
        q[-1] = np.cos(norm_v)
        q[:3] = np.sin(norm_v) * v / norm_v
    return q

def err(q1, q2):
    """
    Computes the difference between two quaternions.
    """
    return 2.0 * log(prod(q1, conj(q2)))

def integ(q, omega, dt):
    """
    Performs quaternion integration.
    """
    return prod(exp(0.5 * dt * omega), q)

def _prod(q1, q2):
    """
    Helper function that computes the product between two quaternions.
    """
    q = np.zeros((4,1))
    q[-1] = q1[-1] * q2[-1] - np.dot(q1[:3].T, q2[:3])
    q[:3] = q1[-1] * q2[:3] + q2[-1] * q1[:3] + np.cross(q1[:3], q2[:3], axis=0)
    return q
    
