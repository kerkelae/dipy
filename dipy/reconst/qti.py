# -*- coding: utf-8 -*-
"""
Code for q-space trajectory imaging.

For details of the model, please see "Q-space trajectory imaging for
multidimensional diffusion MRI of the human brain" by Westin et al. (2016)
https://doi.org/10.1016/j.neuroimage.2016.02.039
"""

import numpy as np


def from_3x3_to_6x1(T):
    """Convert symmetric second order tensor to first order tensor. (Eq 47)
    Args:
        T : 3 x 3 tensor.
    Returns:
        V : Corresponding 6 x 1 tensor.
    """
    C = np.sqrt(2)
    V = np.array([[T[0, 0],
                   T[1, 1],
                   T[2, 2],
                   C * T[1, 2],
                   C * T[0, 2],
                   C * T[0, 1]]]).T
    return V


def from_6x1_to_3x3(V):
    """Convert first order tensor to symmetric second order tensor.
    Args:
        V : 6 x 1 tensor.
    Returns:
        T : Corresponding 3 x 3 tensor.
    """
    C = np.sqrt(1 / 2)
    T = np.array([[V[0, 0], C * V[5, 0], C * V[4, 0]],
                  [C * V[5, 0], V[1, 0], C * V[3, 0]],
                  [C * V[4, 0], C * V[3, 0], V[2, 0]]])
    return T


def from_6x6_to_21x1(T):
    """Convert symmetric second order tensor to first order tensor. (Eq 48)
    Args:
        T : 6 x 6 array.
    Returns:
        V : Corresponding 21 x 1 tensor.
    """
    C2 = np.sqrt(2)
    V = np.array([[T[0, 0], T[1, 1], T[2, 2],
                   C2 * T[1, 2], C2 * T[0, 2], C2 * T[0, 1],
                   C2 * T[0, 3], C2 * T[0, 4], C2 * T[0, 5],
                   C2 * T[1, 3], C2 * T[1, 4], C2 * T[1, 5],
                   C2 * T[2, 3], C2 * T[2, 4], C2 * T[2, 5],
                   T[3, 3], T[4, 4], T[5, 5],
                   C2 * T[3, 4], C2 * T[4, 5], C2 * T[5, 3]]]).T
    return V


def from_21x1_to_6x6(V):
    """Convert first order tensor to symmetric second order tensor.
    Args:
        T: 21 x 1 tensor.
    Returns:
        Corresponding 6 x 6 tensor.
    """
    v = V[:, 0]  # Easier to read without extra dimension
    C2 = np.sqrt(1 / 2)
    T = np.array([[v[0], C2*v[5], C2*v[4], C2*v[6], C2*v[7], C2*v[8]],
                  [C2*v[5], v[1], C2*v[3], C2*v[9], C2*v[10], C2*v[11]],
                  [C2*v[4], C2*v[3], v[2], C2*v[12], C2*v[13], C2*v[14]],
                  [C2*v[6], C2*v[9], C2*v[12], v[15], C2*v[18], C2*v[20]],
                  [C2*v[7], C2*v[10], C2*v[13], C2*v[18], v[16], C2*v[19]],
                  [C2*v[8], C2*v[11], C2*v[14], C2*v[20], C2*v[19], v[17]]])
    return T

