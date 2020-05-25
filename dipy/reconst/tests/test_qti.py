# -*- coding: utf-8 -*-
"""
Tests for dipy/reconst/qti.py
"""

import numpy as np
import numpy.testing as npt

import dipy.reconst.qti as qti


def test_tensor_conversion():
    # Test conversion between 6 x 1 and 3 x 3 arrays
    np.random.seed(12345)
    Vs = np.random.random((100, 6, 1))
    for V in Vs:
        T = qti.from_6x1_to_3x3(V)
        for i in range(3):
            for j in [k for k in range(3) if k != i]:
                npt.assert_equal(T[i, j], T[i, j])
        V_converted = qti.from_3x3_to_6x1(T)
        npt.assert_almost_equal(V_converted, V)
    # Test conversion between 21 x 1 and 6 x 6 arrays
    Vs = np.random.random((100, 21, 1))
    for V in Vs:
        T = qti.from_21x1_to_6x6(V)
        for i in range(6):
            for j in [k for k in range(6) if k != i]:
                npt.assert_equal(T[i, j], T[i, j])
        V_converted = qti.from_6x6_to_21x1(T)
        npt.assert_almost_equal(V_converted, V)
    return
