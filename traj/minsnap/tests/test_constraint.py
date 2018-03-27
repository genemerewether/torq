# -*- coding: utf-8 -*-

"""
Copyright 2016, by the California Institute of Technology. ALL RIGHTS
RESERVED. United States Government Sponsorship acknowledged. Any commercial
use must be negotiated with the Office of Technology Transfer at the
California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this
software, the user agrees to comply with all applicable U.S. export laws and
regulations. User has the responsibility to obtain export licenses, or other
export authority as may be required before exporting such information to
foreign countries or providing access to foreign persons.
"""

__author__ = "Gene Merewether"
__email__ = "mereweth@jpl.nasa.gov"


# Import statements
# =================

import unittest
# from unittest.case import SkipTest
from minsnap import constraint
from minsnap import utils

import numpy as np
import scipy.sparse
sp = scipy


class Base(unittest.TestCase):

    """
    Base class for tests

    This class defines a common `setUp` method that defines attributes which are
    used in the various tests.
    """

    def setUp(self):
        pass

    @classmethod
    def setUpClass(cls):
        """Called once before all tests in this class."""
        pass

    @classmethod
    def tearDownClass(cls):
        """Called once after all tests in this class."""
        pass


class Input(Base):

    def setUp(self):
        """
        TODO(mereweth@jpl.nasa.gov) check each bad argument in isolation
        """
        self.ok_times = [1, 2]
        self.ok_order = 9

        self.nan_times = [1, np.nan]
        self.nan_order = np.nan

        self.zero_times = [0, 1]
        self.zero_order = 0

        self.negative_times = [1, -4]
        self.negative_order = -4

    def test_sparse_values_nan(self):
        np.testing.assert_equal(constraint.sparse_values(self.nan_times,
                                                         self.nan_order),
                                ([], []))

    def test_sparse_values_negative(self):
        np.testing.assert_equal(constraint.sparse_values(self.negative_times,
                                                         self.negative_order),
                                ([], []))

    def test_sparse_values_zero(self):
        np.testing.assert_equal(constraint.sparse_values(self.zero_times,
                                                         self.zero_order),
                                ([], []))

    def test_sparse_indices_nan(self):
        np.testing.assert_equal(constraint.sparse_indices(self.nan_times,
                                                          self.nan_order),
                                ([], [], [], []))

    def test_sparse_indices_negative(self):
        np.testing.assert_equal(constraint.sparse_indices(self.negative_times,
                                                          self.negative_order),
                                ([], [], [], []))

    def test_sparse_indices_zero(self):
        np.testing.assert_equal(constraint.sparse_indices(self.zero_times,
                                                          self.zero_order),
                                ([], [], [], []))


class Output(Base):
    # TODO(mereweth@jpl.nasa.gov) - check block_constraint output sparse
    # matrix type?
    pass


class IntComputationSingleSegment(Base):

    def test_block_constraint_odd_order(self):
        times = 3
        order = 9

        expect = [[1, 0, 0,  0,   0,   0,    0,     0,      0,      0],
                  [0, 1, 0,  0,   0,   0,    0,     0,      0,      0],
                  [0, 0, 2,  0,   0,   0,    0,     0,      0,      0],
                  [0, 0, 0,  6,   0,   0,    0,     0,      0,      0],
                  [0, 0, 0,  0,  24,   0,    0,     0,      0,      0],
                  [1, 3, 9, 27,  81, 243,  729,  2187,   6561,  19683],
                  [0, 1, 6, 27, 108, 405, 1458,  5103,  17496,  59049],
                  [0, 0, 2, 18, 108, 540, 2430, 10206,  40824, 157464],
                  [0, 0, 0,  6,  72, 540, 3240, 17010,  81648, 367416],
                  [0, 0, 0,  0,  24, 360, 3240, 22680, 136080, 734832]]

        # Compare result and expected value
        result = constraint.block_constraint(times, order)
        np.testing.assert_array_equal(result.todense(), expect)

    def test_sparse_values_odd_order(self):
        times = 2
        order = 7

        a_0_val = [1, 1, 2, 6]
        a_t_val = [1, 2, 1, 4, 4, 2, 8, 12, 12, 6, 16, 32, 48, 48, 32, 80, 160,
                   240, 64, 192, 480, 960, 128, 448, 1344, 3360]

        expect = (a_0_val, a_t_val)

        # Compare result and expected value
        result = constraint.sparse_values(times, order)

        assert(len(expect) == len(result))
        for i in range(0, len(expect)):
            np.testing.assert_array_equal(result[i], expect[i])

    def test_sparse_indices_odd_order(self):
        times = 2
        order = 7

        a_0_row = [0, 1, 2, 3]
        a_0_col = a_0_row

        a_t_row = [4, 4, 5, 4, 5, 6, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5,
                   6, 7, 4, 5, 6, 7]
        a_t_col = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6,
                   6, 6, 7, 7, 7, 7]

        expect = (a_0_row, a_0_col, a_t_row, a_t_col)

        # Compare result and expected value
        result = constraint.sparse_indices(times, order)

        assert(len(expect) == len(result))
        for i in range(0, len(expect)):
            np.testing.assert_array_equal(result[i], expect[i])

    def test_block_constraint_even_order(self):
        times = 2
        order = 8

        expect = [[1,     0,     0,     0,     0,     0,     0,     0,     0],
                  [0,     1,     0,     0,     0,     0,     0,     0,     0],
                  [0,     0,     2,     0,     0,     0,     0,     0,     0],
                  [0,     0,     0,     6,     0,     0,     0,     0,     0],
                  [1,     2,     4,     8,    16,    32,    64,   128,   256],
                  [0,     1,     4,    12,    32,    80,   192,   448,  1024],
                  [0,     0,     2,    12,    48,   160,   480,  1344,  3584],
                  [0,     0,     0,     6,    48,   240,   960,  3360, 10752]]

        # Compare result and expected value
        result = constraint.block_constraint(times, order)
        np.testing.assert_array_equal(result.todense(), expect)

    def test_sparse_values_even_order(self):
        times = 2
        order = 6

        a_0_val = [1, 1, 2]
        a_t_val = [1, 2, 1, 4, 4, 2, 8, 12, 12, 16, 32, 48, 32, 80, 160, 64,
                   192, 480]

        expect = (a_0_val, a_t_val)

        # Compare result and expected value
        result = constraint.sparse_values(times, order)

        assert(len(expect) == len(result))
        for i in range(0, len(expect)):
            np.testing.assert_array_equal(result[i], expect[i])

    def test_sparse_indices_even_order(self):
        times = 3
        order = 6

        a_0_row = [0, 1, 2]
        a_0_col = a_0_row

        a_t_row = [3, 3, 4, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5]
        a_t_col = [0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]

        expect = (a_0_row, a_0_col, a_t_row, a_t_col)

        # Compare result and expected value
        result = constraint.sparse_indices(times, order)

        assert(len(expect) == len(result))
        for i in range(0, len(expect)):
            np.testing.assert_array_equal(result[i], expect[i])

    def test_block_ineq_constraint_internal_ineq(self):
        der_fixed = np.array([[True,False,False,False,True],
                              [True,False,False,False,True],
                              [True,False,False,False,True],
                              [True,False,False,False,True],
                              [True,False,False,False,True]])
        der_ineq = np.array([[False,True,True,True,False],
                             [False,False,False,False,False],
                             [False,False,False,False,False],
                             [False,False,False,False,False],
                             [False,False,False,False,False]])

        expect = np.array([ [1,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0],
                            [0,0,0,0,0, 1,0,0,0,0, 0,0,0,0,0],
                            [0,0,0,0,0, 0,0,0,0,0, 1,0,0,0,0],
                            [-1,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0],
                            [0,0,0,0,0, -1,0,0,0,0, 0,0,0,0,0],
                            [0,0,0,0,0, 0,0,0,0,0, -1,0,0,0,0]])

        result = constraint.block_ineq_constraint(der_fixed,der_ineq).toarray()

        np.testing.assert_array_equal(result, expect)


    def test_block_ineq_constraint_one_fixed(self):
        der_fixed = np.array([[True,False,True,False,True],
                              [True,False,False,False,True],
                              [True,False,False,False,True],
                              [True,False,False,False,True],
                              [True,False,False,False,True]])
        der_ineq = np.array([[False,True,False,True,False],
                             [False,False,False,False,False],
                             [False,False,False,False,False],
                             [False,False,False,False,False],
                             [False,False,False,False,False]])

        expect = np.array([ [1,0,0,0,0, 0,0,0,0, 0,0,0,0,0],
                            [0,0,0,0,0, 0,0,0,0, 1,0,0,0,0],
                            [-1,0,0,0,0, 0,0,0,0, 0,0,0,0,0],
                            [0,0,0,0,0, 0,0,0,0, -1,0,0,0,0]])

        result = constraint.block_ineq_constraint(der_fixed,der_ineq).toarray()
        #import pdb; pdb.set_trace()
        np.testing.assert_array_equal(result, expect)

    def test_insert(self):
        times = [2]
        order = 3
        E1 = np.array([[  1.,   0.,   0.,   0.],
                        [  0.,   1.,   0.,   0.],
                        [  1.,   3.,   9.,  27.],
                        [  0.,   1.,   6.,  27.]])
        E2 = np.array([[ 1,  0,  0,  0],
                        [ 0,  1,  0,  0],
                        [ 1,  2,  4,  8],
                        [ 0,  1,  4, 12]])
        expect1 = sp.linalg.block_diag(E1,E2)
        expect2 = sp.linalg.block_diag(E1,E1)
        expect3 = sp.linalg.block_diag(E2,E1)
        result = constraint.block_constraint(times, order).tocsc(copy=True)
        result1 = constraint.insert(result,0,[3,3],order).copy()
        result2 = constraint.insert(result,1,[3,3],order).copy()
        result3 = constraint.insert(result,2,[3,3],order).copy()
        np.testing.assert_allclose(result1.todense(), expect1)
        np.testing.assert_allclose(result2.todense(), expect2)
        np.testing.assert_allclose(result3.todense(), expect3)


class FloatComputationSingleSegment(Base):

    def test_block_constraint_even_order(self):
        times = 3.5
        order = 4

        expect = [[1.0,    0.0,    0.00,    0.000,    0.0000],
                  [0.0,    1.0,    0.00,    0.000,    0.0000],
                  [1.0,    3.5,   12.25,   42.875,  150.0625],
                  [0.0,    1.0,    7.00,   36.750,  171.5000]]

        # Compare result and expected value
        result = constraint.block_constraint(times, order)
        np.testing.assert_allclose(result.todense(), expect)


class IntComputationMultiSegment(Base):

    def test_sparse_indices_even_order(self):
        times = [2, 2, 3]
        order = 4

        a_0_row = [0, 1, 4, 5, 8, 9]
        a_0_col = [0, 1, 5, 6, 10, 11]

        a_t_row = [2, 2, 3, 2, 3, 2, 3, 2, 3, 6, 6, 7, 6, 7, 6, 7, 6, 7, 10, 10,
                   11, 10, 11, 10, 11, 10, 11]
        a_t_col = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 11,
                   11, 12, 12, 13, 13, 14, 14]

        expect = (a_0_row, a_0_col, a_t_row, a_t_col)

        # Compare result and expected value
        result = constraint.sparse_indices(times, order)

        assert(len(expect) == len(result))
        for i in range(0, len(expect)):
            np.testing.assert_array_equal(result[i], expect[i])


class FloatComputationMultiSegment(Base):

    def test_block_constraint_even_order(self):
        times = [2.9, 1.2]
        order = 4

        expect = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 2.9, 8.41, 24.389, 70.7281, 0, 0, 0, 0, 0],
                  [0, 1, 5.8, 25.23, 97.556, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 1.2, 1.44, 1.728, 2.0736],
                  [0, 0, 0, 0, 0, 0, 1, 2.4, 4.32, 6.912]]

        # Compare result and expected value
        result = constraint.block_constraint(times, order)
        np.testing.assert_allclose(result.todense(), expect)


    def test_insert_2_seg(self):
        costs = [1, 0, 0, 0, 0]
        times = [2,2]
        order = 3
        E1 = np.array([[  1.,   0.,   0.,   0.],
                        [  0.,   1.,   0.,   0.],
                        [  1.,   3.,   9.,  27.],
                        [  0.,   1.,   6.,  27.]])
        E2 = np.array([[ 1,  0,  0,  0],
                        [ 0,  1,  0,  0],
                        [ 1,  2,  4,  8],
                        [ 0,  1,  4, 12]])
        expect0 = sp.linalg.block_diag(E1,E2,E2)
        expect1 = sp.linalg.block_diag(E1,E1,E2)
        expect2 = sp.linalg.block_diag(E2,E1,E1)
        expect3 = sp.linalg.block_diag(E2,E2,E1)
        result = constraint.block_constraint(times, order).tocsc(copy=True)
        result0 = constraint.insert(result,0,[3,3], order).copy()
        result1 = constraint.insert(result,1,[3,3], order).copy()
        result2 = constraint.insert(result,2,[3,3], order).copy()
        result3 = constraint.insert(result,3,[3,3], order).copy()
        np.testing.assert_allclose(result0.todense(), expect0)
        np.testing.assert_allclose(result1.todense(), expect1)
        np.testing.assert_allclose(result2.todense(), expect2)
        np.testing.assert_allclose(result3.todense(), expect3)

    def test_delete_2_seg(self):
        costs = [1, 0, 0, 0, 0]
        times = [3,2]
        order = 3
        E1 = np.array([[  1.,   0.,   0.,   0.],
                        [  0.,   1.,   0.,   0.],
                        [  1.,   3.,   9.,  27.],
                        [  0.,   1.,   6.,  27.]])
        E2 = np.array([[ 1,  0,  0,  0],
                        [ 0,  1,  0,  0],
                        [ 1,  2,  4,  8],
                        [ 0,  1,  4, 12]])
        result = constraint.block_constraint(times, order).tocsc(copy=True)
        result1 = constraint.delete(result,2,3, order).copy()
        result2 = constraint.delete(result,1,3, order).copy()
        result3 = constraint.delete(result,0,2, order).copy()
        np.testing.assert_allclose(result1.todense(), E1)
        np.testing.assert_allclose(result2.todense(), E1)
        np.testing.assert_allclose(result3.todense(), E2)

    def test_delete_multi_seg(self):
        costs = [1, 0, 0, 0, 0]
        times = [2,2,2]
        order = 3
        E1 = np.array([[  1.,   0.,   0.,   0.],
                        [  0.,   1.,   0.,   0.],
                        [  1.,   3.,   9.,  27.],
                        [  0.,   1.,   6.,  27.]])
        E2 = np.array([[ 1,  0,  0,  0],
                        [ 0,  1,  0,  0],
                        [ 1,  2,  4,  8],
                        [ 0,  1,  4, 12]])
        expect1 = sp.linalg.block_diag(E1,E2)
        expect2 = sp.linalg.block_diag(E2,E1)
        result = constraint.block_constraint(times, order).tocsc(copy=True)
        result1 = constraint.delete(result,1,3, order).copy()
        result2 = constraint.delete(result,2,3, order).copy()
        np.testing.assert_allclose(result1.todense(), expect1)
        np.testing.assert_allclose(result2.todense(), expect2)

    def test_delete_then_insert(self):
        costs = [0, 0, 0, 1, 0]
        times = [2,2,2,2,2]
        order = 9
        expect = constraint.block_constraint(times, order).tocsc(copy=True)
        result1 = constraint.delete(expect,3,2, order)
        result = constraint.insert(result1,3,[2,2], order)
        np.testing.assert_allclose(result.todense(), expect.todense())

    def test_insert_then_delete(self):
        costs = [0, 0, 0, 1, 0]
        times = [2,2,2,2,2]
        order = 9
        expect = constraint.block_constraint(times, order).tocsc(copy=True)
        result1 = constraint.insert(expect,3,[2,2], order)
        result = constraint.delete(result1,3,2, order)
        np.testing.assert_allclose(result.todense(), expect.todense())



if __name__ == '__main__':
    unittest.main()
