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
from minsnap import cost
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
        self.ok_cost = [0, 0, 0, 1]

        self.nan_times = [2.5, np.nan]
        self.nan_order = np.nan
        self.nan_cost = [0, 0, 0, np.nan]

        self.negative_times = [1, -4]
        self.negative_order = -4
        # TODO(mereweth@jpl.nasa.gov) what should we do here?
        self.negative_cost = [0, 0, 0, -1]

        self.zero_times = [0, 1]
        self.zero_order = 0
        # TODO(mereweth@jpl.nasa.gov) what should we do here?
        self.zero_cost = [0, 0, 0, 0]

        # TODO(mereweth@jpl.nasa.gov) test!!


class IntComputationSingleSegment(Base):

    def test_block_cost_odd_order(self):
        times = 3
        order = 9
        costs = [0, 0, 0, 1]

        expect = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 216, 1296, 6480, 29160, 122472, 489888, 1889568],
                  [0, 0, 0, 1296, 10368, 58320, 279936, 1224720, 5038848,
                   19840464],
                  [0, 0, 0, 6480, 58320, 349920, 1749600, 7873200, 33067440,
                   132269760],
                  [0, 0, 0, 29160, 279936, 1749600, 8997942.85714286, 41334300, 176359680,
                   714256704],
                  [0, 0, 0, 122472, 1224720, 7873200, 41334300, 192893400, 833299488,
                   3408952450.90909],
                  [0, 0, 0, 489888, 5038848, 33067440, 176359680, 833299488, 3636215947.63636,
                   14999390784],
                  [0, 0, 0, 1889568, 19840464, 132269760, 714256704, 3408952450.90909, 14999390784,
                   62305161718.1538]]

        # Compare result and expected value
        result = cost.block_cost(times, costs, order)
        np.testing.assert_allclose(result.todense(), expect)

    def test_insert_1_seg(self):
        costs = [1, 0, 0, 0, 0]
        times = 2
        order = 3
        E1 = np.array([[6.0,9.0,18.0,40.5],
                    [9., 18., 40.5,97.2],
                    [18., 40.5,97.2,243.],
                    [40.5, 97.2, 243., 624.85714286]])
        E2 = np.array([[4., 4., 5.33333333,8.],
                    [4.,5.33333333,8., 12.8],
                    [5.33333333,8.,12.8,21.33333333],
                    [8.,12.8,21.33333333,36.57142857]])
        expect1 = sp.linalg.block_diag(E1,E2)
        expect2 = sp.linalg.block_diag(E1,E1)
        expect3 = sp.linalg.block_diag(E2,E1)
        result = cost.block_cost(times, costs, order).tocsc(copy=True)
        result1 = cost.insert(result,0,[3,3],costs,order).copy()
        result2 = cost.insert(result,1,[3,3],costs,order).copy()
        result3 = cost.insert(result,2,[3,3],costs,order).copy()
        np.testing.assert_allclose(result1.todense(), expect1)
        np.testing.assert_allclose(result2.todense(), expect2)
        np.testing.assert_allclose(result3.todense(), expect3)



class FloatComputationSingleSegment(Base):
    pass


class IntComputationMultiSegment(Base):
    pass


class FloatComputationMultiSegment(Base):
    def test_insert_2_seg(self):
        costs = [1, 0, 0, 0, 0]
        times = [2,2]
        order = 3
        E1 = np.array([[6.0,9.0,18.0,40.5],
                    [9., 18., 40.5,97.2],
                    [18., 40.5,97.2,243.],
                    [40.5, 97.2, 243., 624.85714286]])
        E2 = np.array([[4., 4., 5.33333333,8.],
                    [4.,5.33333333,8., 12.8],
                    [5.33333333,8.,12.8,21.33333333],
                    [8.,12.8,21.33333333,36.57142857]])
        expect0 = sp.linalg.block_diag(E1,E2,E2)
        expect1 = sp.linalg.block_diag(E1,E1,E2)
        expect2 = sp.linalg.block_diag(E2,E1,E1)
        expect3 = sp.linalg.block_diag(E2,E2,E1)
        result = cost.block_cost(times, costs, order).tocsc(copy=True)
        result0 = cost.insert(result,0,[3,3],costs,order).copy()
        result1 = cost.insert(result,1,[3,3],costs,order).copy()
        result2 = cost.insert(result,2,[3,3],costs,order).copy()
        result3 = cost.insert(result,3,[3,3],costs,order).copy()
        np.testing.assert_allclose(result0.todense(), expect0)
        np.testing.assert_allclose(result1.todense(), expect1)
        np.testing.assert_allclose(result2.todense(), expect2)
        np.testing.assert_allclose(result3.todense(), expect3)

    def test_delete_2_seg(self):
        costs = [1, 0, 0, 0, 0]
        times = [3,2]
        order = 3
        expect1 = [[6.0,9.0,18.0,40.5],
                    [9., 18., 40.5,97.2],
                    [18., 40.5,97.2,243.],
                    [40.5, 97.2, 243., 624.85714286]]

        expect2 =   [[4., 4., 5.33333333,8.],
                    [4.,5.33333333,8., 12.8],
                    [5.33333333,8.,12.8,21.33333333],
                    [8.,12.8,21.33333333,36.57142857]]

        result = cost.block_cost(times, costs, order).tocsc(copy=True)
        result1 = cost.delete(result,2,3,costs,order).copy()
        result2 = cost.delete(result,1,3,costs,order).copy()
        result3 = cost.delete(result,0,2,costs,order).copy()
        np.testing.assert_allclose(result1.todense(), expect1)
        np.testing.assert_allclose(result2.todense(), expect1)
        np.testing.assert_allclose(result3.todense(), expect2)

    def test_delete_multi_seg(self):
        costs = [1, 0, 0, 0, 0]
        times = [2,2,2]
        order = 3
        E1 = np.array([[6.0,9.0,18.0,40.5],
                    [9., 18., 40.5,97.2],
                    [18., 40.5,97.2,243.],
                    [40.5, 97.2, 243., 624.85714286]])
        E2 = np.array([[4., 4., 5.33333333,8.],
                    [4.,5.33333333,8., 12.8],
                    [5.33333333,8.,12.8,21.33333333],
                    [8.,12.8,21.33333333,36.57142857]])
        expect1 = sp.linalg.block_diag(E1,E2)
        expect2 = sp.linalg.block_diag(E2,E1)
        result = cost.block_cost(times, costs, order).tocsc(copy=True)
        result1 = cost.delete(result,1,3,costs,order).copy()
        result2 = cost.delete(result,2,3,costs,order).copy()
        np.testing.assert_allclose(result1.todense(), expect1)
        np.testing.assert_allclose(result2.todense(), expect2)

    def test_delete_then_insert(self):
        costs = [0, 0, 0, 1, 0]
        times = [2,2,2,2,2]
        order = 9
        expect = cost.block_cost(times, costs, order).tocsc(copy=True)
        result1 = cost.delete(expect,3,2,costs,order)
        result = cost.insert(result1,3,[2,2],costs,order)
        np.testing.assert_allclose(result.todense(), expect.todense())

    def test_insert_then_delete(self):
        costs = [0, 0, 0, 1, 0]
        times = [2,2,2,2,2]
        order = 9
        expect = cost.block_cost(times, costs, order).tocsc(copy=True)
        result1 = cost.insert(expect,3,[2,2],costs,order)
        result = cost.delete(result1,3,2,costs,order)
        np.testing.assert_allclose(result.todense(), expect.todense())
