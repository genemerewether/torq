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
from minsnap import utils

import numpy as np


class Base(unittest.TestCase):

    """
    Base class for tests

    This class defines a common `setUp` method that defines attributes which are used in the various tests.
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
        # TODO(mereweth@jpl.nasa.gov) check each bad argument in isolation
        self.nan_order = np.nan
        self.negative_order = -1
        self.float_order = 9.5

        # TODO(mereweth@jpl.nasa.gov) tests for to_dup None, etc

    def test_dup_internal_1d_list_columns(self):
        """Test duplication function with 1d plain list as input (not numpy)"""
        to_dup = [0, 1, 2]
        expect = np.array([[0, 1, 1, 2]])
        np.testing.assert_equal(utils.dup_internal(to_dup, 1), expect)

    def test_dup_internal_1d_list_rows(self):
        """Test duplication function with 1d plain list as input (not numpy)"""
        to_dup = [0, 1, 2]
        expect = np.array([[0],
                           [1],
                           [1],
                           [2]])
        np.testing.assert_equal(utils.dup_internal(to_dup), expect)

    def test_n_coeffs_free_ders_nan(self):
        np.testing.assert_equal(utils.n_coeffs_free_ders(self.nan_order),
                                (0, 0))

    def test_n_coeffs_free_ders_negative(self):
        np.testing.assert_equal(utils.n_coeffs_free_ders(self.negative_order),
                                (0, 0))

    def test_n_coeffs_free_ders_float(self):
        np.testing.assert_equal(utils.n_coeffs_free_ders(self.float_order),
                                (10, 5))


class Computation(Base):

    def test_dup_internal_multi_segment_columns(self):
        """Test duplication function with multiple internal columns"""
        to_dup = np.array([[0, 1, 2, 3],
                           [4, 5, 6, 7]])
        expect = np.array([[0, 1, 1, 2, 2, 3],
                           [4, 5, 5, 6, 6, 7]])
        result = utils.dup_internal(to_dup, 1)
        np.testing.assert_equal(result, expect)

    def test_dup_internal_multi_segment_columns_more_than_two(self):
        """Test duplication function with multiple internal columns"""
        to_dup = np.array([[0,  1,  2,  3,  4,  5,  6],
                           [7,  8,  9, 10, 11, 12, 13],
                           [14, 15, 16, 17, 18, 19, 20],
                           [21, 22, 23, 24, 25, 26, 27],
                           [28, 29, 30, 31, 32, 33, 34]])
        expect = np.array([[0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6],
                           [7,  8,  8,  9,  9, 10, 10, 11, 11, 12, 12, 13],
                           [14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20],
                           [21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27],
                           [28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34]])
        result = utils.dup_internal(to_dup, 1)
        np.testing.assert_equal(result, expect)

    def test_dup_internal_multi_segment_rows(self):
        """Test duplication function with multiple internal rows"""
        to_dup = np.array([[0, 4],
                           [1, 5],
                           [2, 6],
                           [3, 7]])
        expect = np.array([[0, 4],
                           [1, 5],
                           [1, 5],
                           [2, 6],
                           [2, 6],
                           [3, 7]])
        result = utils.dup_internal(to_dup)
        np.testing.assert_equal(result, expect)

    def test_n_coeffs_free_ders_odd(self):
        """Test # coeffs and free ders with odd order"""
        np.testing.assert_equal(utils.n_coeffs_free_ders(9), (10, 5))

    def test_n_coeffs_free_ders_even(self):
        """Test # coeffs and free ders with even order"""
        np.testing.assert_equal(utils.n_coeffs_free_ders(10), (11, 5))

    def test_ltri_mgrid_odd(self):
        """Test tri mesh grid with odd order"""
        order = 9
        expect = np.array([[0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0],
                           [2, 1, 0, 0, 0],
                           [3, 2, 1, 0, 0],
                           [4, 3, 2, 1, 0],
                           [5, 4, 3, 2, 1],
                           [6, 5, 4, 3, 2],
                           [7, 6, 5, 4, 3],
                           [8, 7, 6, 5, 4],
                           [9, 8, 7, 6, 5]])

        result = utils.ltri_mgrid(order)
        np.testing.assert_array_equal(result, expect)

    def test_ltri_mgrid_even(self):
        """Test tri mesh grid with even order"""
        order = 8
        expect = np.array([[0, 0, 0, 0],
                           [1, 0, 0, 0],
                           [2, 1, 0, 0],
                           [3, 2, 1, 0],
                           [4, 3, 2, 1],
                           [5, 4, 3, 2],
                           [6, 5, 4, 3],
                           [7, 6, 5, 4],
                           [8, 7, 6, 5]])

        result = utils.ltri_mgrid(order)
        np.testing.assert_array_equal(result, expect)

    def test_ltri_mgrid_trunc(self):
        """Test tri mesh grid with max width"""
        order = 9
        max_der = 2
        expect = np.array([[0, 0],
                           [1, 0],
                           [2, 1],
                           [3, 2],
                           [4, 3],
                           [5, 4],
                           [6, 5],
                           [7, 6],
                           [8, 7],
                           [9, 8]])

        result = utils.ltri_mgrid(order, max_der)
        np.testing.assert_array_equal(result, expect)

    def test_poly_der_coeffs_odd(self):
        """Test product of decreasing exponents with odd order"""
        order = 9

        # This function also returns lti_mgrid, which is used internally
        expect = (np.array([[0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0],
                            [2, 1, 0, 0, 0],
                            [3, 2, 1, 0, 0],
                            [4, 3, 2, 1, 0],
                            [5, 4, 3, 2, 1],
                            [6, 5, 4, 3, 2],
                            [7, 6, 5, 4, 3],
                            [8, 7, 6, 5, 4],
                            [9, 8, 7, 6, 5]]),
                  np.array([[1,    1,    1,    1,    1],
                            [1,    1,    1,    1,    1],
                            [1,    2,    2,    2,    2],
                            [1,    3,    6,    6,    6],
                            [1,    4,   12,   24,   24],
                            [1,    5,   20,   60,  120],
                            [1,    6,   30,  120,  360],
                            [1,    7,   42,  210,  840],
                            [1,    8,   56,  336, 1680],
                            [1,    9,   72,  504, 3024]]))

        result = utils.poly_der_coeffs(order)
        np.testing.assert_array_equal(result, expect)

    def test_poly_der_coeffs_even(self):
        """Test product of decreasing exponents with even order"""
        order = 8

        # This function also returns lti_mgrid, which is used internally
        expect = (np.array([[0, 0, 0, 0],
                            [1, 0, 0, 0],
                            [2, 1, 0, 0],
                            [3, 2, 1, 0],
                            [4, 3, 2, 1],
                            [5, 4, 3, 2],
                            [6, 5, 4, 3],
                            [7, 6, 5, 4],
                            [8, 7, 6, 5]]),
                  np.array([[1,    1,    1,    1],
                            [1,    1,    1,    1],
                            [1,    2,    2,    2],
                            [1,    3,    6,    6],
                            [1,    4,   12,   24],
                            [1,    5,   20,   60],
                            [1,    6,   30,  120],
                            [1,    7,   42,  210],
                            [1,    8,   56,  336]]))

        result = utils.poly_der_coeffs(order)
        np.testing.assert_array_equal(result, expect)

    def test_poly_hessian_coeffs_odd_odd(self):
        """
        Test Hessian coefficient factors with odd polynomial order and odd
            derivative penalty
        """
        order = 7
        der_order = 3  # 0th derivative is zero order

        # This function also returns lti_mgrid, which is used internally
        expect = (np.array([[0, 0, 0],
                            [1, 0, 0],
                            [2, 1, 0],
                            [3, 2, 1],
                            [4, 3, 2],
                            [5, 4, 3],
                            [6, 5, 4],
                            [7, 6, 5]]),
                  np.array([[0, 0, 0,    0,     0,     0,     0,      0],
                            [0, 0, 0,    0,     0,     0,     0,      0],
                            [0, 0, 0,    0,     0,     0,     0,      0],
                            [0, 0, 0,   36,   144,   360,   720,   1260],
                            [0, 0, 0,  144,   576,  1440,  2880,   5040],
                            [0, 0, 0,  360,  1440,  3600,  7200,  12600],
                            [0, 0, 0,  720,  2880,  7200, 14400,  25200],
                            [0, 0, 0, 1260,  5040, 12600, 25200,  44100]]))

        result = utils.poly_hessian_coeffs(order, der_order)
        assert(len(expect) == len(result))
        for i in range(0, len(expect)):
            np.testing.assert_array_equal(result[i], expect[i])


if __name__ == '__main__':
    unittest.main()
