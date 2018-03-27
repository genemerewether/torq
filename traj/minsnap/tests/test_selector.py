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
from minsnap import selector
from minsnap import utils

import numpy as np
import scipy.sparse
sp = scipy

import copy


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
        n_internal = 1
        n_der = 4
        inner = [[True] + [False] * n_internal + [True]] * n_der
        # fix 0th derivative at internal waypoints
        self.reference_der_fixed = [[True] * (n_internal + 2)]
        self.reference_der_fixed.extend(inner)

    def test_block_selector_reference(self):
        der_fixed = utils.nested_copy(self.reference_der_fixed)
        expect = selector.block_selector(der_fixed)
        np.testing.assert_array_equal(selector.block_selector(
            self.reference_der_fixed).todense(), expect.todense())


class ComputationSingleSegment(Base):

    def test_block_selector_ends_fixed(self):
        der_fixed = [[True, True],
                     [True, True],
                     [True, True],
                     [True, True],
                     [True, True]]

        expect = sp.sparse.identity(10, dtype=int, format="coo")

        np.testing.assert_array_equal(
            selector.block_selector(der_fixed).todense(), expect.todense())


class ComputationMultiSegment(Base):

    def test_block_selector_ends_fixed_middle_0th_fixed(self):
        der_fixed = [[True, True, True],
                     [True, False, True],
                     [True, False, True],
                     [True, False, True],
                     [True, False, True]]

        expect = sp.sparse.coo_matrix((np.ones(20, dtype=int),
                                       ([0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10,
                                         11, 12, 13, 14, 11, 12, 13, 14],
                                        [0, 1, 2, 3, 4, 5, 10, 15, 16, 17, 18,
                                         19, 6, 7, 8, 9, 11, 12, 13, 14])),
                                      shape=(15, 20))

        np.testing.assert_array_equal(
            selector.block_selector(der_fixed).todense(), expect.todense())

    def test_block_selector_ends_fixed_middle_0th_2nd_fixed(self):
        der_fixed = [[True, True, True],
                     [True, False, True],
                     [True, True, True],
                     [True, False, True],
                     [True, False, True]]

        expect = sp.sparse.coo_matrix((np.ones(20, dtype=int),
                                       ([0, 1, 2, 3, 4, 5, 6, 5, 6, 7, 8, 9, 10,
                                         11, 12, 13, 14, 12, 13, 14],
                                        [0, 1, 2, 3, 4, 5, 7, 10, 12, 15, 16,
                                         17, 18, 19, 6, 8, 9, 11, 13, 14])),
                                      shape=(15, 20))

        np.testing.assert_array_equal(
            selector.block_selector(der_fixed).todense(), expect.todense())
