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

# Import statements
# =================

import unittest
# from unittest.case import SkipTest
from astro import traj_qr

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

class Computations(Base):

    def testLegendreMatrix(self):
        pass
        # waypoints = dict()
        # waypoints['x'] = np.zeros([5,4])
        # waypoints['x'][0,0] = 1.0
        # waypoints['x'][0,0] = -1.0
        # waypoints['y'] = np.zeros([5,4])
        # waypoints['z'] = np.zeros([5,4])
        # waypoints['yaw'] = np.zeros([3,4])
        # traj = traj_qr.traj_qr(waypoints)
        # expect = np.array([[ 0.   ,  0.   ,  0.   ,  0.   ,  1.   ],
        #                   [ 0.   ,  0.   ,  0.   ,  1.   ,  0.   ],
        #                   [ 0.   ,  0.   ,  1.5  ,  0.   , -0.5  ],
        #                   [ 0.   ,  2.5  ,  0.   , -1.5  ,  0.   ],
        #                   [ 4.375,  0.   , -3.75 ,  0.   ,  0.375]])
        # result = traj.legendreMatrix(5)
        #
        # np.testing.assert_allclose(expect, result)
