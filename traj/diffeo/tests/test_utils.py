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
from diffeo import utils
import transforms3d
from transforms3d import euler
from transforms3d import quaternions

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


# class Input(Base):

    # def setUp(self):



class Computation(Base):
    pass
    # def setUp(self):

    def test_quat_rates(self):
        """ Test standard cases for quat rates"""
        ang_vel = np.array([0.4,0.0,0.0])
        quat = transforms3d.euler.euler2quat(0.0,0.0,0.0,'rzyx')
        expect = np.array([0,0.2,0.,0.])
        result = utils.quaternion_rates(ang_vel,quat)
        np.testing.assert_allclose(result,expect)

        ang_vel = np.array([0.0,0.8,0.0])
        expect = np.array([0.,0.,0.4,0.])
        result = utils.quaternion_rates(ang_vel,quat)
        np.testing.assert_allclose(result,expect)

        ang_vel = np.array([0.,0.0,0.6])
        expect = np.array([0,0.0,0.,0.3])
        result = utils.quaternion_rates(ang_vel,quat)
        np.testing.assert_allclose(result,expect)







if __name__ == '__main__':
    unittest.main()
