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
from minsnap import joint_optimize
from minsnap import utils

import numpy as np
import scipy.sparse
sp = scipy

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
        pass

class ComputationSingleSegment(Base):

    def setUp(self):
        order = 9
        n_der = utils.n_coeffs_free_ders(order)[1]
        costs = [0, 0, 0, 0, 1]  # minimum snap
        waypoints = [[0, 1]]
        waypoints.extend([[0] * len(waypoints[0])] * (n_der - 1))

        # fix all derivatives at beginning and end
        der_fixed = [[True] * 2] * n_der

        self.zero_one_traj = joint_optimize.PolyTraj(waypoints, order, costs, der_fixed)

    def test_get_poly_coeffs_zero_one(self):
        """Test solving for polynomial coeffs with waypoints at zero and one"""

        times = np.ones(self.zero_one_traj.n_seg)
        expect = np.asmatrix([70, -315, 540, -420, 126,
                              0, 0, 0, 0, 0]).T

        self.zero_one_traj.update_times(times)
        result = self.zero_one_traj.coeffs

        np.testing.assert_allclose(expect, result)
        assert(self.zero_one_traj.check_continuity())

class ComputationMultiSegment(Base):

    def setUp(self):
        order = 9
        n_der = utils.n_coeffs_free_ders(order)[1]
        costs = [0, 0, 0, 0, 1]  # minimum snap
        waypoints = [[0, 1, -1, 1, -1, 0]]
        waypoints.extend([[0] * len(waypoints[0])] * (n_der - 1))

        num_internal = len(waypoints[0]) - 2
        # float derivatives at internal waypoints and fix at beginning and end
        inner = [[True] + [False] * num_internal + [True]] * (n_der - 1)
        # fix 0th derivative at internal waypoints
        der_fixed = [[True] * (num_internal + 2)]
        der_fixed.extend(inner)

        times = np.ones(np.shape(waypoints)[1] - 1)
        self.plus_minus_traj = joint_optimize.PolyTraj(waypoints, order, costs,
                                                       der_fixed, times=times)

        mod_waypoints = np.array(waypoints).copy()
        mod_waypoints[0][0] = -1
        self.to_update_waypoints_traj = joint_optimize.PolyTraj(mod_waypoints, order,
                                                                costs, der_fixed,
                                                                times=times)

        mod_times = np.array(times).copy()
        mod_times[0] = 10
        mod_times[-1] = 10
        self.to_update_times_traj = joint_optimize.PolyTraj(waypoints, order,
                                                            costs, der_fixed,
                                                            times=mod_times)

        self.expect_plus_minus_traj_free_ders = np.asmatrix([0.577012360703533, -8.36470699594831,
                                                             -18.740356378657, 144.225255378299,
                                                             -0.34866334176225, 11.2028742718137,
                                                             4.3364914077213, -131.839671835648,
                                                             -0.348663340762795, -11.202874272838,
                                                             4.33649140009847, 131.839671864721,
                                                             0.577012360355965, 8.3647069971665,
                                                             -18.7403563792646, -144.225255396109]).T

    def test_get_free_ders_plus_minus(self):
        """Test solving for free derivatives with alternate plus/minus waypoints"""

        expect = self.expect_plus_minus_traj_free_ders

        result = self.plus_minus_traj.free_ders

        np.testing.assert_allclose(expect, result)
        assert(self.plus_minus_traj.check_continuity())

    def test_update_waypoint_plus_minus(self):
        """Test updating waypoint with alternate plus/minus waypoints"""
        self.to_update_waypoints_traj.update_waypoint(0, [0, 0, 0, 0, 0])

        # this is included in the equality check below; just for debugging
        #expect = self.expect_plus_minus_traj_free_ders
        #result = self.to_update_waypoints_traj.free_ders
        #np.testing.assert_allclose(expect, result)

        expect = self.plus_minus_traj
        result = self.to_update_waypoints_traj

        assert(expect == result)
        assert(result.check_continuity())

    def test_update_times_plus_minus(self):
        """Test updating times with alternate plus/minus waypoints"""

        times = self.to_update_times_traj.times
        times[0] = 1
        times[-1] = 1
        times_changed = np.array([0, len(times)-1])
        self.to_update_times_traj.update_times(times, times_changed=times_changed)

        # this is included in the equality check below; just for debugging
        expect = self.expect_plus_minus_traj_free_ders
        result = self.to_update_times_traj.free_ders
        #import pdb; pdb.set_trace()
        np.testing.assert_allclose(expect, result)

        expect = self.plus_minus_traj
        result = self.to_update_times_traj

        assert(expect == result)
        assert(result.check_continuity())

    def test_insert_delete(self):

        # Create polynomial
        pol = self.plus_minus_traj

        for index in range(0,pol.waypoints.shape[1]):
            new_index = index
            new_waypoint = pol.waypoints[:,new_index].copy()
            new_waypoint[0] = 1.5
            if new_index == pol.waypoints.shape[1]-1:
                t = pol.times[new_index-1]
            else:
                t = pol.times[new_index]
            new_times = [t/2,t/2]
            new_der_fixed = pol.der_fixed[:,new_index].copy()

            expect = pol.free_ders.copy()
            pol.insert(new_index, new_waypoint, new_times, new_der_fixed, defer=False)
            pol.delete(new_index, t, defer=False)
            result = pol.free_ders.copy()
            np.testing.assert_allclose(result, expect)

    def test_insert_delete_end(self):
        # Create polynomial
        pol = self.plus_minus_traj

        new_index = pol.waypoints.shape[1]
        new_waypoint = pol.waypoints[:,new_index-1].copy()
        new_waypoint[0] = 1.5
        t = pol.times[new_index-2]
        new_times = [t/2,t/2]
        new_der_fixed = pol.der_fixed[:,new_index-1].copy()

        expect = pol.free_ders.copy()
        pol.insert(new_index, new_waypoint, new_times, new_der_fixed, defer=False)
        pol.delete(new_index, t, defer=False)
        result = pol.free_ders.copy()
        np.testing.assert_allclose(result, expect)
