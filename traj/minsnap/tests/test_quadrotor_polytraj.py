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
from minsnap import quadrotor_polytraj
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

    def test_basic_input(self):
        order=dict(x=9, y=9, z=5, yaw=5)
        temp_waypoints = utils.load_waypoints('share/sample_data/waypoints_two_ellipse.yaml')
        waypoints = utils.form_waypoints_polytraj(temp_waypoints,order)
        time_penalty = 100.0
        qr_p = quadrotor_polytraj.QRPolyTraj(waypoints, time_penalty)
        # Note that there is an issue here if order is not the default and is not input

    def test_basic_input_order(self):
        order=dict(x=9, y=9, z=5, yaw=5)
        temp_waypoints = utils.load_waypoints('share/sample_data/waypoints_two_ellipse.yaml')
        waypoints = utils.form_waypoints_polytraj(temp_waypoints,order)
        time_penalty = 100.0
        qr_p = quadrotor_polytraj.QRPolyTraj(waypoints, time_penalty, order=order)

    def test_input_restrict_freespace(self):
        # With single input l_max and A_max
        order=dict(x=9, y=9, z=5, yaw=5)
        temp_waypoints = utils.load_waypoints('share/sample_data/waypoints_two_ellipse.yaml')
        waypoints = utils.form_waypoints_polytraj(temp_waypoints,order)
        time_penalty = 100.0
        l_max = 1.0
        A_max = 10.0
        qr_p = quadrotor_polytraj.QRPolyTraj(waypoints, time_penalty, order=order,
                                            restrict_freespace=True,l_max=l_max,A_max=A_max)

    def test_input_restrict_freespace_vector_l_A(self):
        order=dict(x=9, y=9, z=9, yaw=5)
        temp_waypoints = utils.load_waypoints('share/sample_data/waypoints_two_ellipse.yaml')
        waypoints = utils.form_waypoints_polytraj(temp_waypoints,order)
        time_penalty = 100.0
        l_max = np.ones(waypoints['x'].shape[1]-1)*0.5
        l_max[0] = 3.0
        l_max[5] = 2.0
        l_max[-1] = 1.0
        A_max = np.ones(waypoints['x'].shape[1]-1)*20.0
        A_max[0] = 10.0
        A_max[-1] = 10.0
        qr_p = quadrotor_polytraj.QRPolyTraj(waypoints, time_penalty, order=order,
                                            restrict_freespace=True,l_max=l_max,A_max=A_max)

class ComputationSingleSegment(Base):

    def setUp(self):
        pass

class ComputationMultiSegment(Base):

    def setUp(self):
        order=dict(x=9, y=9, z=5, yaw=5)
        temp_waypoints = utils.load_waypoints('share/sample_data/waypoints_two_ellipse.yaml')
        waypoints = utils.form_waypoints_polytraj(temp_waypoints,order)
        time_penalty = 100.0
        self.qr_p = quadrotor_polytraj.QRPolyTraj(waypoints, time_penalty, order=order)

    def test_insert_delete(self):
        new_waypoint = dict()
        for key in self.qr_p.waypoints.keys():
            new_waypoint[key] = 1.5

        time1 = self.qr_p.times

        expect = self.qr_p.quad_traj['x'].free_ders

        for index in range(0,self.qr_p.waypoints['x'].shape[1]):

            if index == 0:
                new_times = [time1[index]/2,time1[index]/2]
            else:
                new_times = [time1[index-1]/2,time1[index-1]/2]

            self.qr_p.insert(index, new_waypoint, new_times)

            self.qr_p.delete(index, new_times[0]*2)

            result = self.qr_p.quad_traj['x'].free_ders

            time2 = self.qr_p.times
            print(index)
            np.testing.assert_allclose(result, expect)
            np.testing.assert_allclose(time1, time2)

    def test_insert_delete_end(self):
        new_waypoint = dict()
        for key in self.qr_p.waypoints.keys():
            new_waypoint[key] = 1.5

        time1 = self.qr_p.times

        expect = self.qr_p.quad_traj['x'].free_ders

        index = self.qr_p.waypoints['x'].shape[1]

        new_times = [time1[index-2]/2,time1[index-2]/2]

        self.qr_p.insert(index, new_waypoint, new_times, new_der_fixed=dict(x=True,y=True,z=True,yaw=True))

        self.qr_p.delete(index, new_times[0]*2)

        result = self.qr_p.quad_traj['x'].free_ders

        time2 = self.qr_p.times

        np.testing.assert_allclose(result, expect)
        np.testing.assert_allclose(time1, time2)
