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
from diffeo import body_frame
from diffeo import angular_rates_accel
from minsnap import utils
from minsnap import quadrotor_polytraj

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

class Computations(Base):

    def setUp(self):
        waypoints = dict(x= np.array([0.0, 1.0, -1.0, 1.0, -1.0, 0.0]),
                         y= np.array([0.0, 1.0, 1.0, -1.0, -1.0, 0.0]),
                         z= np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                         yaw=np.array([ 0.,  0.,  0.,  0.,  0.,  0.]))
        qr_p = quadrotor_polytraj.QRPolyTraj(waypoints, 5000)

        # Compute times
        trans_times = utils.seg_times_to_trans_times(qr_p.times)
        t1 = np.linspace(trans_times[0], trans_times[-1], 100)

        # To test single input:
        t = t1[4]

        # Yaw
        yaw = qr_p.quad_traj['yaw'].piece_poly(t)
        self.yaw_dot = qr_p.quad_traj['yaw'].piece_poly.derivative()(t)
        self.yaw_ddot = qr_p.quad_traj['yaw'].piece_poly.derivative().derivative()(t)

        # accel
        accel = np.array([qr_p.quad_traj['x'].piece_poly.derivative().derivative()(t),
                            qr_p.quad_traj['y'].piece_poly.derivative().derivative()(t),
                            qr_p.quad_traj['z'].piece_poly.derivative().derivative()(t)])

        # jerk
        self.jerk = np.array([qr_p.quad_traj['x'].piece_poly.derivative().derivative().derivative()(t),
                         qr_p.quad_traj['y'].piece_poly.derivative().derivative().derivative()(t),
                         qr_p.quad_traj['z'].piece_poly.derivative().derivative().derivative()(t)])


        # snap
        self.snap = np.array([qr_p.quad_traj['x'].piece_poly.derivative().derivative().derivative().derivative()(t),
                         qr_p.quad_traj['y'].piece_poly.derivative().derivative().derivative().derivative()(t),
                         qr_p.quad_traj['z'].piece_poly.derivative().derivative().derivative().derivative()(t)])

        # Get rotation matrix
        euler, self.quat, self.R, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'all')

        # Thrust
        thrust, self.thrust_mag = body_frame.get_thrust(accel)

    def test_get_angular_vel(self):
        #TODO(bmorrell@jpl.nasa.gov) come up with a test case to verify the output
        ang_vel = angular_rates_accel.get_angular_vel(self.thrust_mag,self.jerk,self.R,self.yaw_dot)

    def test_get_angular_accel(self):
        #TODO(bmorrell@jpl.nasa.gov) come up with a test case to verify the output
        ang_vel = angular_rates_accel.get_angular_vel(self.thrust_mag,self.jerk,self.R,self.yaw_dot)
        ang_accel = angular_rates_accel.get_angular_accel(self.thrust_mag,self.jerk,self.snap,self.R,ang_vel,self.yaw_ddot)

    def test_get_angular_vel_quat(self):
        # Use the Euler approach
        ang_vel_1 = angular_rates_accel.get_angular_vel(self.thrust_mag,self.jerk,self.R,self.yaw_dot)
        # convert to quaternions
        from diffeo import utils
        quat_dot = utils.quaternion_rates(ang_vel_1,self.quat)
        ang_vel_2 = angular_rates_accel.get_angular_vel_quat(self.thrust_mag,self.jerk,self.R,quat_dot[3],self.quat)
        np.testing.assert_allclose(ang_vel_2,ang_vel_1)

    def test_get_angular_accel_quat(self):
        # Use the Euler approach
        ang_vel_1 = angular_rates_accel.get_angular_vel(self.thrust_mag,self.jerk,self.R,self.yaw_dot)
        ang_accel_1 = angular_rates_accel.get_angular_accel(self.thrust_mag,self.jerk,self.snap,self.R,ang_vel_1,self.yaw_ddot)
        # convert to quaternions
        from diffeo import utils
        quat_dot = utils.quaternion_rates(ang_vel_1,self.quat)
        quat_ddot = utils.quaternion_accel(ang_accel_1,self.quat,quat_dot)
        ang_vel_2 = angular_rates_accel.get_angular_vel_quat(self.thrust_mag,self.jerk,self.R,quat_dot[3],self.quat)
        ang_accel_2 = angular_rates_accel.get_angular_accel_quat(self.thrust_mag,self.jerk,self.snap,self.R,ang_vel_1,quat_ddot[3],self.quat)

        np.testing.assert_allclose(ang_accel_2,ang_accel_1)
