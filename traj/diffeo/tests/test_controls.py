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
from diffeo import controls
from diffeo import settings
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

    def test_load_params(self):
        params = controls.load_params("TestingCode/test_load_params.yaml")

class Computations(Base):

    def setUp(self):
        waypoints = dict(x= np.array([0.0, 1.0, -1.0, 1.0, -1.0, 0.0]),
                         y= np.array([0.0, 1.0, 1.0, -1.0, -1.0, 0.0]),
                         z= np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                         yaw=np.array([ 0.,  0.,  0.,  0.,  0.,  0.]))
        qr_p = quadrotor_polytraj.QRPolyTraj(waypoints, 5000)

        self.params = dict()
        self.params['mass'] = 0.48446 # kg
        Lxx = 1131729.47
        Lxy = -729.36
        Lxz = -5473.45
        Lyx = -729.36
        Lyy = 1862761.14
        Lyz = -2056.74
        Lzx = -5473.45
        Lzy = -2056.74
        Lzz = 2622183.34

        unit_conv = 1*10**-9
        self.params['Inertia'] = np.array([[Lxx,Lxy,Lxz],[Lyx,Lyy,Lyz],[Lzx,Lzy,Lzz]])*unit_conv

        # Thrust coefficeint
        self.params['k1'] = 2.25*10**-6 # dow 5045 x3 bullnose prop
        self.params['k2'] = 0.0 # dow 5045 x3 bullnose prop
        self.params['k4'] = 0.0# dow 5045 x3 bullnose prop
        self.params['Cq'] = self.params['k1']*10**-2
        self.params['Dx'] = 0.088 # meters
        self.params['Dy'] = 0.088 # meters

        # Compute times
        trans_times = utils.seg_times_to_trans_times(qr_p.times)
        t1 = np.linspace(trans_times[0], trans_times[-1], 100)

        # To test single input:
        t = t1[4]


        # Yaw
        yaw = qr_p.quad_traj['yaw'].piece_poly(t)
        yaw_dot = qr_p.quad_traj['yaw'].piece_poly.derivative()(t)
        yaw_ddot = qr_p.quad_traj['yaw'].piece_poly.derivative().derivative()(t)

        # accel
        accel = np.array([qr_p.quad_traj['x'].piece_poly.derivative().derivative()(t),
                            qr_p.quad_traj['y'].piece_poly.derivative().derivative()(t),
                            qr_p.quad_traj['z'].piece_poly.derivative().derivative()(t)])

        # jerk
        jerk = np.array([qr_p.quad_traj['x'].piece_poly.derivative().derivative().derivative()(t),
                         qr_p.quad_traj['y'].piece_poly.derivative().derivative().derivative()(t),
                         qr_p.quad_traj['z'].piece_poly.derivative().derivative().derivative()(t)])


        # snap
        snap = np.array([qr_p.quad_traj['x'].piece_poly.derivative().derivative().derivative().derivative()(t),
                         qr_p.quad_traj['y'].piece_poly.derivative().derivative().derivative().derivative()(t),
                         qr_p.quad_traj['z'].piece_poly.derivative().derivative().derivative().derivative()(t)])

        # Get rotation matrix
        R, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'matrix')

        # Thrust
        thrust, self.thrust_mag = body_frame.get_thrust(accel)

        # Angular rates
        self.ang_vel = angular_rates_accel.get_angular_vel(self.thrust_mag,jerk,R,yaw_dot)

        # Angular accelerations
        self.ang_accel = angular_rates_accel.get_angular_accel(self.thrust_mag,jerk,snap,R,self.ang_vel,yaw_ddot)

    def test_get_torques(self):
        #TODO(bmorrell@jpl.nasa.gov) come up with a test case to verify tye output
        torques = controls.get_torques(self.ang_vel, self.ang_accel, self.params)

    def test_get_rpm(self):
        #TODO(bmorrell@jpl.nasa.gov) come up with a test case to verify tye output
        torques = controls.get_torques(self.ang_vel, self.ang_accel, self.params)
        # Get rotor speeds
        rpm = controls.get_rotor_speeds(torques,self.thrust_mag*self.params['mass'],self.params)


    def test_sequence(self):

        params = controls.load_params("TestingCode/test_load_params.yaml")

        params['k1'] = params['Cq']*1e2

        accel = np.array([1,2,3])
        jerk = np.array([0.3,0.1,0.4])
        snap = np.array([0.01,0.02,0.05])

        yaw = np.pi/4
        yaw_dot = 0.01
        yaw_ddot = 0.001

        # Get rotation matrix
        R, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'matrix')

        # Thrust
        thrust, thrust_mag = body_frame.get_thrust(accel)
        thrust[2] = thrust[2] - settings.GRAVITY + 9.81
        thrust_mag = np.linalg.norm(thrust)

        # Angular rates
        ang_vel = angular_rates_accel.get_angular_vel(thrust_mag,jerk,R,yaw_dot)

        # Angular accelerations
        ang_accel = angular_rates_accel.get_angular_accel(thrust_mag,jerk,snap,R,ang_vel,yaw_ddot)

        # torques
        torques = controls.get_torques(ang_vel, ang_accel, params)

        # Get rotor speeds
        rpm = controls.get_rotor_speeds(torques,thrust_mag*params['mass'],params)

        expect_ang_vel = np.array([0.01270282,0.01643325,0.009851])
        expect_ang_acc = np.array([-0.00154177,0.00060613,0.0009851])
        expect_torques = np.array([-4.30155261e-6,2.04911597e-6,6.11937597e-6])
        expect_rpm = np.array([8794.2645461,8793.64597287,8794.3097302,8793.74083079])


        np.testing.assert_allclose(expect_ang_vel, ang_vel,rtol=1e-3)
        np.testing.assert_allclose(expect_ang_acc, ang_accel,rtol=1e-3)
        np.testing.assert_allclose(expect_torques, torques,rtol=1e-3)
        np.testing.assert_allclose(expect_rpm, rpm,rtol=1e-5)
