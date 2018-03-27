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

    def accel_yaw_from_attitude_quat_input(self):
        attitude = np.array([np.sqrt(0.5),0,0,np.sqrt(0.5)])
        thr_mag = 5.0
        expect_acc = [0.,0.,-4.80665]
        expect_yaw = np.pi/2
        accel_vec, yaw = accel_yaw_from_attitude(attitude,thr_mag)
        np.testing.assert_allclose(expect_acc, accel_vec)
        np.testing.assert_allclose(expect_yaw, yaw)

    def accel_yaw_from_attitude_Euler_input(self):
        quat = np.array([np.sqrt(0.5),0,0,np.sqrt(0.5)])
        yaw, roll, pitch = np.array(transforms3d.euler.quat2euler(quat,'rzyx'))
        attitude = np.array([roll, pitch,yaw])
        thr_mag = 5.0
        expect_acc = [0.,0.,-4.80665]
        expect_yaw = np.pi/2
        accel_vec, yaw = accel_yaw_from_attitude(attitude,thr_mag)
        np.testing.assert_allclose(expect_acc, accel_vec)
        np.testing.assert_allclose(expect_yaw, yaw)

    def accel_yaw_from_attitude_matrix_input(self):
        quat = np.array([np.sqrt(0.5),0,0,np.sqrt(0.5)])
        attitude = transforms3d.quaternions.quat2mat(quat)
        thr_mag = 5.0
        expect_acc = [0.,0.,-4.80665]
        expect_yaw = np.pi/2
        accel_vec, yaw = accel_yaw_from_attitude(attitude,thr_mag)
        np.testing.assert_allclose(expect_acc, accel_vec)
        np.testing.assert_allclose(expect_yaw, yaw)

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
        self.yaw = qr_p.quad_traj['yaw'].piece_poly(t)

        # accel
        self.accel = np.array([qr_p.quad_traj['x'].piece_poly.derivative().derivative()(t),qr_p.quad_traj['y'].piece_poly.derivative().derivative()(t),
                                qr_p.quad_traj['z'].piece_poly.derivative().derivative()(t)])

    def test_get_z_body(self):
        """ test z_body derivation """
        accel = np.array([0,0,1.])
        expect = np.array([0,0,1.])
        result = body_frame.get_z_body(accel)
        np.testing.assert_allclose(expect, result)

    def test_get_x_y_b(self):
        """ testing the extraction of the x and y body axes """
        z_b = np.array([0,1/np.sqrt(2),1/np.sqrt(2)])
        yaw = 0.
        expect1 = np.array([1.,0,0])
        expect2 = np.array([0,1/np.sqrt(2),-1/np.sqrt(2)])
        result1, result2 = body_frame.get_x_y_body(yaw, z_b)
        np.testing.assert_allclose(expect1, result1)
        np.testing.assert_allclose(expect2, result2)

    def test_get_x_y_body_second_angle(self):
        """ testing the extraction of the x and y body axes with second yaw angle """
        z_b = np.array([0,1/np.sqrt(2),1/np.sqrt(2)])
        yaw = 0.
        expect1 = np.array([1.,0,0])
        expect2 = np.array([0,1/np.sqrt(2),-1/np.sqrt(2)])
        result1, result2 = body_frame.get_x_y_body_second_angle(yaw, z_b)
        np.testing.assert_allclose(expect1, result1)
        np.testing.assert_allclose(expect2, result2)

    def test_get_body_frame(self):
        yaw = 0.
        accel = np.array([0.,0.,0.])
        expect_eul = np.array([0.0,0.0,0.0])
        expect_q = np.array([1.0,0.0,0.0,0.0])
        expect_R = np.identity(3)
        eul, q, R, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'all')
        np.testing.assert_allclose(expect_eul, eul)
        np.testing.assert_allclose(expect_q, q)
        np.testing.assert_allclose(expect_R, R)

    def test_get_body_frame_second_angle(self):
        yaw = 0.
        accel = np.array([0.,0.,0.])
        expect_eul = np.array([0.0,0.0,0.0])
        expect_q = np.array([1.0,0.0,0.0,0.0])
        expect_R = np.identity(3)
        eul, q, R, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'all','second_angle')
        np.testing.assert_allclose(expect_eul, eul)
        np.testing.assert_allclose(expect_q, q)
        np.testing.assert_allclose(expect_R, R)

    def test_get_body_frame_quat(self):
        yaw = 0.
        accel = np.array([1.0,2.34,5.67])
        expect_eul, expect_q, expect_R, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'all')
        eul, q, R, data = body_frame.body_frame_from_q3_and_accel(expect_q[3], accel,'all')
        # np.testing.assert_allclose(expect_eul, eul,rtol=1e-2)
        # np.testing.assert_allclose(expect_q, q,rtol=1e-2)
        # np.testing.assert_allclose(expect_R, R,rtol=1e-2)

    def test_zero_accel_get_body_frame_quat(self):
        q3 = 0.
        accel = np.array([0.,0.,0.])
        expect_eul = np.array([0.0,0.0,0.0])
        expect_q = np.array([1.0,0.0,0.0,0.0])
        expect_R = np.identity(3)
        eul, q, R, data = body_frame.body_frame_from_q3_and_accel(q3, accel,'all')
        np.testing.assert_allclose(expect_eul, eul)
        np.testing.assert_allclose(expect_q, q)
        np.testing.assert_allclose(expect_R, R)


class Singularities(Base):
        def test_x_y_body_singularity(self):
            z_b = np.array([1.0,0,0])
            yaw = np.array(0.)
            expect_x = np.array([0.,0.,-1.])
            expect_y = np.array([0.,1.,0.])
            x_b, y_b = body_frame.get_x_y_body_second_angle(yaw, z_b)
            np.testing.assert_allclose(expect_x, x_b)
            np.testing.assert_allclose(expect_y, y_b)

        def test_yaw_singularity_zero_yaw(self):
            accel = np.array([1.0,0,-settings.GRAVITY])
            yaw = np.array(0.0)
            q3 = np.sin(yaw/2)
            x_b_current = np.array([np.sqrt(0.5),0,-np.sqrt(0.5)])
            expect = np.array([[0.,0.,1.],
                               [0.,1.,0.],
                               [-1,0.,0.]])
            R1, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'matrix','yaw_only',x_b_current=x_b_current)
            R2, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'matrix','yaw_only',x_b_current=None)
            R3, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'matrix','second_angle')
            R4, data = body_frame.body_frame_from_q3_and_accel(q3, accel, 'matrix')
            np.testing.assert_allclose(expect, R1)
            np.testing.assert_allclose(expect, R2)
            # np.testing.assert_allclose(expect, np.round(R3))

        def test_yaw_singularity_180_yaw(self):
            accel = np.array([1.0,0,-settings.GRAVITY])
            yaw = np.array(np.pi)
            q3 = np.sin(yaw/2)
            x_b_current = np.array([-np.sqrt(0.5),0,np.sqrt(0.5)])
            expect = np.array([[0.,0.,1.],
                               [0.,-1.,0.],
                               [1,0.,0.]])
            R1, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'matrix','yaw_only',x_b_current=x_b_current)
            R2, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'matrix','yaw_only',x_b_current=None)
            R3, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'matrix','second_angle')
            R4, data = body_frame.body_frame_from_q3_and_accel(q3, accel, 'matrix')
            np.testing.assert_allclose(expect, R1)
            np.testing.assert_allclose(expect, R2)
            # np.testing.assert_allclose(expect, np.round(R3))

        def test_before_yaw_singularity(self):
            accel = np.array([1.0,0,0.1-settings.GRAVITY])
            yaw = np.array(0.0)
            q3 = np.sin(yaw/2)
            x_b_current = np.array([np.sqrt(0.5),0,-np.sqrt(0.5)])
            expect = np.array([[0.09950372,0.,0.99503719],
                               [0.,1.,0.],
                               [-0.99503719,0.,0.09950372]])
            R1, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'matrix','yaw_only',x_b_current=x_b_current)
            R2, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'matrix','yaw_only',x_b_current=None)
            R3, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'matrix','second_angle')
            R4, data = body_frame.body_frame_from_q3_and_accel(q3, accel, 'matrix')
            np.testing.assert_allclose(expect, R1)
            np.testing.assert_allclose(expect, R2)
            np.testing.assert_allclose(expect, R4)

        def test_after_yaw_singularity(self):
            accel = np.array([1.0,0,-0.1-settings.GRAVITY])
            yaw = np.array(0.0)
            q3 = np.sin(yaw/2)
            x_b_current = np.array([np.sqrt(0.5),0,-np.sqrt(0.5)])
            expect = np.array([[-0.09950372,0.,0.99503719],
                               [0.,1.,0.],
                               [-0.99503719,0.,-0.09950372]])
            R1, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'matrix','yaw_only',x_b_current=x_b_current)
            R2, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'matrix','yaw_only',x_b_current=None)
            R3, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'matrix','second_angle')
            R4, data = body_frame.body_frame_from_q3_and_accel(q3, accel, 'matrix')
            np.testing.assert_allclose(expect, R1)
            np.testing.assert_allclose(expect, R2)
            np.testing.assert_allclose(expect, R4)

        def test_before_yaw_singularity_180_yaw(self):
            accel = np.array([1.0,0,0.1-settings.GRAVITY])
            yaw = np.array(np.pi)
            q3 = np.sin(yaw/2)
            x_b_current = np.array([-np.sqrt(0.5),0,np.sqrt(0.5)])
            expect = np.array([[-0.09950372,0.,0.99503719],
                               [0.,-1.,0.],
                               [0.99503719,0.,0.09950372]])
            R1, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'matrix','yaw_only',x_b_current=x_b_current)
            R2, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'matrix','yaw_only',x_b_current=None)
            R3, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'matrix','second_angle')
            R4, data = body_frame.body_frame_from_q3_and_accel(q3, accel, 'matrix')
            np.testing.assert_allclose(expect, R1,atol=1e-5)
            np.testing.assert_allclose(expect, R2,atol=1e-5)
            # np.testing.assert_allclose(expect, R3,atol=1e-5)

        def test_after_yaw_singularity_180_yaw(self):
            accel = np.array([1.0,0,-0.1-settings.GRAVITY])
            yaw = np.array(np.pi)
            q3 = np.sin(yaw/2)
            x_b_current = np.array([-np.sqrt(0.5),0,np.sqrt(0.5)])
            expect = np.array([[0.09950372,0.,0.99503719],
                               [0.,-1.,0.],
                               [0.99503719,0.,-0.09950372]])
            R1, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'matrix','yaw_only',x_b_current=x_b_current)
            R2, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'matrix','yaw_only',x_b_current=None)
            R3, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'matrix','second_angle')
            R4, data = body_frame.body_frame_from_q3_and_accel(q3, accel, 'matrix')
            np.testing.assert_allclose(expect, R1,atol=1e-5)
            np.testing.assert_allclose(expect, R2,atol=1e-5)
            # np.testing.assert_allclose(expect, R3)
