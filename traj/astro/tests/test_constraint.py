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

__author__ = "Benjamin Morrell"
__email__ = "benjamin.morrell@sydney.edu.au"

# Import statements
# =================

import unittest
# from unittest.case import SkipTest
import astro
from astro import constraint
from astro import utils

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


# class Input(Base):
#
#
#     def test_cyl(self):
#         waypoints = dict()
#         waypoints['x'] = np.zeros([5,4])
#         waypoints['y'] = np.zeros([5,4])
#         waypoints['z'] = np.zeros([5,4])
#         waypoints['yaw'] = np.zeros([3,4])
#
#         waypoints['x'][0,:] = np.array([1.,0.5,-0.5,-1.0])
#         waypoints['y'][0,:] = np.array([0.5,-0.5,0.5,-0.5])
#         waypoints['z'][0,:] = np.array([-0.5,-0.5,0.5,0.5])
#
#         traj = traj_qr.traj_qr(waypoints)
#
#
#     def test_position_waypoint_only_input(self):
#         waypoints = dict()
#         waypoints['x'] = np.array([1.,0.5,-0.5,-1.0])
#         waypoints['y'] = np.array([0.5,-0.5,0.5,-0.5])
#         waypoints['z'] = np.array([-0.5,-0.5,0.5,0.5])
#         waypoints['yaw'] = np.array([0,0,0,0])
#
#         traj = traj_qr.traj_qr(waypoints)

class Computations(Base):

    def setUp(self):
        pass
        # waypoints = dict()
        # waypoints['x'] = np.zeros([5,4])
        # waypoints['y'] = np.zeros([5,4])
        # waypoints['z'] = np.zeros([5,4])
        # waypoints['yaw'] = np.zeros([3,4])
        #
        # waypoints['x'][0,:] = np.array([1.,0.5,-0.5,-1.0])
        # waypoints['y'][0,:] = np.array([0.5,-0.5,0.5,-0.5])
        # waypoints['z'][0,:] = np.array([-0.5,-0.5,0.5,0.5])
        #
        # n_wayp = waypoints['x'].size
        #
        # self.traj = traj_qr.traj_qr(waypoints)
        #
        # self.traj.run_astro()

    def test_cylinder_base_cost_grad(self):

        weight = 1.0
        x1 = np.array([-10.0,0.0,-0.])
        x2 = np.array([10.0,0.0,-0.])
        r = 1.0
        l = 0.01
        keep_out = True
        der = 0

        constr = constraint.cylinder_constraint(weight,keep_out,der,x1,x2,r,l,active_seg = 0,dynamic_weighting=False,doCurv=True,sum_func = False)

        test_store = np.array([ [0.0,0.5,0],
                                [0.0,0,0],
                                [0,0,0.5]])

        cost_expect = np.array([0.75 , 1.0, 0.75])
        grad_store = -np.array([[0,1.0,0],[0,0,0.0],[0,0,01.0]])

        for i in range(test_store.shape[0]):
            state_test = dict()
            grad_expect = dict()
            for key in ['x','y','z']:
                state_test[key] = np.zeros((1,1,1))

            grad_expect['x'] = grad_store[i,0]
            grad_expect['y'] = grad_store[i,1]
            grad_expect['z'] = grad_store[i,2]

            state_test['x'][0,:,0] = test_store[i,0]
            state_test['y'][0,:,0] = test_store[i,1]
            state_test['z'][0,:,0] = test_store[i,2]

            cost, grad, curv, max_ID = constr.cost_grad_curv(state_test, 0, doGrad=True, doCurv=True)

            np.testing.assert_allclose(cost, cost_expect[i])

            for key in ['x','y','z']:
                np.testing.assert_allclose(grad[key], grad_expect[key])

    def test_cylinder_base_cost_grad_sum(self):

        weight = 1.0
        x1 = np.array([-10.0,0.0,-0.])
        x2 = np.array([10.0,0.0,-0.])
        r = 1.0
        l = 0.01
        keep_out = True
        der = 0

        constr = constraint.cylinder_constraint(weight,keep_out,der,x1,x2,r,l,active_seg = 0,dynamic_weighting=False,doCurv=True,sum_func = True)

        state_test = dict()
        grad_expect = dict()
        for key in ['x','y','z']:
            state_test[key] = np.zeros((1,3,1))

        grad_expect['x'] = np.array([[0.,0,0.0]])
        grad_expect['y'] = np.array([[-1.,0,0.0]])
        grad_expect['z'] = np.array([[0.,0,1.]])

        state_test['x'][0,:,0] = np.array([0,0.0,0])
        state_test['y'][0,:,0] = np.array([0.5,0.0,0])
        state_test['z'][0,:,0] = np.array([0,0.0,-0.5])

        cost_expect = 2.5
        # grad_expect = np.array([[0,0.5,0],[0,0,0.0],[0,0,0.5]])

        cost, grad, curv, max_ID = constr.cost_grad_curv(state_test, 0, doGrad=True, doCurv=True)

        np.testing.assert_allclose(cost, np.max(cost_expect))

        for key in ['x','y','z']:
            np.testing.assert_allclose(grad[key], grad_expect[key])

    def test_cylinder_base_cost_grad_slant(self):

        weight = 1.0
        x1 = np.array([-10.0,-2.0,-4.])
        x2 = np.array([10.0,2.0,4.])
        r = 1.0
        l = 0.01
        keep_out = True
        der = 0

        constr = constraint.cylinder_constraint(weight,keep_out,der,x1,x2,r,l,active_seg = 0,dynamic_weighting=False,doCurv=True,sum_func = False)

        test_store = np.array([ [0.0,0.5,0],
                                [0,0,-0.5],
                                [0.1,0.4,0.7],
                                [0.5,0.1,0.2],
                                [1.5,-3.0,-4.0]])

        cost_expect = np.array([0.758333333333333 , 0.78333333333333, 0.5163333333333,1.0,0])
        grad_store = -np.array([ [-0.1666666666,0.966666666666,-0.066666666666666],
                                [0.333333333333333,0.066666666666666,-0.86666666666],
                                [-.56666666666666,0.64666666666666,1.09333333333333],
                                [0.0,0.0,0.0],
                                [0.0,0.0,0.0]])

        for i in range(test_store.shape[0]):
            state_test = dict()
            grad_expect = dict()
            for key in ['x','y','z']:
                state_test[key] = np.zeros((1,1,1))
            grad_expect['x'] = grad_store[i,0]
            grad_expect['y'] = grad_store[i,1]
            grad_expect['z'] = grad_store[i,2]

            state_test['x'][0,:,0] = test_store[i,0]
            state_test['y'][0,:,0] = test_store[i,1]
            state_test['z'][0,:,0] = test_store[i,2]

            cost, grad, curv, max_ID = constr.cost_grad_curv(state_test, 0, doGrad=True, doCurv=True)

            np.testing.assert_allclose(cost, cost_expect[i])

            for key in ['x','y','z']:
                np.testing.assert_allclose(grad[key], grad_expect[key])

    def test_cylinder_base_cost_grad_slant_keep_in(self):

        weight = 1.0
        x1 = np.array([-10.0,-2.0,-4.])
        x2 = np.array([10.0,2.0,4.])
        r = 1.0
        l = 0.01
        keep_out = False
        der = 0

        constr = constraint.cylinder_constraint(weight,keep_out,der,x1,x2,r,l,active_seg = 0,dynamic_weighting=False,doCurv=True,sum_func = False)

        test_store = np.array([     [1.5,-3.0,-4.0]])

        cost_expect = np.array([25.84166666666666])
        grad_store = np.array([ [4.166666666666,-5.766666666666,-7.533333333333]])

        for i in range(test_store.shape[0]):
            state_test = dict()
            grad_expect = dict()
            for key in ['x','y','z']:
                state_test[key] = np.zeros((1,1,1))
            grad_expect['x'] = grad_store[i,0]
            grad_expect['y'] = grad_store[i,1]
            grad_expect['z'] = grad_store[i,2]

            state_test['x'][0,:,0] = test_store[i,0]
            state_test['y'][0,:,0] = test_store[i,1]
            state_test['z'][0,:,0] = test_store[i,2]

            cost, grad, curv, max_ID = constr.cost_grad_curv(state_test, 0, doGrad=True, doCurv=True)

            np.testing.assert_allclose(cost, cost_expect[i])

            for key in ['x','y','z']:
                np.testing.assert_allclose(grad[key], grad_expect[key])

if __name__ == '__main__':
    unittest.main()
