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
from astro import traj_qr
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


class Input(Base):

    # check the size of the output?
    # Load an example from Matlab to check against??
    #TODO
    # simple example for intDerSq with low dimensions where we can compute the answer
        # Just with one dimension as well

    def test_full_waypoint_input(self):
        waypoints = dict()
        waypoints['x'] = np.zeros([5,4])
        waypoints['y'] = np.zeros([5,4])
        waypoints['z'] = np.zeros([5,4])
        waypoints['yaw'] = np.zeros([3,4])

        waypoints['x'][0,:] = np.array([1.,0.5,-0.5,-1.0])
        waypoints['y'][0,:] = np.array([0.5,-0.5,0.5,-0.5])
        waypoints['z'][0,:] = np.array([-0.5,-0.5,0.5,0.5])

        traj = traj_qr.traj_qr(waypoints)


    def test_position_waypoint_only_input(self):
        waypoints = dict()
        waypoints['x'] = np.array([1.,0.5,-0.5,-1.0])
        waypoints['y'] = np.array([0.5,-0.5,0.5,-0.5])
        waypoints['z'] = np.array([-0.5,-0.5,0.5,0.5])
        waypoints['yaw'] = np.array([0,0,0,0])

        traj = traj_qr.traj_qr(waypoints)

class Computations(Base):

    def test_Legendre_Matrix(self):
        pass

    def setUp(self):
        waypoints = dict()
        waypoints['x'] = np.zeros([5,4])
        waypoints['y'] = np.zeros([5,4])
        waypoints['z'] = np.zeros([5,4])
        waypoints['yaw'] = np.zeros([3,4])

        waypoints['x'][0,:] = np.array([1.,0.5,-0.5,-1.0])
        waypoints['y'][0,:] = np.array([0.5,-0.5,0.5,-0.5])
        waypoints['z'][0,:] = np.array([-0.5,-0.5,0.5,0.5])

        n_wayp = waypoints['x'].size

        self.traj = traj_qr.traj_qr(waypoints)

        self.traj.run_astro()

    def test_insert_delete(self):

        waypoints = self.traj.waypoints
        n_wayp = waypoints['x'].shape[1]

        new_waypoint_in = dict()
        for key in waypoints.keys():
            new_waypoint_in[key] = 2.0

        time1 = self.traj.times

        for index in range(0,n_wayp+1):

            if index == 0:
                new_times = [time1[index]/2,time1[index]/2]
            elif index == n_wayp:
                new_times = [time1[index-2]/2,time1[index-2]/2]
            else:
                new_times = [time1[index-1]/2,time1[index-1]/2]

            # index = 2
            new_waypoint = new_waypoint_in.copy()

            expect = self.traj.c_leg_poly

            self.traj.insert_waypoint(index,new_waypoint,new_times,new_der_fixed=dict(x=True, y=True, z=True,
                               yaw=True),defer=False)
            self.traj.delete_waypoint(index,new_times[0]*2,defer=False)
            result = self.traj.c_leg_poly

            for key in self.traj.c_leg_poly.keys():
                np.testing.assert_allclose(result[key], expect[key])

    def test_P_BC(self):
        traj = self.traj
        traj.initial_guess

        for key in traj.c_leg_poly.keys():
            result = traj.P_BC_C[key]*np.matrix(traj.c_leg_poly[key]).T
            expect = np.matrix(traj.bc[key]).T
            np.testing.assert_allclose(result, expect,rtol=1e-7,atol=1e-7)

    def test_rescale_P_BC(self):
        traj = self.traj
        traj.initial_guess()

        indices = np.atleast_1d([0,1,2])

        old_times = traj.times
        traj.times = old_times/2

        expect = traj.c_leg_poly
        traj.rescale_P_BC(indices,old_times)
        traj.initial_guess()

        old_times = traj.times
        traj.times = old_times*2

        traj.rescale_P_BC(indices,old_times)
        traj.initial_guess()

        result = traj.c_leg_poly

        for key in traj.c_leg_poly.keys():
            np.testing.assert_allclose(result[key], expect[key])

    def test_update_times_single(self):
        traj = self.traj

        for k in range(0,3):
            indices = np.atleast_1d([k])
            old_times = traj.times.copy()
            new_times = old_times[indices]/2
            traj.update_times(indices, new_times, new_coeffs=False, defer=True)

            # state: dict for each dim, 3D np.array([n_der,n_samp,n_seg]). Stores all timesteps for each dimension for each segment
            for key in traj.state.keys():
                # Check boundary conditions
                expect_bc = traj.bc[key]
                result_bc = np.round(traj.P_BC_C[key].dot(traj.c_leg_poly[key]),4)
                np.testing.assert_allclose(result_bc, expect_bc)

                for i in range(0,old_times.size-1):
                    # Check internal waypoints for continuit
                    expect_c = traj.state[key][:,-1,i]
                    result_c = traj.state[key][:,0,i+1]
                    np.testing.assert_allclose(result_c, expect_c)

    def test_update_times_multiple(self):
        traj = self.traj

        indices = np.atleast_1d([0,2])
        old_times = traj.times.copy()
        new_times = old_times[indices]/2
        traj.update_times(indices, new_times, new_coeffs=False, defer=True)

        # state: dict for each dim, 3D np.array([n_der,n_samp,n_seg]). Stores all timesteps for each dimension for each segment
        for key in traj.state.keys():
            # Check boundary conditions
            expect_bc = traj.bc[key]
            result_bc = np.round(traj.P_BC_C[key].dot(traj.c_leg_poly[key]),4)
            np.testing.assert_allclose(result_bc, expect_bc)

            for i in range(0,old_times.size-1):
                # Check internal waypoints for continuit
                expect_c = traj.state[key][:,-1,i]
                result_c = traj.state[key][:,0,i+1]
                np.testing.assert_allclose(result_c, expect_c)

    def test_update_times_all(self):
        traj = self.traj

        indices = np.atleast_1d([0,1,2])
        old_times = traj.times.copy()
        new_times = old_times[indices]/2
        traj.update_times(indices, new_times, new_coeffs=False, defer=True)

        # state: dict for each dim, 3D np.array([n_der,n_samp,n_seg]). Stores all timesteps for each dimension for each segment
        for key in traj.state.keys():
            # Check boundary conditions
            expect_bc = traj.bc[key]
            result_bc = np.round(traj.P_BC_C[key].dot(traj.c_leg_poly[key]),4)
            np.testing.assert_allclose(result_bc, expect_bc)

            for i in range(0,old_times.size-1):
                # Check internal waypoints for continuit
                expect_c = traj.state[key][:,-1,i]
                result_c = traj.state[key][:,0,i+1]
                np.testing.assert_allclose(result_c, expect_c)

    def test_time_cost_no_modification_of_traj(self):
        traj = self.traj

        indices = np.atleast_1d([0, 2])#,1,2])
        old_times = traj.times.copy()
        new_times = old_times[indices]/2

        # Starting values:
        P_BC = traj.P_BC_C.copy()
        proj_mat = traj.proj_mat.copy()
        cleg = traj.c_leg_poly.copy()
        state = traj.poly.state_scaled.copy()

        # Compute cost:
        cost=traj.time_opt_cost(indices,new_times)

        # output
        P_BC2 = traj.P_BC_C.copy()
        proj_mat2 = traj.proj_mat.copy()
        cleg2 = traj.c_leg_poly.copy()
        state2 = traj.poly.state_scaled.copy()

        for key in cleg.keys():
            np.testing.assert_allclose(P_BC[key],P_BC2[key])
            np.testing.assert_allclose(proj_mat[key],proj_mat2[key])
            np.testing.assert_allclose(cleg[key],cleg[key])
            np.testing.assert_allclose(state[key],state2[key])

    def test_time_gradient_no_modification_of_traj(self):
        traj = self.traj

        indices = np.atleast_1d([0, 2])#,1,2])
        old_times = traj.times.copy()
        new_times = old_times[indices]/2

        # Starting values:
        P_BC = traj.P_BC_C.copy()
        proj_mat = traj.proj_mat.copy()
        cleg = traj.c_leg_poly.copy()
        state = traj.poly.state_scaled.copy()

        # Compute gradient:
        cost=traj.time_opt_gradient(traj.times)

        # output
        P_BC2 = traj.P_BC_C.copy()
        proj_mat2 = traj.proj_mat.copy()
        cleg2 = traj.c_leg_poly.copy()
        state2 = traj.poly.state_scaled.copy()

        for key in cleg.keys():
            np.testing.assert_allclose(P_BC[key],P_BC2[key])
            np.testing.assert_allclose(proj_mat[key],proj_mat2[key])
            np.testing.assert_allclose(cleg[key],cleg[key])
            np.testing.assert_allclose(state[key],state2[key])

    def test_simple_basis_converstion(self):
        traj = self.traj

        traj.get_trajectory()

        new_coeffs = traj.convert_poly_to_simple_basis()

        trans_times = utils.seg_times_to_trans_times(traj.times)

        ppoly_new = dict()
        new_states = dict()

        t = np.array([])
        for i in range(traj.n_seg):
            t = np.concatenate([t,np.linspace(trans_times[i], trans_times[i+1], traj.n_samp)])

        for key in traj.c_leg_poly.keys():
            ppoly_new[key] = sp.interpolate.PPoly(new_coeffs[key],trans_times, extrapolate=False)
            new_states[key] = ppoly_new[key](t)
            for k in range(traj.state[key].shape[2]):
                np.testing.assert_allclose(traj.state[key][0,:,k],new_states[key][traj.n_samp*k:traj.n_samp*(k+1)])

class largeTrajInput(Base):

    def setUp(self):

        filename = '/home/torq-dev/catkin_ws/src/torq_gcs/trajectories/344_10_26/344_wayp_red_002.yaml'
        in_waypoints = utils.load_waypoints(filename)

        n_wayp = np.size(in_waypoints['x'])

        """ Default waypoints"""
        waypoints = dict()
        waypoints['x'] = np.zeros([5,n_wayp])
        waypoints['y'] = np.zeros([5,n_wayp])
        waypoints['z'] = np.zeros([5,n_wayp])
        waypoints['yaw'] = np.zeros([3,n_wayp])

        waypoints['x'][0,:] = in_waypoints['x']
        waypoints['y'][0,:] = in_waypoints['y']
        waypoints['z'][0,:] = in_waypoints['z']

        der_fixed = None

        costs = dict()
        costs['x'] = [0, 0, 0, 0, 1]  # minimum snap
        costs['y'] = [0, 0, 0, 0, 1]  # minimum snap
        costs['z'] = [0, 0, 0, 0, 1]  # minimum snap
        costs['yaw'] = [0, 0, 1]  # minimum acceleration
        order=dict(x=9, y=9, z=9, yaw=5)
        # order=dict(x=12, y=12, z=12, yaw=5)
        seed_times = np.ones(waypoints['x'].shape[1]-1)*1.0

        self.traj = traj_qr.traj_qr(waypoints,
                                costs=costs,
                                order=order,
                                seed_times=seed_times,
                                curv_func=False,
                                der_fixed=der_fixed,
                                path_weight=None)

        self.traj.run_astro()

    def test_P_BC_C_large_traj(self):
        traj = self.traj
        traj.initial_guess()

        for key in traj.c_leg_poly.keys():
            result = traj.P_BC_C[key]*np.matrix(traj.c_leg_poly[key]).T
            expect = np.matrix(traj.bc[key]).T
            np.testing.assert_allclose(result, expect,rtol=1e-7,atol=1e-7)


if __name__ == '__main__':
    unittest.main()
