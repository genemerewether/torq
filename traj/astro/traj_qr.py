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

import numpy as np
import scipy as sp
#from scipy.linalg import block_diag

import time
import abc

from astro import trajectory
from astro import poly_astro
from astro import constraint
from astro import data_track
from astro import utils
from trajectory import trajectoryBase
from poly_astro import poly_astro
from data_track import data_track

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

from minsnap import exceptions

#
# from utils import *

class traj_qr(trajectoryBase):

    def __init__(self,waypoints,costs=None, order=dict(x=9, y=9, z=9, yaw=5),
                der_fixed=None, seed_times=None, seed_avg_vel=0.2, path_weight=None,
                curv_func=False,n_samp = 100,yaw_to_traj=True,exit_on_feasible=False):
        """

        """

        # Initialise parent case class
        super(traj_qr,self).__init__(n_samp=n_samp,curv_func=curv_func)

        self.order = order

        self.time_penalty = 1e5
        self.del_t = 1e-3
        self.exit_tol_time = 1e-6
        self.max_viol = 0.0

        self.feasible = False
        self.exit_on_feasible = exit_on_feasible
        self.esdf_feasibility_check = False

        self.c_leg_poly = None

        self.temp_state = None

        self.optimise_yaw_only = False
        self.yaw_to_traj = yaw_to_traj

        self.fig = None
        self.ax = None

        """ Costs """
        if costs is None:
            costs = dict()
            costs['x'] = np.array([0, 0, 0, 0, 1])  # minimum snap
            costs['y'] = np.array([0, 0, 0, 0, 1])  # minimum snap
            costs['z'] = np.array([0, 0, 0, 0, 1])  # minimum snap
            costs['yaw'] = np.array([0, 0, 1])  # minimum acceleration
        else:
            # Loop for each dimension
            for key in costs.keys():
                costs[key] = np.array(costs[key])
        self.costs = costs

        """ Number of deriviatives """
        self.n_der = utils.n_der_from_costs(costs)

        """ Waypoints """
        # for x, we only received positions
        if len(np.shape(waypoints['x'])) == 0:
            raise exceptions.InputError(waypoints['x'],
                                        "Invalid waypoint array")
        elif len(np.shape(waypoints['x'])) > 2:
            raise exceptions.InputError(waypoints['x'],
                                        "Invalid waypoint array")

        self.waypoints = utils.form_waypoints_polytraj(waypoints,self.n_der)
        self.n_seg = np.shape(self.waypoints['x'])[1] - 1

        """ Fixed Derivatives """
        if der_fixed is None:
            der_fixed = dict()
            num_internal = self.n_seg - 1
            der_fixed = utils.default_der_fixed_polytraj(num_internal,self.n_der)

        # Store in class
        self.der_fixed = der_fixed

        """ Times """
        # times is the time per segment, not the transition times
        if seed_times is None:
            zeroth_deriv_xyz = dict()
            for key in ['x', 'y', 'z']:
                zeroth_deriv_xyz[key] = self.waypoints[key][0,:]
            self.times = utils.get_seed_times(zeroth_deriv_xyz, seed_avg_vel)
        else:
            self.times = seed_times

        """ Check Input """
        for key in self.waypoints.keys():
            if np.shape(self.waypoints[key]) != np.shape(self.der_fixed[key]):
                raise exceptions.InputError(self.waypoints[key],
                                            "Mismatch between size of waypoints"
                                            + " array and size of derivative fixed"
                                            + "array")

            if np.size(self.times) != np.shape(self.der_fixed[key])[1] - 1:
                raise exceptions.InputError(self.times,
                                            "Mismatch between number of segment times"
                                            + " and number of segments")

        """ Number of coefficients """
        self.n_coeff = utils.update_n_coeffs(order,self.n_seg)

        """ Path Weight """
        if path_weight is None:
            path_weight = dict()
            # Loop for each dimension
            for key in waypoints.keys():
                path_weight[key] = 1.0

        self.path_weight = path_weight

        """ Constraints """
        self.constraint_list = []

        """Initial Setup"""
        self.reset_astro(new_times_flag=True,new_size=True)

    def reset_astro(self,new_times_flag=True,new_size=True, indices=None,old_times = None):
        """ Hessian Inverse """
        self.H_inv = np.identity(self.n_coeff)

        """ Data Tracking """
        # initilise data track class
        self.data_track = data_track(self.waypoints.keys())

        """ Compute initial matrices """
        if new_size:
            # Legendre Polynomails
            self.generate_legendre_poly()

        if new_size or new_times_flag:
            # Scale the matrices with time
            self.poly.scale_matrices(self.times)

        if new_size:
            # Re-create the P_BC_C matrix
            self.create_P_BC()
        elif new_times_flag and not new_size:
            # Scale P_BC
            self.rescale_P_BC(indices,old_times)

        # Pre compute matrix for subspace projection
        self.compute_projected_matrix()

    def run_astro(self,replan=False):
        """
        High level function to run the astro trajectory optimisation

        Runs the optimisastion and computes the trajectory for the result

        Uses:
            All

        Modifies:
            All
        """
        # Run initial guesstimes
        if not replan or self.c_leg_poly is None:
            self.initial_guess()

        if not replan:
            # Reset hessian Inverse
            self.H_inv = np.identity(self.n_coeff)

        # Optimise
        self.optimise()

        # Compute the trajectory
        self.get_trajectory()

        print("\nCOMPLETED RUN_ASTRO\n")

    def run_astro_with_increasing_weight(self,weights=1,weight_factor=100,reset_count=False):

        # Run initial guesstimes
        if self.c_leg_poly is None:
            self.initial_guess()

        # weights = 1
        # weight_factor = 100
        # reset_count = False
        exit_flag = False
        start_time = time.time()
        while not exit_flag:

            # Set weightings to zero
            for i in range(np.size(self.constraint_list)):

                self.set_constraint_weight(weights,self.constraint_list[i].constraint_type)

            # Optimise with mutation
            self.mutate_optimise()

            if self.feasible:
                # Free of collisions
                exit_flag = True
                break

            weights *= weight_factor

            if weights > 1e10 and reset_count is False:
                weights = 10
                reset_count = True

        self.data_track.outer_opt_time = time.time() - start_time
        # Compute the trajectory
        self.get_trajectory()

    def update_times(self, indices, new_times, new_coeffs=False, defer=False):
        """
        Function to update the times for each segment


        """
        old_times = self.times.copy()
        self.times[indices] = new_times

        # Reset the matrices for the new times
        self.reset_astro(new_times_flag=True,new_size=False,indices=indices,old_times=old_times)

        # Enforce boundary conditions for the existing C_leg_poly
        if not new_coeffs:
            self.c_leg_poly = self.enforce_BCs(self.c_leg_poly)
        else:
            self.initial_guess()


        if not defer:
            self.optimise()

        self.get_trajectory()

    def update_waypoint(self,index,new_waypoint,new_der_fixed=dict(x=None, y=None, z=None,
                       yaw=None),defer=False):
        # change the waypoint
        for key in new_waypoint.keys():
            if new_waypoint[key] is not None: # No changes if input is none
                n_to_append = self.n_der[key] - np.size(new_waypoint[key])
                if n_to_append > 0:
                    if new_der_fixed[key] is not None:
                        if (index == 0) or (index == (self.n_seg + 1)):
                            # first or last waypoint of entire trajectory -> fix the
                                # higher order derivatives by default
                            new_der_fixed[key] = np.append(np.atleast_1d(new_der_fixed[key]),
                                                           [True] * n_to_append,
                                                           axis=0)
                        else:
                            # float the higher order derivatives by default at internal
                                # waypoints
                            new_der_fixed[key] = np.append(np.atleast_1d(new_der_fixed[key]),
                                                           [False] * n_to_append,
                                                           axis=0)
                        # Update
                        self.der_fixed[key][:,index] = new_der_fixed[key].copy()

                    new_waypoint[key] = np.append(np.atleast_1d(new_waypoint[key]),
                                                  [0.0] * n_to_append,
                                                  axis=0)

                # Update
                self.waypoints[key][:,index] = new_waypoint[key].copy()
                # self.der_ineq[key] = self.quad_traj[key].der_ineq

        # Recompute matrices with new waypoint
        self.reset_astro(new_times_flag=True,new_size=True)
        self.initial_guess()

        if not defer:
            self.optimise()

        self.get_trajectory()

    def insert_waypoint(self,new_index,new_waypoint,new_times,new_der_fixed=dict(x=True, y=True, z=True,
                       yaw=True),defer=False):
        """ inserting a new waypoint """
        # change the waypoint
        for key in new_waypoint.keys():
            if new_waypoint[key] is not None: # No changes if input is none
                n_to_append = self.n_der[key] - np.size(new_waypoint[key])
                if n_to_append > 0:
                    if new_der_fixed[key] is None:
                        new_der_fixed[key] = True
                    if (new_index == 0) or (new_index == (self.n_seg + 1)):
                        # first or last waypoint of entire trajectory -> fix the
                            # higher order derivatives by default
                        new_der_fixed[key] = np.append(np.atleast_1d(new_der_fixed[key]),
                                                       [True] * n_to_append,
                                                       axis=0)
                    else:
                        # float the higher order derivatives by default at internal
                            # waypoints
                        new_der_fixed[key] = np.append(np.atleast_1d(new_der_fixed[key]),
                                                       [False] * n_to_append,
                                                       axis=0)

                    new_waypoint[key] = np.append(np.atleast_1d(new_waypoint[key]),
                                                  [0.0] * n_to_append,
                                                  axis=0)

                # Modify waypoints
                self.waypoints[key] = np.insert(self.waypoints[key],new_index,new_waypoint[key],axis=1)

                # Modify der_fixed
                self.der_fixed[key] = np.insert(self.der_fixed[key],new_index,new_der_fixed[key],axis=1)
                if new_index == 0:
                    self.der_fixed[key][1:,1] = False
                elif new_index == (self.n_seg + 1):
                    self.der_fixed[key][1:,-2] = False

        # modify n_seg - increase by 1
        self.n_seg += 1

        # modify times - add a time and change time for segments on either side of new point
        if new_index > self.times.size:
            self.times = np.append(self.times,new_times[0])
        else:
            self.times = np.insert(self.times,new_index,new_times[1])

        if new_index != 0 and new_index <= self.times.size-1:
            self.times[new_index-1] = new_times[0]

        # Update number of coefficients
        self.n_coeff = utils.update_n_coeffs(self.order,self.n_seg)

        # Recompute matrices with new waypoint
        self.reset_astro(new_times_flag=False,new_size=True)
        self.initial_guess()

        if not defer:
            self.optimise()

        self.get_trajectory()

    def delete_waypoint(self, delete_index, new_time, defer=False):
        """ Delete a waypoint """
        # Take out waypoint and der_fixed
        mask = np.ones(self.n_seg+1,dtype=bool)
        mask[delete_index] = False
        for key in self.waypoints.keys():
            self.waypoints[key] = self.waypoints[key][:,mask]
            self.der_fixed[key] = self.der_fixed[key][:,mask]
            if delete_index == 0:
                self.der_fixed[key][:,0] = True
            if delete_index == self.n_seg:
                self.der_fixed[key][:,-1] = True

        # modify n_seg - decrease by 1
        self.n_seg -= 1

        # modify times - delete and replace segment with input time
        if delete_index < self.times.size:
            self.times = np.delete(self.times,delete_index)
        else:
            # Delete the last term
            self.times = np.delete(self.times,self.times.size-1)

        if delete_index != 0 and delete_index < self.times.size+1:
            # Update the time of the segment
            self.times[delete_index-1] = new_time

        # Update number of coefficients
        self.n_coeff = utils.update_n_coeffs(self.order,self.n_seg)

        # Recompute matrices with new waypoint
        self.reset_astro(new_times_flag=False,new_size=True)
        self.initial_guess()

        if not defer:
            self.optimise()

        self.get_trajectory()

    def initial_guess(self,c_leg_poly=None):
        """
        Generate the initial least squares guess to fit the boundary conditions

        Uses:
            self.
            order: Dict with order of poly for each dimension
            waypoints: dict with waypoints for each dimension (np.array([n_der,n_waypoints]))
            der_fixed: dict with bool of which waypoint derivatives are fixed for each dimension (np.array([n_der,n_waypoints]))
            P_BC_C: dict for each dimension with the evalualted polynomial values for the boundary conditions (including continuity)
                    np.array([number of boundary conditions, number of coefficients])
            n_der: dict with the number of derivatives for each dimension
            n_seg: number of segments

        Modifies:
            self.
            c_leg_poly: coefficients to optimise. dictionary for each dimension
                    In each dimension is an np.array storing the coefficients for that dimension across all segments (stacked 1 segment at a time)
            bc:     boundary conditions that are fixed. dictionary for each dimension
                    In each dimension is an np.array storing the fixed derivatives from waypoint 1 to the last
        """

        # Name for convenience
        waypoints = self.waypoints
        der_fixed = self.der_fixed
        P_BC_C = self.P_BC_C
        n_der = self.n_der

        # Initialise
        bc = dict()
        c_start = dict()

        if self.optimise_yaw_only:
            iter_list = ["yaw"]
        else:
            iter_list = P_BC_C.keys()

        # loop for each dimension
        for key in iter_list:
            N = self.order[key]+1 # Number of coefficients
            # Initialise coefficient matrix
            c_start[key] = np.zeros(N*self.n_seg)

            # Number of constraints to fix (number of boundary conditions)
            nBCs = int(P_BC_C[key].shape[0])

            # split of terms to solve for per segment
            # Equal number of terms per segment
            n_C = np.ones([self.n_seg,1],dtype=int)*int(np.floor(nBCs/self.n_seg))
            # Divide the remaining terms (remainder from even division)
            delta = nBCs - sum(n_C)
            n_C[0:int(delta)] += 1
            # TODO(bmorrell@jpl.nasa.gov) work out how to do this more smartly - to select a good number per segment

            # initialise index
            p_index = np.array([],dtype=int)
            # loop for each segment
            for k in range(0,self.n_seg):
                # index for P_BC to use for the least squares solution:
                # from the start of the segment to the number of coeffs to solve for that segment
                p_index = np.concatenate([p_index,np.arange(k*N,k*N+n_C[k],1,dtype=int)])

            # Group together the boundary condition values for each dimension. Zeros for the continuity constraints
            bc[key] = np.concatenate([waypoints[key].T[np.nonzero(der_fixed[key].T)].T,np.zeros([n_der[key]*(self.n_seg-1)])])

            # Solve least squares for the coefficients in each segment, for each dimension, to fit the boundary conditions
            # c_start[key][p_index] = np.array(np.matrix(sp.linalg.pinv2(P_BC_C[key][:,p_index]))*np.matrix(bc[key]).T)

            mat = P_BC_C[key][:,p_index]
            # SVD to get the inverse
            # um, sm, vm = sp.linalg.svd(mat)
            # sm = sp.linalg.diagsvd(sm,sm.size,sm.size)
            # mat_inv = sp.matrix(vm).T*(sp.matrix(sp.linalg.inv(sm)))*sp.matrix(um).T
            # mat_inv1 = sp.dot(vm.transpose(),sp.dot(sp.linalg.inv(sm),um.transpose()))
            # mat_inv2 = sp.linalg.pinv(mat)
            mat_inv = sp.linalg.pinv2(mat)

            c_start[key][p_index] = np.array(np.matrix(mat_inv)*np.matrix(bc[key]).T)[:,0]

        # store the result
        if self.optimise_yaw_only:
            if c_leg_poly is not None:
                c_leg_poly['yaw'] = c_start['yaw']
                self.bc['yaw'] = bc['yaw']
                return c_leg_poly
            else:
                self.c_leg_poly['yaw'] = c_start['yaw']
                self.bc['yaw'] = bc['yaw']
        else:
            self.c_leg_poly = c_start
            self.bc = bc

    def compute_projected_matrix(self):
        """
        To pre-compute the matrix for projecting onto subspace

        Uses:
            self.
            P_BC_C: dict for each dimension with the evalualted polynomial values for the boundary conditions (including continuity).
                    np.array([number of boundary conditions, number of coefficients])

        Modifies:
            self.
            proj_mat: matrix used in projecting solutions onto the subspace that adheres to the BCs
                    In each dimension is a 2D np.array
        """

        P_BC_C = self.P_BC_C

        # initialise
        proj_mat = dict()

        # Loop for each dimension
        for key in P_BC_C.keys():
            # number of boundary conditions
            nBCs = int(P_BC_C[key].shape[0])
            # matrix for the boundary condition polynomial matrix
            P_BC_mat = sp.matrix(P_BC_C[key])
            # identity matrix
            eye_mat = sp.matrix(np.identity(nBCs))

            # Compute the projected matrix and convert to a np array
            # proj_mat[key] = np.array(P_BC_mat.T*(np.linalg.inv(P_BC_mat*P_BC_mat.T)*eye_mat))
            # pm_1 = np.array(P_BC_mat.T*(np.linalg.inv(P_BC_mat*P_BC_mat.T)*eye_mat))
            mat = P_BC_mat*P_BC_mat.T
            mat_inv = sp.linalg.pinv2(mat)

            proj_mat[key] = np.array(P_BC_mat.T*(sp.matrix(mat_inv)*eye_mat))

        self.proj_mat = proj_mat

    def create_P_BC(self):
        """
        Create the matrix of evalualted polynomails at the boundary conditions.

        The matrix, for each dimension, when multiplied by the polynomail coefficients
        gives the value of the trajectory at the boundary conditions.

        Used in computing the initial guess and in enforcing compliance with the boundary conditions

        Uses:
            self.
            order: Dict with order of poly for each dimension
            waypoints: dict with waypoints for each dimension (np.array([n_der,n_waypoints]))
            der_fixed: dict with bool of which waypoint derivatives are fixed for each dimension (np.array([n_der,n_waypoints]))
            state_scaled: dict of np.arrays with the scaled, evaluated polynomials at each timestep, for each dimension (np.array([n_der,n_timesteps]))
            n_der: dict with the number of derivatives for each dimension
            n_seg: number of segments

        Modifies:
            self.
            P_BC_C: dict for each dimension with the evalualted polynomial values for the boundary conditions (including continuity)
                    np.array([number of boundary conditions, number of coefficients])
        """
        waypoints = self.waypoints
        state_scaled = self.poly.state_scaled
        der_fixed = self.der_fixed
        n_seg = self.n_seg
        n_der = self.n_der

        # initialise
        P_BC_C = dict()
        start_index = dict()

        # Loop for each dimension
        for key in waypoints.keys():
            # number of coefficients
            N = self.order[key]+1

            # initialise the reference vector
            bcVec = np.zeros([n_der[key]*2,1],dtype=bool)
            start_index[key] = np.zeros(n_seg+1,dtype=int)

            # Create P_BC - polynomial matrix for boundary conditions
            # Loop for each segment
            for i in range(0,n_seg):
                # Base boundary conditions matrix, describing the polynomial values at the start and end of segment
                P_BC_base = np.vstack([state_scaled[key][0,:,:,i].T,state_scaled[key][-1,:,:,i].T])

                if i == 0:
                    # First segment, capture both start and end constraints
                    bcVec = np.reshape(der_fixed[key][:,i:(i+2)],n_der[key]*2,1)
                else:
                    # Other segments will have the continuity constraints fix the start, so leave it as zeros
                    bcVec = np.concatenate([np.zeros(n_der[key],dtype=bool),np.reshape(der_fixed[key][:,i+1],n_der[key],1)])

                # build together the blocks that have a fixed constraint
                if i == 0:# initialise
                    P_BC = P_BC_base[bcVec,:]
                else:
                    P_BC = sp.linalg.block_diag(P_BC,P_BC_base[bcVec,:])

                start_index[key][i+1] = start_index[key][i] + sum(bcVec)

            # Form continuity matrix
            if n_seg>1:
                # loop for each internal waypoint
                for i in range(0,n_seg-1):
                    # Base boundary conditions matrix, describing the polynomial values at the start and end of segment
                    P_BC_base1 = np.vstack([state_scaled[key][0,:,:,i].T,state_scaled[key][-1,:,:,i].T]) # For earlier Seg
                    P_BC_base2 = np.vstack([state_scaled[key][0,:,:,i+1].T,state_scaled[key][-1,:,:,i+1].T]) # For later seg

                    if i == 0:
                        # initialise
                        P_c_end = P_BC_base1[n_der[key]:,:]
                        P_c_start = P_BC_base2[0:n_der[key],:]
                    else:
                        # the end states
                        P_c_end = sp.linalg.block_diag(P_c_end,P_BC_base1[n_der[key]:,:])
                        # the starting states
                        P_c_start = sp.linalg.block_diag(P_c_start,P_BC_base2[0:n_der[key],:])

                # Offset start and end matrices
                P_c_end = np.hstack([P_c_end,np.zeros([(n_seg-1)*n_der[key],N])])
                P_c_start = np.hstack([np.zeros([(n_seg-1)*n_der[key],N]),P_c_start])

                # combine together to get the continuity matrix
                P_C = P_c_end - P_c_start

                # Stack BC and C matrices
                P_BC_C[key] = np.vstack([P_BC,P_C])
            else:
                # No contiuity constraints, just take the boundary conditions (for the case of one segment)
                P_BC_C[key] = P_BC

        self.P_BC_C = P_BC_C
        self.start_index_p_bc = start_index

    def rescale_P_BC(self,indices,old_times):
        """

        times: time duration for each segment

        """

        # Check input
        # indices - at aleast 1 d

        der_fixed = self.der_fixed
        n_seg = self.n_seg
        n_der = self.n_der
        P_BC_C = self.P_BC_C
        start_index = self.start_index_p_bc
        times = self.times

        # To rescale - new times over the old times
        scale_factors = times/old_times

        # Loop for each dimension
        for key in der_fixed.keys():
            # number of coefficients
            N = self.order[key]+1

            # Powers vector (for time scaling)
            powers = np.flipud(np.arange(0,n_der[key],1))

            # initialise the reference vector
            bcVec = np.zeros([n_der[key]*2,1],dtype=bool)

            # Start indices
            s_ind = start_index[key]

            # Modify P_BC - polynomial matrix for boundary conditions
            # Loop for each segment that is changed
            for i in indices:
                if i == 0:
                    # First segment, capture both start and end constraints
                    bcVec = np.reshape(der_fixed[key][:,i:(i+2)],n_der[key]*2,1)

                else:
                    # Other segments will have the continuity constraints fix the start, so leave it as zeros
                    bcVec = np.concatenate([np.zeros(n_der[key],dtype=bool),np.reshape(der_fixed[key][:,i+1],n_der[key],1)])


                # Vector to scale times (start and end constraints on that segment)
                rescale_vec = np.concatenate([(scale_factors[i])**powers,(scale_factors[i])**powers]).reshape(len(bcVec),1)

                # Take out the relevant terms for boundary conditions
                rescale_vec_bc = rescale_vec[bcVec,:]

                # Rescale boundary conditions
                P_BC_C[key][s_ind[i]:s_ind[i+1],N*i:N*(i+1)] = (rescale_vec_bc*np.ones(N)) * P_BC_C[key][s_ind[i]:s_ind[i+1],N*i:N*(i+1)]

                # Rescale continuity conditions
                if n_seg>1:
                    if i == 0:
                        P_BC_C[key][s_ind[-1]:(s_ind[-1]+n_der[key]),N*i:N*(i+1)] = (rescale_vec[0:n_der[key],:]*np.ones(N)) * P_BC_C[key][s_ind[-1]:(s_ind[-1]+n_der[key]),N*i:N*(i+1)]

                    elif i == n_seg-1:
                        P_BC_C[key][(s_ind[-1]+n_der[key]*(i-1)):(s_ind[-1]+n_der[key]*(i)),N*i:N*(i+1)] = (rescale_vec[0:n_der[key],:]*np.ones(N)) * P_BC_C[key][(s_ind[-1]+n_der[key]*(i-1)):(s_ind[-1]+n_der[key]*(i)),N*i:N*(i+1)]

                    else:
                        P_BC_C[key][(s_ind[-1]+n_der[key]*(i-1)):(s_ind[-1]+n_der[key]*(i+1)),N*i:N*(i+1)] = (rescale_vec*np.ones(N)) * P_BC_C[key][(s_ind[-1]+n_der[key]*(i-1)):(s_ind[-1]+n_der[key]*(i+1)),N*i:N*(i+1)]

            self.P_BC_C = P_BC_C

    def generate_legendre_poly(self):
        """
        Compute the Legendre polynomial components required for the optimisation

        This requries computing the base Legendre polynomial coefficients and
        then the integrals and derivatives of these

        The output is used in evaluating the trajectory, and enforcing boundary conditions

        Uses:
            self.
            order: Dict with order of poly for each dimension
            n_der: dict with the number of derivatives for each dimension
            n_samp: Number of samples along the trajectory

        Methods:
            self.legendre_matrix: function that generates a matrix of Legendre polynomial coefficients
            poly_astro: Class to store polynomail components

        Modifies:
            self.
            pol_leg: dict for each dimension with the coefficients for the Legendre polynomials at each derivative
                     each is a 3D np.array([number of basis polynomials (N), number of coefficients (N), number of derivatives])
            tsteps: np.array vector for each time sample along the trajectory
            poly: class poly_astro that stores the polynomail evaluations and cost terms
        """

        order = self.order
        n_der = self.n_der

        # initialise
        pol_legStore = dict()

        # Loop for each dimension
        for key in order.keys():
            # highest order derivative
            max_deriv = (n_der[key]-1)
            N = order[key]+1 # Number of coefficients

            # initialise
            pol_leg = np.zeros([N,N,max_deriv+1])

            # polynomial for the highest order derviative (where the cost is computed)
            # Highest order derivative is placed last in the 3rd dimension (so position is at index 0 in the 3rd dimension)
            pol_leg[max_deriv:,max_deriv:,max_deriv] = self.legendre_matrix(N - max_deriv)

            # initialise the constant for each derivative (other than highest derivative)
            for j in range(0,max_deriv):
                pol_leg[j,-1,j] = 1

            # Loop through each Legendre Polynomial and compute the derivative and integral
            for i in range(1,N):
                # Get the higher order derivatives
                for k in range(1,max_deriv+1):
                    # Counting from the highest derivative back (see reference "max_deriv-k")
                    # Take the lower order polynomial for the set at one integration up
                    if i <= N - (max_deriv+1-k): # if there is an existing derivative for the given i
                        pol_leg[max_deriv-k+i,-(i+1):,max_deriv-k] = np.polyint(np.poly1d(pol_leg[max_deriv-k+i,:,max_deriv-(k-1)]),1,0).c

                # Set constant of integration, to make it equal to zero at x = +_ 1
                if i>1: # special case for the first term , so only for i>1
                    # for each derivative
                    for ijk in range(1,max_deriv+1):
                        if i <= N - (max_deriv+1-ijk):
                            # Sum up all the coefficients to give the constant term (so when t = 1, the value is zero)
                            pol_leg[max_deriv-ijk+i,-1,max_deriv-ijk] = sum(pol_leg[max_deriv-ijk+i,0:-1,max_deriv-ijk])

            # Store for each dimension
            pol_legStore[key] = pol_leg

        # Store
        self.pol_leg = pol_legStore

        #######################################################################
        # Evaluate the polynomials
        # vector for the timesteps
        self.tsteps = np.linspace(-1,1,self.n_samp)

        # Evaluate polynomials and cost integrals - gets the Unscaled trajectory (Need to be scaled)
        self.poly = poly_astro(self.pol_leg,self.tsteps)

    def get_trajectory(self,c_leg_poly_test=None):
        """
        Compute the trajectory for the current c_leg_poly - the optimisation coefficients

        Used for evaluating constraint costs, and plotting the trajectory

        Args:
            c_leg_poly_test: Used only if testing the cost of a coefficient set (otherwise default of None)
                    coefficients to optimise. dictionary for each dimension
                    In each dimension is an np.array storing the coefficients for that dimension across all segments (stacked 1 segment at a time)

        Uses:
            self.
            order: Dict with order of poly for each dimension
            n_der: dict with the number of derivatives for each dimension
            c_leg_poly: coefficients to optimise. dictionary for each dimension
                    In each dimension is an np.array storing the coefficients for that dimension across all segments (stacked 1 segment at a time)
            n_samp: number of samples in time
            n_seg: number of segments
            poly.state_scaled: dict for each dimension: the scaled polynomial evaluations at each timestep.
                               3D np.array([n_samp,number of coefficients (N), n_der])

        Modifies:
            self.
            state: dict for each dim, 3D np.array([n_der,n_samp,n_seg]). Stores all timesteps for each dimension for each segment
            state_combined: as with state but combines the samples in a stack for each segment (np.array([n_der,n_samp*n_seg]))
        """

        if c_leg_poly_test is None:
            c_leg_poly = self.c_leg_poly
        else:
            c_leg_poly = c_leg_poly_test

        n_der = self.n_der

        # initialise
        state = dict()
        state_combined = dict()

        # Loop through each dimension
        for key in c_leg_poly.keys():
            # initialise. Rows for each derivative, columns for each time index
            state[key] = np.zeros([n_der[key],self.n_samp,self.n_seg]) # 3rd dimension for each segment
            state_combined[key] = np.zeros([n_der[key],self.n_samp*self.n_seg]) # Segments stacked
            N = self.order[key]+1
            # Loop through each derivative
            for i in range(0,n_der[key]):
                    # Loop for each segment
                    for k in range(0,self.n_seg):
                        # Compute the state for the given derivative and segment
                        state_part = self.poly.state_scaled[key][:,:,i,k].dot(c_leg_poly[key][N*k:N*(k+1)])
                        # Assign to the state dict
                        state[key][i,:,k] = state_part

                        # Stack in the state_combined dict
                        state_combined[key][i,self.n_samp*k:self.n_samp*(k+1)] = state_part

        # Store
        if c_leg_poly_test is None:
            self.state = state
            self.state_combined = state_combined
        else:
            return state


    def set_yaw_des_from_traj(self, state=None):

        self.yaw_to_traj = True

        trans_times = utils.seg_times_to_trans_times(self.times)

        #TODO(mereweth@jpl.nasa.gov) - use self.n_seg?
        n_seg = np.size(self.times)

        # set the desired yaw values at the waypoints to point the quadrotor
        # along the current xyz trajectory

        # velocity might be zero at a waypoint but unlikely to be in between?
        # TODO(mereweth@jpl.nasa.gov) - check if velocity is zero for either
        # x or y and move by increments of yaw_eps

        # we still need curvature at every waypoint to tell us whether to
            # increment/decrement by 2*pi. This is independent of whether
            # face_front is set for a given waypoint.
        x_vel = np.zeros(np.shape(trans_times))
        y_vel = np.zeros(np.shape(trans_times))
        x_acc = np.zeros(np.shape(trans_times))
        y_acc = np.zeros(np.shape(trans_times))

        if state is None:
            state = self.state

        start_index = 1
        x_vel[0] = state['x'][1,start_index,0]
        y_vel[0] = state['y'][1,start_index,0]
        x_acc[0] = state['x'][2,start_index,0]
        y_acc[0] = state['y'][2,start_index,0]

        x_vel[1:-1] = state['x'][1,0,1:]
        y_vel[1:-1] = state['y'][1,0,1:]
        x_acc[1:-1] = state['x'][2,0,1:]
        y_acc[1:-1] = state['x'][2,0,1:]

        end_index = -2
        x_vel[-1] = state['x'][1,end_index,-1]
        y_vel[-1] = state['y'][1,end_index,-1]
        x_acc[-1] = state['x'][2,end_index,-1]
        y_acc[-1] = state['y'][2,end_index,-1]

        # TODO(mereweth@jpl.nasa.gov) - check self.face_front for each waypoint
        # face_front_idx = np.nonzero(self.face_front)
        self.waypoints['yaw'][0, 1:-1] = np.arctan2(y_vel, x_vel)[1:-1]

        # calculate curvature
        curv = (x_vel * y_acc - y_vel * x_acc) / \
            (x_vel ** 2.0 + y_vel ** 2.0) ** 1.5

        # TODO(mereweth@jpl.nasa.gov) - can this formulation produce more than
        # 2pi rotation per waypoint?

        # was using the curvature to decide whether to offset +2pi, -2pi, or none
        # instead, simply check whether we would rotate by more than pi in either
            # direction
        yaw_offset = 0.0
        for i in range(n_seg):
            # if (next_yaw - yaw) < 0 AND positive curvature:
            #     increase next_yaw by 2pi
            #if (yaw_offset + self.waypoints['yaw'][0, i + 1]
            #        - self.waypoints['yaw'][0, i]) < 0 and curv[i] > 0:
            while (yaw_offset + self.waypoints['yaw'][0, i + 1]
                    - self.waypoints['yaw'][0, i]) < -np.pi:
                yaw_offset += 2 * np.pi

            # if (next_yaw - yaw) > 0 AND negative curvature:
            #     decrease next_yaw by 2pi
            #elif (yaw_offset + self.waypoints['yaw'][0, i + 1]
            #      - self.waypoints['yaw'][0, i]) > 0 and curv[i] < 0:
            while (yaw_offset + self.waypoints['yaw'][0, i + 1]
                  - self.waypoints['yaw'][0, i]) > np.pi:
                yaw_offset -= 2 * np.pi

            self.waypoints['yaw'][0, i + 1] += yaw_offset

        # print("yaw is\n")
        # print(self.waypoints['yaw'])
            # HACK
            # self.waypoints['yaw'][0, i + 1] = np.pi/2

        # set last yaw waypoint to be the same as the second last
        # self.waypoints['yaw'][0,-1] = self.waypoints['yaw'][0,-2]

    def set_yaw_to_zero(self):

        self.yaw_to_traj = False

        self.waypoints['yaw'][0,:] = 0.0

        self.optimise_yaw_only = True

        # update yaw components of CLegPoly
        self.initial_guess()

        self.optimise_yaw_only = False


    def set_yaw_trajectory_to_velocity_trajectory(self):

        self.get_trajectory()

        self.set_yaw_des_from_traj()

        self.optimise_yaw_only = True

        # update yaw components of CLegPoly
        self.initial_guess()

        # optimise yaw only
        # self.optimise()

        self.optimise_yaw_only = False

    def compute_path_cost_grad(self,c_leg_poly,doGrad=True,doCurv=False):
        """
        Computes cost and cost gradient of the path. A dictionary for each dimension

        Cost and gradient are returned for use in optimisation steps (e.g. linesearch)

        Args:
            c_leg_poly: coefficients to compute the cost for. dictionary for each dimension
                    In each dimension is an np.array storing the coefficients for that dimension across all segments (stacked 1 segment at a time)
            doGrad: boolean to select whether or not to evaluate the gradient

        Uses:
            self.
            order: Dict with order of poly for each dimension
            n_seg: number of segments
            path_weight: the weight for the path cost in each dimension
            poly.int_der_sq: the pre-computed integral of the derivative squared component to evaluation the cost
                             dict for each dimension. Each an np.array of length N

        Returns:
            path_cost: the cost for the path (single number)
            path_cost_grad: the cost gradient for the path. A dict for each dimension,
                            in each an np.array of length N
            path_cost_curv:
        """

        path_weight = self.path_weight
        int_der_sq = self.poly.int_der_sq

        # Initialise
        path_cost = 0.0
        path_cost_grad = dict()
        path_cost_curv = dict()

        # Loop for each dimension
        for key in c_leg_poly.keys():
            N = self.order[key]+1

            # Initialise
            path_cost_grad[key] = np.zeros(N*self.n_seg)
            path_cost_curv[key] = np.zeros(N*self.n_seg)

            if self.optimise_yaw_only and key != "yaw":
                # Flag set to only consider yaw
                print("getting path cost for yaw only")
                continue

            # Loop for each segment
            for i in range(0,self.n_seg):
                # Store the Coefficients as a matrix - taking out the terms for the current segment
                coeff_matt = c_leg_poly[key][N*i:N*(i+1)]

                # evaluation the path cost (utilising the pre-computed cost matrix) - add for each dimension
                path_cost += path_weight[key]*((int_der_sq[key]*coeff_matt).dot(coeff_matt))

                # gradient
                if doGrad:
                    path_cost_grad[key][N*i:N*(i+1)] = 2.0*int_der_sq[key]*coeff_matt*path_weight[key]
                else:
                    # Zeros if gradient not needed
                    path_cost_grad[key][N*i:N*(i+1)] = np.zeros(coeff_matt.shape)

                # Curvature
                if doCurv:
                    path_cost_curv[key][N*i:N*(i+1)] = 2.0*int_der_sq[key]*path_weight[key]


        # Return cost and gradient
        return path_cost, path_cost_grad, path_cost_curv

    def total_cost_grad_curv(self, c_leg_poly, doGrad=True,doCurv=False,num_grad=False):
        """
        Computing the combined cost, gradient and step for the path and constraints

        """

        cost_step = dict()
        cost_step_non_convex = dict()
        cost_grad_non_convex = dict()
        max_viol = 0.0
        feasible = True # initilise
        esdf_check_all_feasible = False # used if an esdf collision checker is used

        for key in c_leg_poly.keys():
            N = np.size(c_leg_poly[key])
            cost_step[key] = np.zeros(N)
            cost_step_non_convex[key] = np.zeros(N)
            cost_grad_non_convex[key] = np.zeros(N)

        # Compute cost and gradient
        path_cost, path_cost_grad, path_cost_curv = self.compute_path_cost_grad(c_leg_poly,doGrad=doGrad,doCurv=doCurv)

        print("Path cost is: {}".format(path_cost))

        grad_mult = 1.0
        # # Zero out path cost for testing (uncomment below)
        # if np.size(self.constraint_list) is not 0:# and False:
        #     path_cost = 0.0
        #     grad_mult = 0.0

        # Add to total cost
        cost = path_cost
        cost_grad = path_cost_grad
        cost_curv = dict()
        if doCurv:
            for key in path_cost_curv.keys():
                cost_curv[key] = np.diag(path_cost_curv[key])*grad_mult

        for key in cost_grad.keys(): cost_grad[key] *= grad_mult

        # if doCurv:
        #     # Using curvature to set gradient step size
        #     cost_step = path_cost_grad./path_cost_curv
        #     cost_step[np.isnan(cost_step)] = 0


        # Don't check constraints if only optimising yaw
        if not self.optimise_yaw_only:
            # Get state to check constraints
            state = self.get_trajectory(c_leg_poly)
            self.temp_state = state

            # Constraint costs
            for i in range(np.size(self.constraint_list)):

                # Compute cost for each constraint
                constr_cost, constr_cost_grad, constr_cost_curv = self.constraint_list[i].compute_constraint_cost_grad_curv(state,
                                                        self.poly.state_scaled,
                                                        doGrad = doGrad,
                                                        doCurv=doCurv,
                                                        path_cost = path_cost)

                # Check feasibility - for all constriants
                feasible = feasible and self.check_if_feasible(i)

                if self.constraint_list[i].constraint_type is "esdf_check":
                    esdf_check_all_feasible = self.check_if_feasible(i)
                    continue

                if doGrad:
                    print("Obstacle cost for constraint {}, of type {} on der: {} is {}".format(i,self.constraint_list[i].constraint_type,self.constraint_list[i].der, constr_cost))

                # Add to total
                cost += constr_cost

                # Track maximum violation
                max_viol = np.max([max_viol,constr_cost])

                if doCurv:
                    if self.constraint_list[i].keep_out is True or self.constraint_list[i].constraint_type is "esdf" :
                        # Non-convex
                        for key in constr_cost_grad.keys():
                            # To have large steps the more a constraint is violated
                            # cost_step_add= constr_cost/constr_cost_grad[key]
                            cost_step_add = constr_cost_grad[key]

                            # cost_step_add = constr_cost_grad[key]/np.linalg.norm(constr_cost_grad[key])
                            # cost_step_add = cost_step_add*cost_step_add_num

                            # if np.sum(constr_cost_grad[key])>0.0:
                            #     import pdb; pdb.set_trace()
                            # TODO Check this
                            # cost_step_add = constr_cost_grad[key]
                            cost_step_add[np.isnan(cost_step_add)] = 0.0
                            cost_step_add[np.isinf(cost_step_add)] = 0.0
                            cost_step_non_convex[key] += cost_step_add

                            # Store to keep separate for adding later
                            cost_grad_non_convex[key] += constr_cost_grad[key]
                    else:
                        # Store just the convex components
                        # # Using curvature to set gradient step size
                        # constr_cost_step = constr_cost_grad./constr_cost_curv
                        # constr_cost_step[np.isnan(constr_cost_step)] = 0

                        for key in cost_grad.keys():
                            cost_grad[key] += constr_cost_grad[key]
                            cost_curv[key] += np.array(constr_cost_curv[key])
                else:
                    for key in cost_grad.keys():
                        cost_grad[key] += constr_cost_grad[key]

            # if num_grad:
            #     cost_grad2 = self.numerical_gradient(c_leg_poly)
            #     import pdb; pdb.set_trace()
            # #     # cost_grad_non_convex['y'][:,0]/np.linalg.norm(cost_grad_non_convex['y'][:,0])
            # #     # cost_grad2['y']/np.linalg.norm(cost_grad2['y'])
            # # #     curv2 = self.numerical_curvature(c_leg_poly)
            #     for key in cost_grad.keys():
            #         # if not np.allclose(cost_grad2[key],cost_grad_non_convex[key][:,0]):
            #         if not np.allclose(cost_grad2[key],cost_grad[key]):
            #             print("Numerical gradient is off!")
            #             import pdb; pdb.set_trace()


        if doCurv:

            # Compute step from convex obstacles
            if self.optimise_yaw_only:
                iter_list = ["yaw"]
            else:
                iter_list = cost_grad.keys()

            for key in iter_list:
                # step_size = np.array((np.matrix(cost_grad[key])*np.matrix(cost_grad[key]).T)/(np.matrix(cost_grad[key])*np.matrix(cost_curv[key])*np.matrix(cost_grad[key]).T))
                # cost_step[key] = step_size[0,0]*cost_grad[key]
                # cost_step[key][np.isnan(cost_step[key])] = 0.0
                #
                # cost_step[key] = path_cost_grad[key]/path_cost_curv[key]
                # cost_step[key][np.isnan(cost_step[key])] = 0.0
                #

                g = cost_grad[key]/np.linalg.norm(cost_grad[key])
                dir_deriv = np.matrix(cost_curv[key])*np.matrix(g).T
                step_size = np.divide(cost_grad[key],np.array(dir_deriv.T)[0,:])
                cost_step[key] = step_size*g
                cost_step[key][np.isnan(cost_step[key])] = 0.0

                # Add non-convex components
                cost_step[key] = cost_step[key] + cost_step_non_convex[key]
                # Add up the gradients (if revert to gradient method outside the function)
                cost_grad[key] = cost_grad[key] + cost_grad_non_convex[key]
        if doGrad:
            # Only update the class variable when also computing gradient (i.e. not for the line search)
            self.max_viol = max_viol


        # Set feasible flag if all constriants are feasible
        if esdf_check_all_feasible:
            self.feasible = True
            if self.exit_on_feasible:
                print("ESDF Check: All constraints are feasible (within inflation region).")
        else:
            if feasible:
                if self.exit_on_feasible:
                    print("All constraints are feasible (within inflation region).")
                self.feasible = True
            else:
                self.feasible = False

        # self.first_in_iteration = False

        return cost, cost_grad, cost_curv, cost_step

    def stack_vector(self,unstacked_vec):
        """
        stacks a parameter that is split into different dimensions in a dictionary
        output is organised as [[x;y;z;yaw;...]_seg1,[x;y;z;yaw;...]_seg2,...]

        used for c_leg_poly, y and sk..., cost_grad...

        Args:
            unstacked_vec: the dict of values to be stacked. dictionary for each dimension
                    In each dimension is an np.array storing the coefficients for that dimension across all segments (stacked 1 segment at a time)

        Uses:
            self.
            order: Dict with order of poly for each dimension
            n_seg: number of segments

        Returns:
            stacked_vec: the stacked vector, an np.array of length N*n_seg*n_dim
        """

        # initialise
        stacked_vec = np.array([])

        # loop for each segment
        for i in range(0,self.n_seg):
            # Loop for each dimension
            for key in unstacked_vec.keys():
                N = self.order[key]+1 # number of coefficients
                # Stack the values for the current
                stacked_vec = np.concatenate([stacked_vec,unstacked_vec[key][N*i:N*(i+1)]])

        return stacked_vec

    def unstack_vector(self,stacked_vec):
        """
        unstacks a parameter from:
         [[x;y;z;yaw;...]_seg1,[x;y;z;yaw;...]_seg2,...]
        into one that is a dictionary for the different dimensions, stacked by segment
        used for c_leg_poly, y, cost_grad and similar

        opposite of stack_vector

        Args:
            stacked_vec: the stacked vector, an np.array of length N*n_seg*n_dim

        Uses:
            self.
            order: Dict with order of poly for each dimension
            n_seg: number of segments

        Returns:
            unstacked_vec: the dict of values unstacked. dictionary for each dimension
                    In each dimension is an np.array storing the coefficients for that dimension across all segments (stacked 1 segment at a time)
        """
        # initialise
        unstacked_vec = dict()

        # Loop for each segment
        for i in range(0,self.n_seg):
            # Set the start index for terms in the given segment
            start_index = int(self.n_coeff/self.n_seg*i)

            # Loop for each dimension
            for key in self.order.keys():
                # Number of coefficients for current dimension
                N = self.order[key]+1

                # Group the terms for the current dimensions
                if i == 0:
                    # First term - just take out
                    unstacked_vec[key] = stacked_vec[start_index:(start_index+N)]
                else:
                    # stack other terms
                    unstacked_vec[key] = np.concatenate([unstacked_vec[key],stacked_vec[start_index:(start_index+N)]])

                # Update starting point for next dimensions
                start_index += N

        # ouput
        return unstacked_vec

    def update_hessian(self,cost_grad,new_cost_grad,poly_step):
        """
        Updates the Hessian Matrix for the BFGS quasi-Newton gradient descent

        Args:
            cost_grad: the cost gradient for the path. A dict for each dimension,
                       in each an np.array or length N
            new_cost_grad: updated cost gradient
            poly_step: The step in the optimisation coefficients (same format as c_leg_poly)

        Uses:
            self.
            H_inv: the Hessian inverse matrice from the BFGS method. 2D np.array squared by N*n_seg*n_dim

        Methods:
            self.
            enforce_BCs: function to project c_leg_poly type data onto the subspace that adheres with BCs
            stack_vector: Stacking vectors from each dimension and segment together

        Modifies:
            self.
            H_inv: the Hessian inverse matrice from the BFGS method. 2D np.array squared by N*n_seg*n_dim
        """

        H_inv = self.H_inv

        # initialise
        y = dict()

        # Loop for each dimension
        for key in cost_grad.keys():
            # Compute change in gradient
            y[key] = new_cost_grad[key] - cost_grad[key]

        # Project onto subspace
        # TODO (bmorrell@jpl.nasa.gov) This projection is supposed to be different!!!!
        y = self.enforce_BCs(y,'grad')

        # Stack vector
        y_vec = self.stack_vector(y)

        # size of step (as a stacked vector)
        sk = self.step_coeff*self.stack_vector(poly_step)

        # Pre-compute y'*sk product
        y_s_prod  = y_vec.dot(sk)

        # Check for positive curvature
        if y_s_prod <= 0:
            # Curvature not positive so reset to identity
            H_inv = np.identity(y_vec.size)
        else:
            # BFGS update of hessian inverse. See http://terminus.sdsu.edu/SDSU/Math693a_f2013/Lectures/18/lecture.pdf.
            # and simplicifcation of update for computational efficiency (shown https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)
            # Change all to matrix type for the calculations
            y_vec = np.matrix(y_vec).T
            sk = np.matrix(sk).T
            H_inv = np.matrix(H_inv)
            H_inv = np.array(H_inv + float((y_s_prod + y_vec.T*H_inv*y_vec))*(sk*sk.T)/(y_s_prod**2) - (H_inv*y_vec*sk.T + sk*y_vec.T*H_inv)/y_s_prod)

        # Output
        self.H_inv = H_inv

    def enforce_BCs(self,param_in,input_type='coeff'):
        """
        Projects c_leg_poly type input onto subspace enforcing BCs

        Could be used for c_leg_poly, or new_poly, or mutated initial guesses
        Updates the Hessian Matrix for the BFGS quasi-Newton gradient descent

        Args:
            param_in: the c_leg_poly type input: dictionary for each dimension,
                      related to optimisation coefficients. In each dimension is
                      an np.array storing the coefficients for that dimension across all segments (stacked 1 segment at a time)

        Uses:
            self.
            P_BC_C: dict for each dimension with the evalualted polynomial values for the boundary conditions (including continuity)
            bc:     boundary conditions that are fixed. dictionary for each dimension
                    In each dimension is an np.array storing the fixed derivatives from waypoint 1 to the last
            proj_mat: matrix used in projecting solutions onto the subspace that adheres to the BCs
                    In each dimension is a 2D np.array
        Returns:
            param_out: the modified c_leg_poly type input from param_in
        """

        P_BC_C = self.P_BC_C
        bc = self.bc
        proj_mat = self.proj_mat

        # initialise
        param_out = dict()

        # Loop for each dimension
        for key in P_BC_C.keys():
            # Project onto feasible subspace for each dimension
            if input_type == 'coeff':
                # Standard type with the optimisation coefficients
                param_out[key] = param_in[key] - proj_mat[key].dot(P_BC_C[key].dot(param_in[key]) - bc[key])
            elif input_type == 'grad':
                # If a gradient type, then the constant term is left out
                param_out[key] = param_in[key] - proj_mat[key].dot(P_BC_C[key].dot(param_in[key]))


        return param_out

    def check_boundary_conditions(self,param_in,input_type='coeff'):
        P_BC_C = self.P_BC_C
        bc = self.bc
        proj_mat = self.proj_mat

        error = 0.0

        for key in param_in.keys():

            error += np.sqrt(np.mean(P_BC_C[key].dot(param_in[key]) -bc[key])**2)#np.mean(np.abs(P_BC_C[key].dot(param_in[key]) -bc[key]))

        return error

    def optimise(self, c_leg_poly=None, mutate_iter=4,run_one_iteration=False,mutate_serial=-1,use_quadratic_line_search=False):
        """
        Main optimisation function. BFGS quasi-Newton gradient descent solution
        using a Armijo condition on a backtracking line search.

        Optimising for the coefficients of Legendre basis polynomials to minimise cost

        Uses:
            self.
            c_leg_poly: coefficients to optimise. dictionary for each dimension
                        In each dimension is an np.array storing the coefficients
                        for that dimension across all segments (stacked 1 segment at a time)
            optim_opt: Dictionary storing optimisation settings
            H_inv: the Hessian inverse matrix from the BFGS method. 2D np.array squared by N*n_seg*n_dim
            n_coeff: total number of coefficients

            mutate_serial - flag and number: -1 to deactive, if >0 then indicates the number of iterations before mutating (randomly modifying the solution)

        Methods:
            self:
            compute_path_cost_grad:
            stack_vector
            unstack_vector:
            enforce_BCs:
            line_search
            compute_path_cost_grad
            update_hessian
            data_track.update_data_track: Variable to track the evolution of the solutions through the optimisation

        Modifies:
            Most, key components being:
            self.
            c_leg_poly: coefficients to optimise. dictionary for each dimension
                        In each dimension is an np.array storing the coefficients
                        for that dimension across all segments (stacked 1 segment at a time)
            cost: The total cost
            cost_grad: the total cost gradient
            iterations: number of iterations for the solution
            data_track: Class storing the data through the optimisation
        """

        # Initialise feasible flag to false
        self.feasible = False

        start_time = time.time()

        if use_quadratic_line_search:
            for constraint in self.constraint_list:
                if constraint.keep_out and (constraint.constraint_type is not "esdf_check"):
                    # Deactivate if there are any non-convex constraints
                    use_quadratic_line_search = False


        if c_leg_poly is None:
            # Normal use case
            c_leg_poly = self.c_leg_poly
            mutation_run = False
        else:
            # input for mutations
            mutation_run = True
        step_size = self.optim_opt['step_size']

        # initialise
        new_poly = dict()
        poly_step = dict()
        pre_step = dict()
        #constr_cost_store = np.zeros(np.size(self.constraint_list))

        # self.first_in_iteration = True

        # Compute cost gradient and step for path and constraints
        cost, cost_grad, cost_curv, cost_step = self.total_cost_grad_curv(c_leg_poly, doGrad=True,doCurv=self.curv_func,num_grad=True)
        print("Cost at start of opt is: {}".format(cost))

        # Initialisation
        exitflag = False
        if not run_one_iteration:
            self.iterations = 0
            self.mutated = False
        use_cost_step = False

        if use_quadratic_line_search is None:
            if self.curv_func or len(self.constraint_list)<1:
                use_quadratic_line_search = True
            else:
                use_quadratic_line_search = False

        if self.feasible:
            # print("Trajectory feasible before first iteration")
            if self.exit_on_feasible:
                print("Exiting because trajectory is feasible before first iteration.\n")
                exitflag = True

        # Start Optimisation loop
        while not exitflag:
            self.first_in_iteration = True
            if self.curv_func:
                # Use curvature computed step
                new_poly = self.stack_vector(c_leg_poly) - step_size*self.stack_vector(cost_step)
                use_cost_step = True
                # delta_poly = - step_size*self.stack_vector(cost_step)
                # self.analyse_convexity(c_leg_poly,self.unstack_vector(- step_size*self.stack_vector(cost_step)))
            else:
                # Take step along the gradient descent direction (BFGS, quasi-Newton optimisation)
                new_poly = self.stack_vector(c_leg_poly) - step_size*self.H_inv.dot(self.stack_vector(cost_grad))

                # delta_poly = - step_size*self.H_inv.dot(self.stack_vector(cost_grad))
                # self.analyse_convexity(c_leg_poly,self.unstack_vector(- step_size*self.H_inv.dot(self.stack_vector(cost_grad))))


            # Unstack vector
            new_poly = self.unstack_vector(new_poly)
            # delta_poly = self.unstack_vector(delta_poly)

            # if not self.curv_func or len(self.constraint_list)>0:
            #
            #     if self.curv_func:
            #         self.analyse_convexity(c_leg_poly,self.unstack_vector(- step_size*self.stack_vector(cost_step)))
            #     else:
            #         self.analyse_convexity(c_leg_poly,self.unstack_vector(- step_size*self.H_inv.dot(self.stack_vector(cost_grad))))
            #
            #     # self.c_leg_poly = new_poly
            #     # self.get_trajectory()
            #     # self.plot()
            #     # # plt.show()
            #     # self.c_leg_poly = c_leg_poly
            #     # self.get_trajectory()
            #     import pdb; pdb.set_trace()

            # Check the step is doing what it should
            # TODO (Remove for speed)
            # check_cost = self.total_cost_grad_curv(new_poly, doGrad=False,doCurv=False)[0]
            # print("check cost {} is {}".format(iteration,check_cost))
            for key in new_poly.keys(): pre_step[key] = new_poly[key] - c_leg_poly[key]

            new_poly = self.enforce_BCs(new_poly)
            err = self.check_boundary_conditions(new_poly)
            # import pdb; pdb.set_trace()
            # delta_poly = self.enforce_BCs(delta_poly,input_type="grad")
            # err_delt = self.check_boundary_conditions(self.unstack_vector(self.stack_vector(c_leg_poly)+self.stack_vector(pre_step)))

            bc_count = 0

            while np.abs(err) > 1e-7:
                # Enforce boundary conditions
                # if np.abs(err) > 1e-5:
                print("reinforcing BC for {} iteration. Err is {}".format(bc_count+1,err))
                new_poly = self.enforce_BCs(new_poly)
                err = self.check_boundary_conditions(new_poly)
                bc_count += 1
                if bc_count > 3:
                    print("Can't comply with BCs. err = {}".format(err))
                    break

            # initialise
            first_order_opt = 0.0

            # Loop for each dimension to assign poly_step
            for key in new_poly.keys():
                # Get the step in coefficients
                poly_step[key] = new_poly[key] - c_leg_poly[key]
            # self.analyse_convexity(c_leg_poly,poly_step)
            # # Analyse step
            # if not self.curv_func or len(self.constraint_list)>0:
            #     # import pdb; pdb.set_trace()
            #     self.analyse_convexity(c_leg_poly,poly_step)
            #     # self.c_leg_poly = new_poly
            #     # self.get_trajectory()
            #     # self.plot()
            #     # # plt.show()
            #     # self.c_leg_poly = c_leg_poly
            #     # self.get_trajectory()
            #     import pdb; pdb.set_trace()


            if self.curv_func and use_cost_step:
                # Check cost for small step
                test_poly = c_leg_poly.copy()
                step_size_check = 0.0
                for key in c_leg_poly.keys():
                    # Small step to ensure it decreases the cost
                    test_poly[key] = c_leg_poly[key] + 0.0001*poly_step[key]
                    step_size_check += np.linalg.norm(poly_step[key])
                # Check cost
                new_cost = self.total_cost_grad_curv(test_poly, doGrad=False,doCurv=False)[0]

                print("using cost step times 0.0001. new cost: {}, old cost: {}, steps is {}. max viol is {}".format(new_cost, cost, np.linalg.norm(poly_step[key]),self.max_viol))

                if (step_size_check < self.optim_opt['exit_tol'] or new_cost >= cost): # criteria was 1e-8

                    if self.max_viol > 0.0:
                        # Step too small, or increases cost only (and constraints are violated)
                        # Revert to gradient method
                        new_poly = self.stack_vector(c_leg_poly) - step_size*self.H_inv.dot(self.stack_vector(cost_grad))
                        new_poly = self.unstack_vector(new_poly)
                        new_poly = self.enforce_BCs(new_poly)
                        for key in new_poly.keys():
                            poly_step[key] = new_poly[key] - c_leg_poly[key]
                        print("reverting to gradient method on iteration {}. Max viol is {}".format(self.iterations,self.max_viol))
                        use_cost_step = False
                    else:
                        # TODO (bmorrell@jpl.nasa.gov) CHECK THIS!!!
                        if self.optim_opt['print']:
                            print('Exit: convex step not improving cost')
                        break

            for key in new_poly.keys():
                # Add to get the first order decrease
                first_order_opt += poly_step[key].dot(cost_grad[key])

            # Check first order tolerance
            if np.abs(first_order_opt) < self.optim_opt['first_order_tol']:
                # Decrease is below the tolerance, so exit the Loop
                if not (self.exit_on_feasible and not self.feasible): # Don't break if it is not feasible
                    if self.optim_opt['print']:
                        print('Exit: first order cost decrease in feasible direction '+str(first_order_opt)+' is less than exit tolerance in '+str(self.iterations)+' iterations.')
                    break

            # Check that step is in descent
            if first_order_opt > 0:
                # Reset Hessian to just be gradient descent if not in descending
                self.H_inv = np.identity(self.n_coeff)
                print('Resetting Hessian to identity')

            """ Line Search """

            if use_quadratic_line_search:
                c_leg_poly, out_cost = self.quadratic_line_search(cost,c_leg_poly,poly_step,cost_grad)
            else:
                c_leg_poly, out_cost = self.line_search(cost,c_leg_poly,poly_step,cost_grad)


            c_leg_poly_check = self.enforce_BCs(c_leg_poly)

            for key in c_leg_poly.keys():
                if not np.allclose(c_leg_poly_check[key],c_leg_poly[key]):
                    print("step violating BCs post line search!")

                    c_leg_poly = self.enforce_BCs(c_leg_poly)
                    # import pdb; pdb.set_trace()

            # Compute cost gradient and step for path and constraints
            new_cost, new_cost_grad, cost_curv, cost_step = self.total_cost_grad_curv(c_leg_poly, doGrad=True,doCurv=self.curv_func)

            if self.optim_opt['print']:
                print("Iteration end cost is {}".format(new_cost))

            # Exit if feasible
            if self.exit_on_feasible and self.feasible:
                print("Trajectory is feasible on iteration {}. Exiting with cost: {}".format(self.iterations,new_cost))
                break


            # Update Hessian
            self.update_hessian(cost_grad,new_cost_grad,poly_step)

            # Check for convergence - if cost decrease is sufficiently small
            if np.abs(new_cost - cost)/np.abs(cost) < self.optim_opt['exit_tol'] or new_cost < self.optim_opt['exit_tol']: #TODO add condition that all constraints need to not be violated
                # Exit if the cost change is below tolerance and the maximum constraint violation cost is sufficiently small
                if not (self.exit_on_feasible and not self.feasible): # Don't break if it is not feasible
                    if self.optim_opt['print']:
                        print('Exit: cost difference is less than exit tolerance in '+str(self.iterations)+' iterations.')
                    break

            # Set costs and prepare for next Loop
            cost = new_cost
            cost_grad = new_cost_grad
            self.iterations += 1 # count iterations of outer Loop

            # Check if maximum number of iterations is reached
            if self.iterations > self.optim_opt['max_iterations']:
                print('Exit: iteration limit exceeded')
                break
            if self.iterations > mutate_iter and mutation_run:
                print('Mutation iteration limit reached.returning result')
                break

            # Store tracking data
            if self.optim_opt['track_data']:
                self.data_track.update_data_track(c_leg_poly,cost,cost_grad)


            # Mutate after set number of iterations
            if mutate_serial > 0:
                # Flag active
                if self.iterations > mutate_serial-1:
                    # Hit the iteration limit - mutate the solution
                    print("\n\n Mutating Polynomial after {} iterations".format(self.iterations))
                    c_leg_list = self.mutate_coeffs(self.c_leg_poly,2)
                    c_leg_poly = c_leg_list[1] # Take the second in the list as the first is the non-modified CLegPoly

                    self.mutated = True
                    # Update mutate_serial so it does not mutate on the next iteration
                    if run_one_iteration:
                        self.iterations = 0 # quick hack to reset iterations to zero
                    else:
                        mutate_serial += self.iterations

                    # Reset HEssian to the identity:
                    self.H_inv = np.identity(self.n_coeff)

            # Exit if set to run just one iteration
            if run_one_iteration:
                print("completed one iteration")
                break
            # self.set_yaw_des_from_traj(self.temp_state)

        # Exit from optimization - optimize yaw separately
        if not self.optimise_yaw_only and self.yaw_to_traj:
            state = self.get_trajectory(c_leg_poly)
            self.set_yaw_des_from_traj(state)

            self.optimise_yaw_only = True

            # update yaw components of CLegPoly
            c_leg_poly = self.initial_guess(c_leg_poly)

            # # optimise yaw only
            # print("OPTIMISING FOR YAW NOW")
            # c_leg_poly = self.optimise(c_leg_poly,mutate_iter=100)[1] # note - set to use mutate flags + run for 100 iterations - takes the second output

            self.optimise_yaw_only = False

        # Print out status from optimisation
        if self.optim_opt['print'] and not self.optimise_yaw_only:
            path_cost = self.compute_path_cost_grad(c_leg_poly,doGrad=False,doCurv=False)[0]
            print("\n\nOptimisation complete in {} iterations\nAt end of optimisation:\n  Path Cost is: {}\n  Constraint costs are: ".format(self.iterations,path_cost))
            state = self.get_trajectory(c_leg_poly)
            feasible = True
            esdf_check_all_feasible = False
            for i in range(np.size(self.constraint_list)):
                # Compute cost for each constraint
                constr_cost = self.constraint_list[i].compute_constraint_cost_grad_curv(state,
                                                        self.poly.state_scaled,
                                                        doGrad = False,
                                                        doCurv=self.curv_func,
                                                        path_cost=path_cost)[0]
                print("    Constr {}, type {}: {}".format(i,self.constraint_list[i].constraint_type,constr_cost))

                # Check feasibility - for all constriants
                feasible = feasible and self.check_if_feasible( i)

                if self.constraint_list[i].constraint_type is "esdf_check":
                    esdf_check_all_feasible = self.check_if_feasible(i)

            if esdf_check_all_feasible:
                self.feasible = True
                print("\nAt end of optimisation, trajectory is feasible with ESDF checking\n")
            else:
                if feasible:
                    print("\nAt end of optimisation, trajectory is feasible\n")
                    self.feasible = True
                else:
                    print("\nAt end of optimisation, trajectory is NOT feasible\n")
                    self.feasible = False

        if mutation_run:
            return cost, c_leg_poly
        else:
            print("using final c_leg_poly")
            # On completion of optimisation, update variables
            self.c_leg_poly = c_leg_poly
            self.get_trajectory()
            # if not np.allclose(self.state['x'],state['x']):
            #     print("State computation changed after optimization!!!")
            self.cost = cost
            self.cost_grad = cost_grad
            # self.iterations = iteration

            # Store the time
            self.data_track.optimise_time = time.time() - start_time
            self.data_track.iterations = self.iterations
            # import pdb; pdb.set_trace()
            if hasattr(self,'ax'):
                if self.ax is not None:
                    self.ax.legend()
                plt.show()

            # for i in range(np.size(constr_cost_store)):
            #     print("cost for obstacle {} is {}.".format(i,constr_cost_store[i]))

    def line_search(self,cost,c_leg_poly,poly_step,cost_grad):
        """
        Armijo linesearch to find the best step size in the gradient descent

        Args:
            cost: The total cost value
            c_leg_poly: coefficients to optimise. dictionary for each dimension
                        In each dimension is an np.array storing the coefficients
                        for that dimension across all segments (stacked 1 segment at a time)
            poly_step: The step in the optimisation coefficients (same format as c_leg_poly)
            cost_grad: the cost gradient for the path. A dict for each dimension,
                       in each an np.array or length N
        Uses:
            self.
            optim_opt: Dictionary storing optimisation settings
            H_inv: the Hessian inverse matrice from the BFGS method. 2D np.array squared by N*n_seg*n_dim
            n_coeff: total number of coefficients

        Methods:
            self:
            compute_path_cost_grad:
            data_track.update_inner_data: Variable to track the evolution of the solutions through the linesearch

        Modifies:
            self.
            step_coeff: the coefficient for the final linesearch step size
            data_track: Class storing the data through the optimisation

        Returns:
            poly_target: the c_leg_poly type result at the end of the linesearch
        """

        sigma = self.optim_opt['sigma']
        beta = self.optim_opt['beta']

        # initialise
        step_coeff = 1.0
        exit = False
        inner_iter = 0
        poly_target = dict()

        delta_tmp = -1e9
        target_tmp = dict()

        # Check for exit criteria
        criteria = 0.0

        # Add to the criteria for each dimension
        for key in c_leg_poly.keys():
            criteria += -sigma*(cost_grad[key].dot(poly_step[key]))

        # Start inner optimisation loop
        while not exit:

            # Get the goal polynomial
            # Loop for each dimension
            for key in c_leg_poly.keys():
                poly_target[key] = c_leg_poly[key] + step_coeff*poly_step[key]

            # Compute the cost
            new_cost = self.total_cost_grad_curv(poly_target,doGrad=False,doCurv=False)[0]
            # print("LS. iter {}, cost: {}".format(inner_iter,new_cost))

            # Exit if feasible
            if self.exit_on_feasible and self.feasible:
                print("In Line Seach: traj is feasible, exiting on iteration {}".format(inner_iter+1))
                break

            # Check if the cost reduction is large enough
            if (cost-new_cost) >= criteria*step_coeff:
                # Exit if satisfied
                break

            # Check for maximum number of iterations
            if inner_iter > self.optim_opt['max_armijo']:
                print('\nLarge Armijo Update!\n')
                # Break if exceeded
                # self.analyse_convexity(c_leg_poly,poly_step)
                # import pdb; pdb.set_trace()
                break

            # Check if we are improving
            if cost-new_cost < delta_tmp and inner_iter != 0:
                print("Line Search: Getting worse in line search. Take the last step, with step_coeff = {}".format(step_coeff/beta))
                new_cost = cost_tmp
                for key in poly_target: poly_target[key] = target_tmp[key].copy()
                break
            else:
                delta_tmp = cost - new_cost

            # Store results for potential future use
            cost_tmp = new_cost
            for key in poly_target: target_tmp[key] = poly_target[key].copy()



            # count iterations
            inner_iter += 1

            # Update backtracking coefficient
            step_coeff *= beta

        # Track data
        if self.optim_opt['track_data']:
            self.data_track.update_inner_data(inner_iter,step_coeff)

        # Store final step size
        self.step_coeff = step_coeff # for use in resetting hessian
        print("inner iter: {}. Step: {}".format(inner_iter,step_coeff))
        # output result
        return poly_target, new_cost

    def quadratic_line_search(self,cost,c_leg_poly,poly_step,cost_grad):
        """
        Use information on gradient and cuvature to get the optimal step size
        """

        # Get the gradient
        delta = 0.001 # TODO - revise this value
        steps = [-2*delta,-delta,0.0,delta,2*delta]
        cost_list = []
        poly_target = dict()

        # Get the gradient and curvature around a step size of 1
        for step_size in steps:
            # Get the polynomial
            # Loop for each dimension
            for key in c_leg_poly.keys():
                poly_target[key] = c_leg_poly[key] + (1+step_size)*poly_step[key]

            # Compute the cost
            cost_list.append(self.total_cost_grad_curv(poly_target,doGrad=False,doCurv=False)[0])


        # Compute the gradient
        grad = []
        for i in range(3):
            # Central differencing
            grad.append((cost_list[i+2]-cost_list[i])/(2*delta))

        # Compute the curvature
        curv = (grad[2] - grad[0])/(2*delta)

        # Compute the step
        step_coeff = 1 - grad[1]/curv

        if step_coeff < 0.0:
            print("\nQuadratic line search gives step less than one. Just do backtracking line search\n")
            poly_target, new_cost = self.line_search(cost,c_leg_poly,poly_step,cost_grad)
            return poly_target, new_cost

        if np.isnan(step_coeff):
            print("\nQuadratic line search gives nan step. Just do backtracking line search\n")
            poly_target, new_cost = self.line_search(cost,c_leg_poly,poly_step,cost_grad)
            return poly_target, new_cost

        # Apply step
        for key in c_leg_poly.keys():
            poly_target[key] = c_leg_poly[key] + step_coeff*poly_step[key]


        # Compute the cost
        new_cost = self.total_cost_grad_curv(poly_target,doGrad=False,doCurv=False)[0]

        if new_cost > cost:
            print("\nQuadratic line search bad, doing backtracking line search\n")
            poly_target, new_cost = self.line_search(cost,c_leg_poly,poly_step,cost_grad)
            return poly_target, new_cost

        # plot to analyse
        # poly_step_plot = dict()
        # for key in c_leg_poly.keys(): poly_step_plot[] = step_coeff*poly_step[key]
        # self.analyse_convexity(c_leg_poly,poly_step,plot_range=[0,step_coeff])

        # Store final step size
        self.step_coeff = step_coeff # for use in resetting hessian
        print("quadratic line search: Step: {}".format(step_coeff))
        # output result
        return poly_target, new_cost


    def check_if_feasible(self, i_constraint):
        """ To check if a constraints cost shows it is actually feasible (just inside the inflation region) """
        # quad_buffer = self.constraint_list[i_constraint].quad_buffer
        # inflate_buffer = self.constraint_list[i_constraint].inflate_buffer
        # weight = self.constraint_list[i_constraint].weight
        # cost_type = self.constraint_list[i_constraint].cost_type
        #
        # if cost_type is "squared":
        #     dist = -np.sqrt(cost/weight)
        # else:
        #     dist = -cost/weights
        #
        # true_dist = dist + inflate_buffer
        #
        # return true_dist >= 0

        # if self.constraint_list[i].constraint_type is "cylinder" and not self.constraint_list[i].keep_out and self.esdf_feasibility_checking:
        #     # If useing cylinder keep-in constriants and using the esdf to check feasibility
        #     return self.check_esdf_collision(self.constraint_list[i].esdf)


        print("Checking constraint feasibility for constriant: {}. Is {}".format(i_constraint,self.constraint_list[i_constraint].feasible))

        return self.constraint_list[i_constraint].feasible

    def check_esdf_collision(self,esdf,state,quad_buffer,seg=None):

        # get x, y, z trajectories
        x = state['x'][0,:,seg]
        y = state['y'][0,:,seg]
        z = state['z'][0,:,seg]

        # Create query points
        query = np.matrix(np.zeros([3,x.size]),dtype='double')
        dist = np.matrix(np.zeros((np.shape(query)[1],1)),dtype='double')
        obs = np.matrix(np.zeros((np.shape(query)[1],1)),dtype='int32')

        # load in x, y, z points
        query[0,:] = np.around(x,4)
        query[1,:] = np.around(y,4)
        query[2,:] = np.around(z,4)

        # Query the database
        # if doGrad:
        grad = np.matrix(np.zeros(np.shape(query)),dtype='double')
        esdf.getDistanceAndGradientAtPosition(query, dist, grad, obs)

        dist -= quad_buffer

        dist[obs!=1] = 0.0

        return np.min(dist) >= 0.0

    def time_opt_cost(self,indices=[None],new_times=None,defer=True):
        """ Computing cost for the outer loop time optimisation """

        if indices[0] is not None:
            t_vec = self.times.copy()
            c_leg_poly = self.c_leg_poly.copy() # Store the coefficients
            self.update_times(indices, new_times, new_coeffs=False, defer=defer)

        # Compute cost
        traj_cost = self.total_cost_grad_curv(self.c_leg_poly, doGrad=False,doCurv=False)[0]

        # Compute combined cost
        cost = traj_cost + self.time_penalty*np.sum(self.times)

        # Reset times:
        if indices[0] is not None:
            self.update_times(np.arange(0,self.n_seg,1), t_vec, new_coeffs=False, defer=True)
            # Reapply the original coefficients (enforcing BCs back and forth modify the coefficients)
            self.c_leg_poly = c_leg_poly.copy()

        return cost

    def time_opt_gradient(self,times,defer=True):

        del_t = self.del_t

        # Original time
        t_vec = times

        # Initialise
        grad = np.zeros(self.n_seg)

        # Save the starting cost
        # cost_start = self.time_opt_cost(np.where(-np.isclose(self.times, t_vec))[0],t_vec)
        cost_start = self.time_opt_cost(np.arange(0,self.n_seg,1),t_vec,defer=defer)

        # Perturb for each segment
        for i in range(0,self.n_seg):
            # perturbed time
            t_change = t_vec.copy()
            t_change[i] += del_t
            # Cost from perturbed time
            # if i == 0:
            #     # Change just the first term
            #     new_cost = self.time_opt_cost(np.atleast_1d(i),np.atleast_1d(t_change[i]),defer=defer)
            # else:
            #     # reset the previous term as well
            #     new_cost = self.time_opt_cost(np.array([i-1,i]),t_change[i-1:i+1],defer=defer)
            new_cost = self.time_opt_cost(np.arange(0,self.n_seg,1),t_change,defer=defer)

            # Gradient - forward differencing
            grad[i] = (new_cost-cost_start)/del_t

        # # Reset times:
        # self.update_times(np.arange(0,self.n_seg,1), t_vec, new_coeffs=False, defer=True)

        return grad

    def time_optimisation(self,max_iter=100,run_snap_opt=False,use_scipy=True):
        """

            run_snap_opt: a flag to select whether or not to run the inner snap optimisation loop

        """
        start_timer = time.time()

        step_size_time = 1.0/self.time_penalty#1.0#self.step_size_time
        beta = self.optim_opt['beta']
        exit_tol_time = self.exit_tol_time
        coeff = 1.0
        exit_flag = False
        inner_exit_flag = False
        t_min = 0.00001
        iter_limit = max_iter

        # Save the starting cost and grad
        cost_start = self.time_opt_cost()
        # Initialise
        old_cost = cost_start.copy()
        t_vec = self.times.copy()
        grad = self.time_opt_gradient(t_vec,not run_snap_opt)
        iter_count = 0

        if use_scipy:
            cons = ({'type': 'ineq',
                     'fun': lambda x: (np.array(x) -
                                       np.ones((np.size(x))) * t_min)})



            cost_wrapper = utils.InputWrapper(self.time_opt_cost,self.times,defer=not run_snap_opt)
            res = sp.optimize.minimize(cost_wrapper.wrap_func, self.times,
                                        method='COBYLA',
                                        constraints=cons,
                                        options=dict(disp=3,maxiter=max_iter))


            t_vec = res.x
            converged = res.success
            iter_count = res.nfev
        else:

            # Outer loop
            while not exit_flag:
                # linesearch
                while not inner_exit_flag:
                    # Get step
                    step = - step_size_time*coeff*grad

                    # Apply step
                    t_new = t_vec + step

                    t_new[t_new<=0] = t_min
                    # step = t_new - t_vec

                    # Get new cost
                    new_cost = self.time_opt_cost(np.arange(0,self.n_seg),t_new, defer=not run_snap_opt)

                    # Check for exist criteria - is cost reduction large enough
                    if (old_cost - new_cost) >= -coeff*step_size_time*grad.dot(step):
                        inner_exit_flag = True
                    else:
                        coeff *= beta


                # Compute the new gradient
                grad = self.time_opt_gradient(t_new,not run_snap_opt)

                # Reset for next loop
                inner_exit_flag = False
                t_vec = t_new.copy()
                coeff = 1.0

                if np.abs(new_cost - old_cost)/np.abs(old_cost) < exit_tol_time:
                    exit_flag = True
                    converged = True

                # Reset cost
                old_cost = new_cost.copy()

                iter_count += 1

                if iter_count > iter_limit:
                    exit_flag = True
                    converged = False
                    print("Iteration Limit Hit")


        # Optimisation finished
        # Apply optimal times.
        self.update_times(np.arange(self.n_seg), t_vec, new_coeffs=False, defer=not run_snap_opt)

        self.data_track.outer_opt_time = time.time() - start_timer

        return converged, iter_count

    def mutate_optimise(self):
        """
        Optimises the trajectory using mutations (random variations) of the initial solution to explore
        a range of possible local minima.
        Settings control how many mutations to run and for how many interations before selecting
        the best candidate
        The best candidate is then optimised

        Args:



        Uses:
            self.
            c_leg_poly
            everything in self.optimise()


        Modifies:
            self.
            c_leg_poly
            everything in self.optimise()

        """

        # Temporary placeholders
        n_mutate = 4
        mutate_iter = 4

        # Initial polynomial - so will by default replan
        c_leg_poly = self.c_leg_poly

        # mutate the polynomial coefficients
        c_leg_mutate_list = self.mutate_coeffs(c_leg_poly,n_mutate)
        cost = np.zeros(len(c_leg_mutate_list))

        # Run each mutated polynomial for set restricted number of iterations
        for i in range(len(c_leg_mutate_list)):
            # optimise current mutation
            cost[i], c_leg_mutate_list[i] = self.optimise(c_leg_mutate_list[i],mutate_iter)

        # Assess the best cost
        win_index = np.where(cost==np.min(cost))[0][0]
        print("cost list is: {}\nWinning cost is {}".format(cost,win_index))

        # Assign best candidate to c_leg_poly
        self.c_leg_poly = c_leg_mutate_list[win_index].copy()

        # Complete optimisation with new c_leg_poly
        self.optimise()

        # Generate the trajectory states
        self.get_trajectory()

    def mutate_coeffs(self,c_leg_poly,n_mutate):
        """
        Mutated the input coefficient set for n_mutate different coefficients
        sets to explore different local minima

        Args:
            c_leg_poly: coefficients to optimise. dictionary for each dimension
                    In each dimension is an np.array storing the coefficients for that dimension across all segments (stacked 1 segment at a time)
            n_mutate: number of mutations to generate

        Returns:
            c_leg_mutate_list: a list of c_leg poly dicts for each mutation

        """

        # Placeholder
        mutation_strength = 5.0

        # initialise list
        c_leg_mutate_list = [c_leg_poly.copy(),]

        # Loop for number of mutations desired
        for i in range(n_mutate-1):

            # initialise mutation
            c_leg_mut = dict()

            # Add a random variation for each dimension
            for key in ['x','y','z']:#c_leg_poly.keys(): Ignore yaw for the moment
                # Generate the random spatial variation
                c_step = (np.random.rand(np.size(c_leg_poly[key]),1)-0.5)*mutation_strength

                # Apply step
                c_leg_mut[key] = c_leg_poly[key].copy() + c_step[:,0]

            c_leg_mut['yaw'] = c_leg_poly['yaw'].copy()

            # Enforce boundary conditions
            c_leg_mut = self.enforce_BCs(c_leg_mut)

            # Add to list
            c_leg_mutate_list.append(c_leg_mut.copy())


        return c_leg_mutate_list



    def add_constraint(self,constraint_type,params,dynamic_weighting=False,sum_func=True,custom_weighting=False):
        """
        Adding a constraint to the constraint array

        """
        print("Obstacle params are: {}".format(params))

        if constraint_type == "ellipsoid":
            weight = params['weight']
            keep_out = params['keep_out']
            der = params['der']
            x0 = params['x0']
            A = np.array(params['A'])
            rot_mat = params['rot_mat']

            self.constraint_list.append((constraint.ellipsoid_constraint(weight,keep_out,der,x0,A,rot_mat,
                                    dynamic_weighting=dynamic_weighting,doCurv=self.curv_func,
                                    sum_func=sum_func,custom_weighting=custom_weighting)))

            self.exit_on_feasible = True
        elif constraint_type == "cylinder":
            weight = params['weight']
            keep_out = params['keep_out']
            der = params['der']
            x1 = params['x1']
            x2 = params['x2']
            r = params['r']
            l = params['l']
            active_seg = params['active_seg']

            self.constraint_list.append((constraint.cylinder_constraint(weight,keep_out,der,x1,x2,r,l=l,
                                            active_seg=active_seg,dynamic_weighting=dynamic_weighting,
                                            doCurv=self.curv_func,sum_func = sum_func,custom_weighting=custom_weighting)))

        else:

            print("Error: invalid constraint type input: {}".format(constraint_type))

        self.feasible = False

    def remove_constraint(self,index):

        self.constraint_list.pop(index)

    def remove_corridor_constraints(self):

        if np.size(self.constraint_list) == 0:
            return

        count = 0
        remove_list = []

        for constraint in self.constraint_list:
            if constraint.constraint_type is "cylinder" and not constraint.keep_out:
                # An existing corridor constraint
                remove_list.append(count)
            count += 1

        print("removing corridor constraints, in list with indices: {}".format(remove_list))

        count = 0
        for index in remove_list:
            self.remove_constraint(index - count)
            count += 1

    def remove_esdf_constraint(self):

        i = 0

        while i < np.size(self.constraint_list):
            if self.constraint_list[i].constraint_type is "esdf":
                self.constraint_list.pop(i)
            else:
                i = i + 1

    def add_esdf_constraint(self,esdf,weight=1e9,quad_buffer=0.0,inflate_buffer=0.0,dynamic_weighting=False,sum_func=False,feasibility_checker=False,custom_weighting=True):
        """
        Add a constriant that uses an esdf map
        the esdf is of the voxblox format
        """
        # Set flag to exit when a feasible trajectory is obtained
        self.exit_on_feasible = True
        self.feasible = False
        if feasibility_checker:
            self.esdf_feasibility_check = True
        self.constraint_list.append((constraint.esdf_constraint(weight,esdf,quad_buffer,inflate_buffer,dynamic_weighting=dynamic_weighting,sum_func=sum_func,feasibility_checker=feasibility_checker,custom_weighting=custom_weighting)))

    def set_constraint_weight(self,weight,constraint_type):

        for i in range(np.size(self.constraint_list)):
            if self.constraint_list[i].constraint_type is constraint_type:
                print("constraint {} updated with weight {} from {}".format(i,weight,self.constraint_list[i].weight))
                # update weight
                self.constraint_list[i].weight = weight

    def convert_poly_to_simple_basis(self):
        """
        Converts from Legendre basis to a simple polynomial basis, with basis
        functions t^0, t^1, t^2, ... t^n

        Uses:
            self.
            pol_leg: 3D np array representing the Legendre polynomail basis functions with
                dim1: basis functions (Legendre base)
                dim2: coefficients
                dim3: derivatives
            order:

        Output
            new_coeffs: Coefficients of the polynomial with basis functions of t^0,
                        t^1, t^2, ... t^N. an np.array
                dim1: coefficents for each basis (highest power first)
                dim2: segment

        """
        new_coeffs = dict()


        for key in self.pol_leg.keys():
            # Reshape the pol_leg matrix for position to have basis polynomials
            # in the columns, and the rows linked with powers of t (t^n at the top)
            comp_mat = self.pol_leg[key][:,:,0].T.copy()
            # Number of coefficients
            N = self.order[key]+1
            # Initialise
            new_coeffs[key] = np.zeros([N,self.n_seg])

            # For each segment
            for i in range(0,self.n_seg):
                # multiply each Legendre basis by the coefficient, and add together
                # the contributions for each power of t.
                # Order with highest power first. Each column is a new segment.
                new_coeffs[key][:,i] = comp_mat.dot(self.c_leg_poly[key][N*i:N*(i+1)])

                # Scale by times
                new_coeffs[key][:,i] = new_coeffs[key][:,i] * (self.times[i]/2)**(self.n_der[key]-1)

                # Change time bases from -1->1 to 0->t_f
                time_rescale_poly = np.poly1d([2/self.times[i],-1.0])
                temp_poly = np.poly1d(new_coeffs[key][:,i])
                temp_poly = temp_poly(time_rescale_poly)

                ncoeff = np.size(temp_poly.coeffs)

                new_coeffs[key][-ncoeff:,i] = temp_poly.coeffs.copy()

        return new_coeffs

    def fill_poly_traj_msg(self, msg):
        """
        Fills a polytraj message to be sent over ROS, using a simple t^0, t^1, t^2 ...
        basis.

        Uses:
            coeffs: from the convert_poly_to_simple_basis function. Coefficients
                    of the polynomial with basis functions of t^0, t^1, t^2, ... t^N.
                    An np.array
                dim1: coefficents for each basis (highest power first)
                dim2: segment
            self.n_seg

        Modifies:
            msg: a struct formatted to get the required information for sending
                the trajectory over ROS

        """

        # Get the coefficients with a simple basis
        coeffs = self.convert_poly_to_simple_basis()

        msg.n_segs = self.n_seg

        msg.n_x_coeffs = np.shape(coeffs['x'])[0] # number of coefficients per segment for x
        msg.n_y_coeffs = np.shape(coeffs['y'])[0] # number of coefficients per segment for y
        msg.n_z_coeffs = np.shape(coeffs['z'])[0] # number of coefficients per segment for z
        msg.n_yaw_coeffs = np.shape(coeffs['yaw'])[0] # number of coefficients per segment for yaw

        # polynomial coefficients, segment by segment, starting at highest order coefficient
        msg.x_coeffs[0:np.size(coeffs['x'])] = coeffs['x'].T.flatten().tolist()
        msg.y_coeffs[0:np.size(coeffs['y'])] = coeffs['y'].T.flatten().tolist()
        msg.z_coeffs[0:np.size(coeffs['z'])] = coeffs['z'].T.flatten().tolist()
        msg.yaw_coeffs[0:np.size(coeffs['yaw'])] = coeffs['yaw'].T.flatten().tolist()

        # transition times, cumulative, segment by segment
        trans_times = utils.seg_times_to_trans_times(self.times)
        msg.t_trans[0:(self.n_seg)] = trans_times[1:]

    def create_callable_ppoly(self):
        """
        Creates a ppoly object that can be called and is in the same format
        as UNCO - basis polynomials as described below.

        Uses:
            coeffs: from the convert_poly_to_simple_basis function. Coefficients
                    of the polynomial with basis functions of t^0, t^1, t^2, ... t^N.
                    An np.array
                dim1: coefficents for each basis (highest power first)
                dim2: segment
            self.n_seg

        Modifies:
            msg: a struct formatted to get the required information for sending
                the trajectory over ROS

        """

        # Get the coefficients with a simple basis
        coeffs = self.convert_poly_to_simple_basis()

        ppoly = dict()

        trans_times = utils.seg_times_to_trans_times(self.times)

        for key in coeffs.keys():
            ppoly[key] = sp.interpolate.PPoly(coeffs[key], trans_times, extrapolate=False)

        return ppoly


    def plot(self,azim=-22.,elev=20.):
        """
        Plot the trajectory and waypoints in 3D
        Currently only a 3D trajectory plot

        Args:
            azim: the azimuth for the view setting
            elev: the elevation for the view setting

        Uses:
            self.
            state_combined: dict for each dim, samples in a stack for each segment (np.array([n_der,n_samp*n_seg]))
            waypoints: dict with waypoints for each dimension (np.array([n_der,n_waypoints]))
        """

        # values
        x = self.state_combined['x'][0,:]
        y = self.state_combined['y'][0,:]
        z = self.state_combined['z'][0,:]

        waypoints = self.waypoints

        # PLOT FIGURE
        if self.fig is None:
            fig = plt.figure(figsize=(10,10))
            fig.suptitle('Static 3D Plot', fontsize=20)
            ax = Axes3D(fig)#fig.add_subplot(111, projection='3d')
            ax.set_xlabel('x (m)', fontsize=14)
            ax.set_ylabel('y (m)', fontsize=14)
            ax.set_zlabel('z (m)', fontsize=14)
            self.plot_index = 0
            self.ax = ax
            self.fig = fig
        else:
            fig = self.fig
            ax = self.ax

        # plot Trajectory
        label = "Traj at iter {}".format(self.plot_index)
        ax.plot(x,y,z, label=label)

        # plot waypoints
        ax.scatter(waypoints['x'][0,:],
                    waypoints['y'][0,:],
                    waypoints['z'][0,:],
                    label='Waypoints')
        ax.axis('equal')
        ax.view_init(azim=azim,elev=elev)

        self.plot_index += 1
        # plt.show()
        return fig

    def plot_yaw(self):
        """
        Plot the yaw trajectory and waypoints in 3D

        Args:
            azim: the azimuth for the view setting
            elev: the elevation for the view setting

        Uses:
            self.
            state_combined: dict for each dim, samples in a stack for each segment (np.array([n_der,n_samp*n_seg]))
            waypoints: dict with waypoints for each dimension (np.array([n_der,n_waypoints]))
        """

        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from mpl_toolkits.mplot3d import Axes3D

        # values
        yaw = self.state_combined['yaw'][0,:]*180/np.pi

        waypoints = self.waypoints

        # PLOT FIGURE
        fig = plt.figure(figsize=(10,10))
        fig.suptitle('Static 3D Plot', fontsize=20)
        ax = fig.add_subplot(111, projection='rectilinear')
        ax.set_ylabel('yaw (deg)', fontsize=14)
        ax.set_xlabel('sample', fontsize=14)

        # plot Trajectory
        ax.plot(yaw)# label='Trajectory')

        # # plot waypoints
        # ax.scatter(waypoints['x'][0,:],
        #             waypoints['y'][0,:],
        #             waypoints['z'][0,:],
        #             label='Waypoints')
        # ax.axis('equal')
        # ax.view_init(azim=azim,elev=elev)
        # plt.show()
        return fig

    def analyse_convexity(self,c_leg_poly,poly_step,plot_range=None):

        n_test = 15

        if plot_range is not None:
            step_sizes = np.linspace(plot_range[0],plot_range[1],n_test)
        else:
            # step_sizes = np.linspace(-2.0e-7,8.0e-7,n_test)
            # step_sizes = np.linspace(-2.0,2.0,n_test)
            # step_sizes = np.linspace(-0.5,2.0,n_test)
            step_sizes = np.linspace(-0.0,1.0,n_test)
            # step_sizes = np.linspace(-1.0e-3,1.0e-3,n_test)

        costs = np.zeros(n_test)

        start_c = self.stack_vector(c_leg_poly)
        step_c = self.stack_vector(poly_step)

        for i in range(n_test):
            test_poly = self.unstack_vector(start_c + step_sizes[i]*step_c)
            # costs[i] = self.total_cost_grad_curv(test_poly, doGrad=True,doCurv=self.curv_func)[0]
            costs[i] = self.total_cost_grad_curv(test_poly, doGrad=False,doCurv=False)[0]

        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from mpl_toolkits.mplot3d import Axes3D

        # plt.style.use('bmh')
        plt.style.use('seaborn-paper')


        # PLOT FIGURE
        fig = plt.figure(figsize=(12,6))
        # fig.suptitle('Cost Change', fontsize=20)
        ax = fig.add_subplot(111)
        ax.set_xlabel('Step size', fontsize=16,fontweight='bold')
        ax.set_ylabel('Cost', fontsize=16,fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        # plot Trajectory
        ax.plot(step_sizes,costs,linewidth=3.0)
        plt.grid()
        plt.show()

        return fig

    def numerical_gradient(self,c_leg_poly):

        # epsilon = 1e-1 # for esdf
        epsilon = 1e-5
        grad = dict()

        test1 = c_leg_poly.copy()
        test2 = c_leg_poly.copy()
        # sp.optimize.approx_fprime(c_leg_poly, f, epsilon, *args)[source]
        for key in c_leg_poly.keys():
            test1[key] = c_leg_poly[key].copy()
            test2[key] = c_leg_poly[key].copy()
            N = np.size(c_leg_poly[key])
            grad[key] = np.zeros(N)

            for i in range(0,N):
                test1[key][i] += epsilon
                test2[key][i] -= epsilon

                cost_f = self.total_cost_grad_curv(test1, doGrad=False,doCurv=False,num_grad=False)[0]
                cost_r = self.total_cost_grad_curv(test2, doGrad=False,doCurv=False,num_grad=False)[0]

                grad[key][i] = (cost_f-cost_r)/(2*epsilon)

                # Reset
                test1[key][i] -= epsilon
                test2[key][i] += epsilon

        return grad

    def numerical_curvature(self,c_leg_poly):

        epsilon = 1e-5
        grad = dict()

        test1 = c_leg_poly.copy()
        # test2 = c_leg_poly.copy()
        # sp.optimize.approx_fprime(c_leg_poly, f, epsilon, *args)[source]

        cost_r, grad_r = self.total_cost_grad_curv(test1, doGrad=False,doCurv=False)[0:2]

        for key in c_leg_poly.keys():
            test1[key] = c_leg_poly[key].copy()
            # test2[key] = c_leg_poly[key].copy()
            N = np.size(c_leg_poly[key])
            curv[key] = np.zeros((N,N))

            for i in range(0,N):
                test1[key][i] += epsilon
                # test2[key][i] -= epsilon

                cost_f, grad_f = self.total_cost_grad_curv(test1, doGrad=False,doCurv=False)[0:2]
                # cost_r = self.total_cost_grad_curv(test2, doGrad=False,doCurv=False)[0]

                curv[key][:,i] = (grad_f[key]-grad_r[key])/(epsilon)

                # Reset
                test1[key][i] -= epsilon
                test2[key][i] += epsilon

        return curv



def main():
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D

    waypoints = dict()
    waypoints['x'] = np.zeros([5,4])
    waypoints['y'] = np.zeros([5,4])
    waypoints['z'] = np.zeros([5,4])
    waypoints['yaw'] = np.zeros([3,4])

    waypoints['x'][0,:] = np.array([1.,0.5,-0.5,-1.0])
    waypoints['y'][0,:] = np.array([0.5,-0.5,0.5,-0.5])
    waypoints['z'][0,:] = np.array([-0.5,-0.5,0.5,0.5])
    waypoints['yaw'][0,:] = np.array([-0.5,0.5,-0.5,0.5])

    # waypoints['x'][0,:] = np.array([0.,0.0,-2.0,-2.0])
    # waypoints['y'][0,:] = np.array([0.,1.0,1.0,0.])
    # waypoints['z'][0,:] = np.array([0.0,0.0,0.0,0.0])

    traj = traj_qr(waypoints,seed_avg_vel=0.5,curv_func=True)#,order=dict(x=12, y=12, z=12, yaw=5))
    import pdb; pdb.set_trace()
    traj.run_astro()

    ppoly = traj.create_callable_ppoly()

    import pdb; pdb.set_trace()

    # fig = traj.plot()
    # plt.show()
    #

    """ Cylinder constriant """
    # constraint_type = "cylinder"
    # params = dict()
    # # params['weight'] = 1e5
    # # params['keep_out'] = True
    # params['der'] = 0
    # # params['x1'] = np.array([0.75,1.0,-0.500001])
    # # params['x2'] = np.array([0.75,-1.0,-0.500001])
    # # params['r'] = 0.3
    # params['l'] = 0.5
    #
    # # params['r'] = 0.7
    # # params['weight'] = 1.0
    # # params['x1'] = np.array([-10.,0.0,-0.00001])
    # # params['x2'] = np.array([10.,-0.0,-0.01])
    # # params['keep_out'] = False
    # # params['active_seg'] = 1
    #
    # params['r'] = 1.0
    # params['weight'] = 100
    # params['x1'] = np.array([-3.0,0.5,0.0])
    # params['x2'] = np.array([1.0,0.5,0.0])
    # params['keep_out'] = False
    # params['active_seg'] = 1
    #
    # traj.add_constraint(constraint_type,params,dynamic_weighting=True,sum_func=False)
    #
    # traj.run_astro()
    #
    # fig = traj.plot()
    #
    # plt.show()
    # import pdb; pdb.set_trace()


    """ vvv """
    constraint_type = "ellipsoid"
    params = dict()
    params['weight'] = 1e5
    params['keep_out'] = True
    params['der'] = 0
    params['x0'] = np.array([0.75,0.0,-0.500001])
    A = np.identity(3)
    A[0,0] = 1/0.2**2
    A[1,1] = 1/0.5**2
    A[2,2] = 1/0.3**2
    params['A'] = A
    params['rot_mat'] = np.identity(3)

    traj.add_constraint(constraint_type,params,dynamic_weighting=True,sum_func=False)

    traj.mutate_optimise()

    traj.get_trajectory()
    #import pdb; pdb.set_trace()
    # traj.run_astro()
    fig = traj.plot()
    fig2 = traj.plot_yaw()
    plt.show()

    import pdb; pdb.set_trace()
    """ Test Constraints """
    # waypoints = dict()
    # der_fixed = dict()
    # waypoints['x'] = np.zeros([5,2])
    # waypoints['y'] = np.zeros([5,2])
    # waypoints['z'] = np.zeros([5,2])
    # waypoints['yaw'] = np.zeros([3,2])
    #
    # waypoints['x'][0,:] = np.array([-1.0,1.0])#,-1.0])
    # waypoints['y'][0,:] = np.array([0.5,0.5])#,0.4])
    # waypoints['z'][0,:] = np.array([-0.5,-0.5])#,-0.5])
    #
    # der_fixed['x'] = np.zeros([5,3],dtype=bool)
    # der_fixed['y'] = np.zeros([5,3],dtype=bool)
    # der_fixed['z'] = np.zeros([5,3],dtype=bool)
    # der_fixed['yaw'] = np.zeros([3,3],dtype=bool)
    #
    # der_fixed['x'][0,:] = True
    # der_fixed['y'][0,:] = True
    # der_fixed['z'][0,:] = True
    # der_fixed['yaw'][0,:] = True
    #
    # traj = traj_qr(waypoints,seed_avg_vel=1.0,curv_func=False)
    # # traj = traj_qr(waypoints,der_fixed=der_fixed,seed_avg_vel=1.0)
    #
    # for key in traj.der_fixed.keys():
    #     traj.der_fixed[key][1:,:] = False
    #
    # traj.reset_astro()
    # traj.run_astro()
    # import pdb; pdb.set_trace()
    # # import pickle
    # # out_filename = "two_simple_astro_waypoints.pickle.traj"
    # # with open(out_filename, "wb") as filp:
    # #     pickle.dump(traj, filp, 2)
    # #
    # #
    # # with open(out_filename, 'rb') as f:
    # #     qr_polytraj = pickle.load(f)
    # #     import pdb; pdb.set_trace()
    # #     if hasattr(qr_polytraj,'c_leg_poly'):
    # #         self.qr_polytraj = qr_polytraj
    #
    constraint_type = "ellipsoid"
    params = dict()
    params['weight'] = 1
    params['keep_out'] = True
    params['der'] = 0
    params['x0'] = np.array([0.0,0.500001,-0.500001])

    params['x0'] = np.array([0.0,0.0,0.])
    params['keep_out'] = False
    params['der'] = 2

    A = np.identity(3)
    lim = 1.0
    A[0,0] = 1/lim**2
    A[1,1] = 1/lim**2
    A[2,2] = 1/lim**2
    params['A'] = A
    params['rot_mat'] = np.identity(3)

    traj.add_constraint(constraint_type,params,sum_func=True)
    #
    # # params['x0'] = np.array([0.0,0.700001,-0.300001])
    # params['keep_out'] = True
    # params['der'] = 0
    # params['x0'] = np.array([0.0,0.500001,-0.500001])
    #
    # traj.add_constraint(constraint_type,params)
    #
    # import pdb; pdb.set_trace()
    # cost, cost_grad, cost_curv, cost_step = traj.total_cost_grad_curv(traj.c_leg_poly, doGrad=True,doCurv=False)
    #
    import pdb; pdb.set_trace()
    traj.run_astro()



    # grad = traj.numerical_gradient(traj.c_leg_poly)

    # import pdb; pdb.set_trace()
    #
    # fig = traj.plot()
    #
    # plt.show()
    #
    #
    ax = traj.state_combined['x'][2,:]
    ay = traj.state_combined['y'][2,:]
    az = traj.state_combined['z'][2,:]
    max_a = np.amax(np.sqrt(ax**2+ay**2+az**2))
    print("Max Accel is {}\n".format(max_a))

    err = traj.check_boundary_conditions(traj.c_leg_poly)
    print("BC error is {}".format(err))
    import pdb; pdb.set_trace()

    traj.time_optimisation(max_iter=200)
    ax = traj.state_combined['x'][2,:]
    ay = traj.state_combined['y'][2,:]
    az = traj.state_combined['z'][2,:]
    max_a = np.amax(np.sqrt(ax**2+ay**2+az**2))
    print("Max Accel is {}\n".format(max_a))
    import pdb; pdb.set_trace()
    traj.get_trajectory()
    constr_cost, constr_cost_grad, constr_cost_curv = traj.constraint_list[0].compute_constraint_cost_grad_curv(traj.state, traj.poly.state_scaled, doGrad = True, doCurv=False)

    # traj.generate_legendre_poly()
    #
    # traj.poly.scale_matrices(traj.tf)
    #
    # traj.create_P_BC()
    #
    # traj.initial_guess()
    #
    # traj.compute_projected_matrix()
    #
    # traj.compute_path_cost_grad(traj.c_leg_poly,True)
    #
    # c_leg_poly_stacked = traj.stack_vector(traj.c_leg_poly)
    #
    # c_leg_poly_unstacked = traj.unstack_vector(c_leg_poly_stacked)
    #
    # c_leg_poly_proj = traj.enforce_BCs(traj.c_leg_poly)
    #
    # traj.optimise()
    #
    # traj.get_trajectory()

    """ Testing waypoint insertion/deletion """
    # waypoints['x'] = np.array([1.,0.5,-0.5,-1.0])
    # waypoints['y'] = np.array([0.5,-0.5,0.5,-0.5])
    # waypoints['z'] = np.array([-0.5,-0.5,0.5,0.5])
    # waypoints['yaw'] = np.array([0,0,0,0])
    #
    # traj2 = traj_qr(waypoints)
    #
    # print("Cost is: {}.\n Exited in {} iterations".format(traj.cost,traj.iterations))
    #
    # new_waypoint = dict()
    # for key in waypoints.keys():
    #     new_waypoint[key] = 2.0
    #
    # index = 2
    # time1 = traj.times
    # new_times = [time1[index-1]/2,time1[index-1]/2]
    # # import pdb; pdb.set_trace()
    # a1 = traj.c_leg_poly
    # traj.insert_waypoint(index,new_waypoint,new_times,defer=False)
    # traj.delete_waypoint(index,time1[index-1],defer=False)
    # a2 = traj.c_leg_poly
    #
    # time2 = traj.times

    """ Testing conversion to format for sending messages to the drone """
    states = traj.state
    # import pdb; pdb.set_trace()
    new_coeffs = traj.convert_poly_to_simple_basis()
    trans_times = utils.seg_times_to_trans_times(traj.times)

    ppoly_new = dict()
    new_states = dict()

    # t = np.linspace(trans_times[0], trans_times[-1], traj.n_samp*traj.n_seg)
    import pdb; pdb.set_trace()
    t = np.array([])
    for i in range(traj.n_seg):
        t = np.concatenate([t,np.linspace(trans_times[i], trans_times[i+1], traj.n_samp)])
    # import pdb; pdb.set_trace()

    for key in traj.c_leg_poly.keys():
        ppoly_new[key] = sp.interpolate.PPoly(new_coeffs[key],trans_times, extrapolate=False)
        new_states[key] = ppoly_new[key](t)
        for k in range(states[key].shape[2]):
            np.testing.assert_allclose(states[key][0,:,k],new_states[key][traj.n_samp*k:traj.n_samp*(k+1)])

    key='y'
    err = states[key][0,:,0] - new_states[key][:traj.n_samp]
    err2 = states[key][0,:,1] - new_states[key][traj.n_samp:traj.n_samp*2]
    err3 = states[key][0,:,2] - new_states[key][2*traj.n_samp:]

    np.max(np.abs(err3))
    import pdb; pdb.set_trace()
    test = np.poly1d([2,-1.0])
    test2 = ppoly_new['x'](test)


    from px4_msgs.msg import PolyTraj as PolyTraj_msg
    msg = PolyTraj_msg()
    traj.fill_poly_traj_msg(msg)

    new_traj = np.polyval(new_coeffs['x'][:,0],traj.tsteps)

    traj.get_trajectory()





    """ Testing Time of P_BC """
    # indices = np.atleast_1d([0,2])
    # old_times = traj.times
    # traj.times = old_times/2
    #
    # expect = traj.c_leg_poly
    # traj.rescale_P_BC(indices,old_times)
    # traj.initial_guess()
    # inter = traj.c_leg_poly
    #
    # old_times = traj.times
    # traj.times = old_times*2
    # traj.rescale_P_BC(indices,old_times)
    # traj.initial_guess()
    # result = traj.c_leg_poly

    """ Testing update time """
    indices = np.atleast_1d([0])#,1,2])
    old_times = traj.times.copy()
    new_times = old_times[indices]/2
    # # import pdb; pdb.set_trace()
    # old_path_cost = traj.compute_path_cost_grad(traj.c_leg_poly,False)[0]
    # traj.update_times(indices, new_times, new_coeffs=False, defer=True)
    # new_path_cost = traj.compute_path_cost_grad(traj.c_leg_poly,False)[0]

    traj.time_penalty = 1e5
    """ Outer loop opt testing """
    # STarting cost
    # cost=traj.time_opt_cost()
    # grad = traj.time_opt_gradient()
    # import pdb; pdb.set_trace()

    old_cost = traj.time_opt_cost()
    old_path_cost = traj.compute_path_cost_grad(traj.c_leg_poly,False)[0]

    # optimise
    import time
    start_time = time.time()
    converged, iter_count = traj.time_optimisation(run_snap_opt=False,use_scipy=True)


    # new_path_cost = traj.compute_path_cost_grad(traj.c_leg_poly,False)[0]
    # time_opt_time = traj.times.copy()

    traj.optimise()
    total_time = time.time()-start_time

    opt_cost = traj.time_opt_cost()
    opt_path_cost = traj.compute_path_cost_grad(traj.c_leg_poly,False)[0]

    print("Optimal cost: {}\nPath Cost: {}\nNumber of iterations: {}\nTime: {}\nSolution: {}".format(opt_cost,opt_path_cost,iter_count,total_time,traj.times))

    import pdb; pdb.set_trace()

    fig = traj.plot()
    plt.show()


if __name__ == '__main__':
    main()
