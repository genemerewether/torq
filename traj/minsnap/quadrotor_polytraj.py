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

from pprint import pprint as pp
import yaml
import argparse
import os
import sys
import time

import pickle

import numpy as np
import scipy
import scipy.optimize
sp = scipy

from minsnap import exceptions
from minsnap import utils
from minsnap import joint_optimize
import minsnap.settings

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


class QRPolyTraj(object):
    """Quadrotor polynomial trajectory in x, y, z, yaw"""

    def __init__(self, partial_waypoints, time_penalty, costs=None, der_fixed=None, der_ineq=None,
                 order=dict(x=9, y=9, z=5, yaw=5),
                 seed_avg_vel=1.0, seed_times=None, face_front=None, delta=None, yaw_eps=0.1,
                 inner_loop_callback=None, restrict_freespace=False, l_max=5.0, A_max=12.0,
                 closed_loop=False,set_yaw_to_traj=False,set_yaw_constant=None):
        """
        Construct a piecewise polynomial trajectory for a differentially-flat
            quadrotor

        Uses:
            Everything used by the following functions:
                self.get_cost()
                self.get_piece_poly()

        Modifies:
            self.
            waypoints:
            time_penalty:
            costs:
            der_fixed:
            order:
            times: time in which to complete each segment (not the transition
                times)
            yaw_eps:
            face_front:

            Calls joint_optimize.PolyTraj constructor and update_times method
                for each flat output trajectory

        Args:
            closed_loop: a flag to indicate the trajectory should be a closed loop

        Returns:

        Raises:
            minsnap.exceptions.InputError if the input values are of incorrect
                sizes or shapes
        """
        self.waypoints = dict()
        self.time_penalty = time_penalty
        self.order = order
        self.yaw_eps = yaw_eps
        self.closed_loop = closed_loop

        self.outer_opt_time = 0.0
        self.opt_time = 0.0

        self.set_yaw_constant = set_yaw_constant # None - do not, otherwise set to constant value
        self.set_yaw_to_traj = set_yaw_to_traj # T/F flag

        # TODO(mereweth@jpl.nasa.gov) - check that x, y, and z (and yaw, if
            # specified) all have the same number of waypoints

        # for x, we only received positions
        if len(np.shape(partial_waypoints['x'])) == 0:
            raise exceptions.InputError(partial_waypoints['x'],
                                        "Invalid waypoint array")
        elif len(np.shape(partial_waypoints['x'])) == 1:
            self.n_seg = np.size(partial_waypoints['x']) - 1

            # bump up the number of dimensions
            for key in partial_waypoints.keys():
                self.waypoints[key] = np.array([partial_waypoints[key]])
        # we received positions, and some higher order derivatives
        elif len(np.shape(partial_waypoints['x'])) == 2:
            self.n_seg = np.shape(partial_waypoints['x'])[1] - 1

            # don't bump up the number of dimensions
            for key in partial_waypoints.keys():
                self.waypoints[key] = np.array(partial_waypoints[key])
        else:
            raise exceptions.InputError(partial_waypoints['x'],
                                        "Invalid waypoint array")

        if self.n_seg < 1:
            raise exceptions.InputError(partial_waypoints['x'],
                                        "Invalid waypoint array; caused number"
                                        + " of segments to be less than 1")

        #TODO(mereweth@jpl.nasa.gov) - investigate this - cost should scale
        # linearly with number of segments; make time cost behave the same way
        self.time_penalty = self.time_penalty# * self.n_seg

        if face_front is None or np.all(face_front):
            self.face_front = [True] * (self.n_seg + 1)
            n_der = utils.n_coeffs_free_ders(self.order['yaw'])[1]
            self.waypoints['yaw'] = np.array([[0.0] * (self.n_seg + 1)] * n_der)
        else:
            if np.size(face_front) != self.n_seg + 1:
                raise exceptions.InputError(face_front,
                                            "Need to specify for each waypoint"
                                            + " whether to face along the"
                                            + " trajectory or use the yaw setpoint")
            self.face_front = face_front

        self.inner_loop_callback = inner_loop_callback

        # times is the time per segment, not the transition times
        if seed_times is None:
            zeroth_deriv_xyz = dict()
            for key in ['x', 'y', 'z']:
                zeroth_deriv_xyz[key] = self.waypoints[key][0]
            self.times = utils.get_seed_times(zeroth_deriv_xyz, seed_avg_vel)
        else:
            self.times = seed_times

        if der_fixed is None:
            der_fixed = dict()
            num_internal = self.n_seg - 1
            for key in self.waypoints.keys():
                n_der = utils.n_coeffs_free_ders(self.order[key])[1]
                # float derivatives at internal waypoints and fix at beginning
                # and end
                if num_internal > 0:
                    inner = [[True] + [False] *
                             num_internal + [True]] * (n_der - 1)
                    # fix 0th derivative at internal waypoints
                    der_fixed[key] = [[True] * (num_internal + 2)]
                    der_fixed[key].extend(inner)
                else:
                    # no internal waypoints; fix all
                    der_fixed[key] = [[True] * 2] * n_der

                der_fixed[key] = np.array(der_fixed[key])

        self.der_fixed = der_fixed

        # for yaw, also control first derivatives at each waypoint if
        # face_front?

        if der_ineq is None:
            der_ineq = dict()
            delta_tmp = delta
            delta = dict()
            for key in self.waypoints.keys():
                der_ineq[key] = np.zeros(der_fixed[key].shape,dtype=bool)
                delta[key] = delta_tmp
        else:
            for key in self.waypoints.keys():
                if (der_ineq[key][der_ineq[key]]==der_fixed[key][der_ineq[key]]).any(): # If any of the positions set to true for the inequality are also true for the fixed
                    raise exceptions.InputError(der_ineq,
                                        "Invalid inequality and fixed input arrays;"
                                        + "Have conflicting selections of constraints"
                                        + "i.e. both are true for the same derivative")


        self.der_ineq = der_ineq
        self.delta = delta

        for key in self.waypoints.keys():
            n_der = utils.n_coeffs_free_ders(self.order[key])[1]

            row = np.shape(self.waypoints[key])[0]
            col = np.shape(self.waypoints[key])[1]
            if row < n_der:
                self.waypoints[key] = np.append(self.waypoints[key],
                                                [[0.0] * col] * (n_der - row), axis=0)

            if np.shape(self.waypoints[key]) != np.shape(self.der_fixed[key]):
                raise exceptions.InputError(self.waypoints[key],
                                            "Mismatch between size of waypoints"
                                            + " array and size of derivative fixed"
                                            + "array")

            if np.size(self.times) != np.shape(self.der_fixed[key])[1] - 1:
                raise exceptions.InputError(self.times,
                                            "Mismatch between number of segment times"
                                            + " and number of segments")

        if costs is None:
            costs = dict()
            costs['x'] = [0, 0, 0, 0, 1]  # minimum snap
            costs['y'] = [0, 0, 0, 0, 1]  # minimum snap
            costs['z'] = [0, 0, 1]  # minimum acceleration
            costs['yaw'] = [0, 0, 1]  # minimum acceleration
        self.costs = costs

        # Closed loop modification
        if self.closed_loop:
            self.close_trajectory_loop()

        # Generate more waypoints and modify der_ineq, der_fixed and delta to suit
        self.restrict_freespace = restrict_freespace
        if restrict_freespace:
            # Check and modify input for l_max and A_max
            if type(l_max).__module__ != np.__name__:
                # Not an array type, so all need to copy it to make it so
                l_max = np.array([l_max]*self.n_seg)
            if type(A_max).__module__ != np.__name__:
                # Not an array type, so all need to copy it to make it so
                A_max = np.array([A_max]*self.n_seg)
            # Generate the waypoints from the nodes
            if not hasattr(self,"der_fixed_nodes"):
                self.waypoints_from_nodes(l_max,A_max)
            else:
                self.l_max = l_max
            self.A_max = A_max
        else:
            self.nodes = None
            self.l_max = l_max
            self.A_max = A_max


        # Initialise joint optimise polynomials for each dimension
        self.quad_traj = dict()

        for key in self.waypoints.keys():
            # if any of the waypoints are specified to point along the trajectory,
                # need to do yaw after first pass
            if not (key == 'yaw' and any(self.face_front)):
                self.quad_traj[key] = joint_optimize.PolyTraj(self.waypoints[key],
                                                              self.order[key],
                                                              self.costs[key],
                                                              self.der_fixed[key],
                                                              self.times,
                                                              self.der_ineq[key],
                                                              self.delta[key],
                                                              closed_loop=self.closed_loop)

        if any(self.face_front):
            self.set_yaw_des_from_traj(set_yaw_to_traj=False)
            self.quad_traj['yaw'] = joint_optimize.PolyTraj(self.waypoints['yaw'],
                                                            self.order['yaw'],
                                                            self.costs['yaw'],
                                                            self.der_fixed['yaw'],
                                                            self.times,
                                                            self.der_ineq['yaw'],
                                                            self.delta['yaw'],
                                                            closed_loop=self.closed_loop)
            if self.set_yaw_to_traj:
                self.set_yaw_des_from_traj(self.set_yaw_to_traj)


    def insert(self, new_index, new_waypoint, new_times,
                new_der_fixed=dict(x=True,y=True,z=True,yaw=True),defer=False):
        """
        Insert new waypoints to start at the selected index

        Uses:
            Everything used by the following functions:
                self.get_cost()
                self.get_piece_poly()

        Modifies:
            self.
            waypoints:
            der_fixed:
            n_seg:
            times: time in which to complete each segment (not the transition
                times)

            Calls joint_optimize.PolyTraj.insert on each flat output trajectory

        Args:
            new_index: index that new waypoints should start at after insertion
            new_waypoints: numpy array
            new_times:
            new_der_fixed:

        Returns:

        Raises:
        """

        for key in new_waypoint.keys():
            if new_waypoint[key] is not None:
                n_der = utils.n_coeffs_free_ders(self.order[key])[1]
                n_to_append = n_der - np.size(new_waypoint[key])
                if n_to_append > 0:
                    if new_der_fixed[key] is not None:
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

                self.quad_traj[key].insert(new_index, new_waypoint[key], new_times,
                                                    new_der_fixed[key], defer=defer)

                #TODO(mereweth@jpl.nasa.gov) - is this the cleanest way to sync
                    # the arrays specifying which derivatives are fixed and what
                    # the waypoint values are at each of them?
                self.der_fixed[key] = self.quad_traj[key].der_fixed
                self.waypoints[key] = self.quad_traj[key].waypoints
                # self.der_ineq[key] = self.quad_traj[key].der_ineq
                # TODO(bmorrell@jpl.nasa.gov) sort outhandling of der_ineq

        if new_waypoint['yaw'] is not None:
            self.face_front = np.insert(self.face_front, new_index, 'false')
        else:
            self.face_front = np.insert(self.face_front, new_index, 'true')

        self.times = self.quad_traj['x'].times
        self.n_seg = self.quad_traj['x'].n_seg

        if new_index != 0 and new_index <= self.times.size-1:
            self.times[new_index-1] = new_times[0]
        # self.face_front = np.insert(self.face_front,new_index,True)

        if self.set_yaw_to_traj:
            self.set_yaw_des_from_traj(self.set_yaw_to_traj)

    def delete(self, delete_index, new_time,defer=False):
        """
        Delete waypoints with specified indices and set duration of joining segment

        Uses:
            Everything used by the following functions:
                self.get_cost()
                self.get_piece_poly()

        Modifies:
            self.
            waypoints:
            der_fixed:
            n_seg:
            times: time in which to complete each segment (not the transition
                times)

            Calls joint_optimize.PolyTraj.delete on each flat output trajectory

        Args:
            delete_indices: indices of waypoints to delete
            new_time: duration of segment that joins the waypoints on each side
                of the deletion

        Returns:

        Raises:
        """
        for key in self.waypoints.keys():
            self.quad_traj[key].delete(delete_index, new_time, defer=defer)

            self.der_fixed[key] = self.quad_traj[key].der_fixed
            self.waypoints[key] = self.quad_traj[key].waypoints
            # self.der_ineq[key] = self.quad_traj[key].der_ineq


        self.times = self.quad_traj['x'].times
        self.n_seg = self.quad_traj['x'].n_seg
        self.face_front = np.delete(self.face_front,delete_index)

        if self.set_yaw_to_traj:
            self.set_yaw_des_from_traj(self.set_yaw_to_traj)

    def prepend(self, new_qr_poly):
        """Utility function for PPoly of different order"""
        piece_polys = dict()
        for key in self.quad_traj.keys():
            piece_polys[key] = self.quad_traj[key].prepend(
                new_qr_poly.quad_traj[key])
        return piece_polys

    def append(self, new_qr_poly):
        """Utility function for PPoly of different order"""
        piece_polys = dict()
        for key in self.quad_traj.keys():
            piece_polys[key] = self.quad_traj[key].append(
                new_qr_poly.quad_traj[key])
        return piece_polys

    def make_n_laps(self, n_laps, entry_ID, exit_ID):
        """ Create multiple laps of the closed loop trajectory """
        if not self.closed_loop:
            print('Need to close loop of trajectory first')
            return

        piece_polys = dict()
        for key in self.quad_traj.keys():
            piece_polys[key] = self.quad_traj[key].create_n_laps(n_laps,entry_ID,exit_ID)

        return piece_polys

    def entry_exit_on_open_traj(self, entry_ID, exit_ID):
        if entry_ID >= exit_ID:
            print("ERRORS LIKELY: Keep entry before exit for open trajectory")
            return
        piece_polys = dict()
        for key in self.quad_traj.keys():
            piece_polys[key] = self.quad_traj[key].create_n_laps(1,entry_ID,exit_ID)

        return piece_polys

    def update_times(self, times):
        # TODO(mereweth@jpl.nasa.gov) - check length of times

        self.times = times
        start_timer = time.time()
        self.quad_traj['x'].update_times(self.times)
        self.quad_traj['y'].update_times(self.times)
        self.quad_traj['z'].update_times(self.times)

        if np.any(self.face_front):
            self.set_yaw_des_from_traj(self.set_yaw_to_traj)
            if not self.set_yaw_to_traj:
                self.quad_traj['yaw'].update_waypoints(self.waypoints['yaw'],
                                                       defer=True)

        if not self.set_yaw_to_traj:
            self.quad_traj['yaw'].update_times(self.times)

        self.opt_time = time.time() - start_timer

    def update_xyz_yaw_partial_waypoint(self, index, new_waypoint,
                                        new_der_fixed=dict(x=None, y=None, z=None,
                                                           yaw=None),
                                        defer=False):
        """Set new waypoint value and whether derivatives are fixed

        Uses:

        Modifies:
            self.
            waypoints:
            der_fixed:
            quad_traj:

        Args:
            new_waypoint: dictionary of arrays of waypoint derivatives with
                length less than or equal to n_der, the number of free
                derivatives based on the polynomial order.
            new_der_fixed: dictionary of Boolean arrays of length less than or
                equal to n_der (max potential number of free derivatives per
                segment). Fixing a derivative at the waypoint is performed by
                setting the corresponding entry to True.

        Returns:

        Raises:
            minsnap.exceptions.InputError if
                np.size(new_waypoint[key]) != np.shape(self.waypoints[key])[1]

            minsnap.exceptions.InputError if
                np.size(new_der_fixed[key]) != np.shape(self.der_fixed[key])[1]
        """

        for key in new_waypoint.keys():
            if new_waypoint[key] is not None:
                n_der = utils.n_coeffs_free_ders(self.order[key])[1]
                n_to_append = n_der - np.size(new_waypoint[key])
                if n_to_append > 0:
                    if new_der_fixed[key] is not None:
                        if (index == 0) or (index == (self.n_seg)):
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

                self.quad_traj[key].update_waypoint(index, new_waypoint[key],
                                                    new_der_fixed[key], defer=defer)

                #TODO(mereweth@jpl.nasa.gov) - is this the cleanest way to sync
                    # the arrays specifying which derivatives are fixed and what
                    # the waypoint values are at each of them?
                self.der_fixed[key] = self.quad_traj[key].der_fixed
                self.waypoints[key] = self.quad_traj[key].waypoints
                # self.der_ineq[key] = self.quad_traj[key].der_ineq
                # if key == "yaw":
                #     print("Yaw set as :{}".format(new_waypoint[key]))

        if self.set_yaw_to_traj:
            self.set_yaw_des_from_traj(self.set_yaw_to_traj)

    def set_yaw_des_from_traj(self,set_yaw_to_traj=False):

        if set_yaw_to_traj:
            trans_times = utils.seg_times_to_trans_times(self.times)
            # t_total = trans_times[-1] - trans_times[0]
            # t = np.linspace(trans_times[0], trans_times[-1], t_total*100)

            # test_yaw = self.quad_traj['yaw'].piece_poly(t)

            # yaw = np.arctan2(self.quad_traj['y'].piece_poly.derivative()(t),self.quad_traj['x'].piece_poly.derivative()(t))


            for i in range(0,self.n_seg):
                t_range = np.linspace(trans_times[i],trans_times[i+1],50)
                t_range2 = np.linspace(0,self.times[i],50)
                yaw = np.arctan2(self.quad_traj['y'].piece_poly.derivative()(t_range),self.quad_traj['x'].piece_poly.derivative()(t_range))

                new_poly = np.polyfit(t_range2,yaw,self.order['yaw'])

                self.quad_traj['yaw'].piece_poly.c[:,i] = new_poly
                self.quad_traj['yaw'].coeffs[:,i] = np.reshape(new_poly,(self.order['yaw']+1,1))

            # yaw_out = self.quad_traj['yaw'].piece_poly(t)

            # import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()
            # spl = sp.interpolate.splrep(t,yaw,k=5)
            # import pdb; pdb.set_trace()
            # self.quad_traj['yaw'].piece_poly = sp.interpolate.PPoly.from_spline(spl)
            #
            # self.quad_traj['yaw'].coeffs = self.quad_traj['yaw'].piece_poly.c
            # self.quad_traj['yaw'].times = utils.trans_times_to_seg_times(self.quad_traj['yaw'].piece_poly.x)

        else:
            # Just set the waypoints

            x_vel_poly = self.quad_traj['x'].piece_poly.derivative()
            y_vel_poly = self.quad_traj['y'].piece_poly.derivative()

            x_acc_poly = x_vel_poly.derivative()
            y_acc_poly = y_vel_poly.derivative()

            trans_times = utils.seg_times_to_trans_times(self.times)



            n_seg = self.n_seg

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

            x_vel[0] = x_vel_poly(trans_times[0] + self.yaw_eps)
            y_vel[0] = y_vel_poly(trans_times[0] + self.yaw_eps)
            x_acc[0] = x_acc_poly(trans_times[0] + self.yaw_eps)
            y_acc[0] = y_acc_poly(trans_times[0] + self.yaw_eps)

            x_vel[1:-1] = x_vel_poly(trans_times[1:-1])
            y_vel[1:-1] = y_vel_poly(trans_times[1:-1])
            x_acc[1:-1] = x_acc_poly(trans_times[1:-1])
            y_acc[1:-1] = y_acc_poly(trans_times[1:-1])

            x_vel[-1] = x_vel_poly(trans_times[-1] - self.yaw_eps)
            y_vel[-1] = y_vel_poly(trans_times[-1] - self.yaw_eps)
            x_acc[-1] = x_acc_poly(trans_times[-1] - self.yaw_eps)
            y_acc[-1] = y_acc_poly(trans_times[-1] - self.yaw_eps)

            # TODO(mereweth@jpl.nasa.gov) - check self.face_front for each waypoint
            face_front_idx = np.nonzero(self.face_front)

            self.waypoints['yaw'][0, face_front_idx] = np.arctan2(y_vel, x_vel)[face_front_idx]

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

        if self.set_yaw_constant is not None:
            # CAUTION - this overrides the flag to set yaw to traj
            self.waypoints['yaw'][0,:] = self.set_yaw_constant

    def set_yaw_to_random(self, delta):
        """
        Sets yaw to randomised values about a mean for system identification flights

        Args:
            delta - amount to randomise about the first yaw setting

        """

        # Take the first waypoint as the mean about which to randomise
        mean_yaw = self.waypoints['yaw'][0,0]

        # COmpute the new yaw with normally distributed random deviations
        new_yaw = np.random.randn(self.n_seg-1)*delta + mean_yaw

        # Set waypoints to the new way (all but the fisrt and last waypoints)
        self.waypoints['yaw'][0,1:-1] = new_yaw
        self.waypoints['yaw'][0,-1] = mean_yaw

        print("New yaw values are {}".format(self.waypoints['yaw']))

        # Update the polynomials
        self.quad_traj['yaw'].update_waypoints(self.waypoints['yaw'],defer=False)

    def set_yaw_to_constant(self, constant_yaw):
        """
        Sets yaw to randomised values about a mean for system identification flights

        Args:
            delta - amount to randomise about the first yaw setting

        """

        # Set waypoints to the new constant yaw
        self.waypoints['yaw'][0,:] = constant_yaw

        # Update the polynomials
        self.quad_traj['yaw'].update_waypoints(self.waypoints['yaw'],defer=False)


    def update_xyz_yaw_partial_nodes(self, index, new_node):

        print("New node is: {}".format(new_node))
        for key in new_node.keys():
            self.nodes[key][0,index] = new_node[key]

    def insert_node(self, new_index, new_node, new_der_fixed=dict(x=None,y=None,z=None,yaw=None)):
        """
            Bare bones insert nodes. Simply inserts in the nodes variable
            User needs to run check minimum distance, and waypoints_from_nodes
            afterwards to generate updated trajectories
            Note that derivatives do not matter for nodes
        """
        print("Inserting new node at index {}\n New node: {}".format(new_index,new_node))
        # Checking new node
        for key in new_node.keys():
            if new_node[key] is not None:
                n_der = utils.n_coeffs_free_ders(self.order[key])[1]
                n_to_append = n_der - np.size(new_node[key])
                if n_to_append > 0:
                    if (new_index == 0) or (new_index == (self.n_seg + 1)):
                        # first or last waypoint of entire trajectory -> fix the
                            # higher order derivatives by default
                        if new_der_fixed[key] is not None:
                            n_der_append = n_der - new_der_fixed['x'].shape[0]
                            new_der_fixed[key] = np.append(np.atleast_1d(new_der_fixed[key]),
                                                           [True] * n_der_append,
                                                           axis=0)
                        else:
                            new_der_fixed[key] = np.ones(n_der,dtype=bool)

                    else:
                        # float the higher order derivatives by default at internal
                            # waypoints
                        if new_der_fixed[key] is not None:
                            n_der_append = n_der - new_der_fixed['x'].shape[0]
                            new_der_fixed[key] = np.append(np.atleast_1d(new_der_fixed[key]),
                                                           [False] * n_der_append,
                                                           axis=0)
                        else:
                            new_der_fixed[key] = np.append([True],np.zeros(n_der-1,dtype=bool),axis=0)

                    new_node[key] = np.append(np.atleast_1d(new_node[key]),
                                                  [0.0] * n_to_append,
                                                  axis=0)

            self.nodes[key] = np.insert(self.nodes[key],new_index,new_node[key],axis=1)

            # modify der_fixed
            self.der_fixed_nodes[key] = np.insert(self.der_fixed_nodes[key],new_index,np.array(new_der_fixed[key]),axis=1)
            # Correct der fixed around it
            if new_index == 0:
                self.der_fixed_nodes[key][1:,1] = False
            elif new_index == (self.n_seg + 1):
                self.der_fixed_nodes[key][1:,-2] = False

            # Set der_ineq to default with all false TODO adjust this if it will be used
            self.der_ineq_nodes[key] = np.zeros(self.der_fixed_nodes[key].shape,dtype=bool)



        # Resize A_max
        if new_index > self.A_max.size-1:
            self.A_max = np.concatenate([self.A_max,np.atleast_1d(self.A_max[0])])
        else:
            self.A_max = np.insert(self.A_max,new_index,self.A_max[0])

    def delete_node(self, delete_index):
        """
            Bare bones delete nodes. Simply removes from the nodes variable
            User needs to run check minimum distance, and waypoints_from_nodes
            afterwards to generate updated trajectories
        """
        print("Deleting node {}".format(delete_index))

        # Take out waypoint and der_fixed
        mask = np.ones(self.nodes['x'].shape[1],dtype=bool)
        mask[delete_index] = False
        for key in self.nodes.keys():
            self.nodes[key] = self.nodes[key][:,mask]
            self.der_fixed_nodes[key] = self.der_fixed_nodes[key][:,mask]
            # Correct der_fixed around it
            if delete_index == 0:
                self.der_fixed_nodes[key][:,0] = True
            if delete_index == self.nodes['x'].shape[1]-1:
                self.der_fixed_nodes[key][:,-1] = True
            # Set der_ineq to default with all false TODO adjust this if it will be used
            self.der_ineq_nodes[key] = np.zeros(self.der_fixed_nodes[key].shape,dtype=bool)

        # nodes_temp = dict()
        # for key in self.nodes.keys():
        #     nodes_temp[key] = np.delete(self.nodes[key],index,axis=1)
        #
        # self.nodes = nodes_temp.copy()

        # Reduce A_max by one
        if delete_index < self.A_max.size:
            self.A_max = np.delete(self.A_max,delete_index)
        else:
            self.A_max = self.A_max[:-1]

    def make_forwards_compatible(self):
        # Define der_ineq for backward compatibility if not defined in the stored trajectory

        if not hasattr(self, 'der_ineq'):
            self.der_ineq = dict()
            self.delta = dict()
            for key in self.waypoints.keys():
                self.der_ineq[key]=self.der_fixed[key].copy()
                self.der_ineq[key][:,:] = False
                self.delta[key] = 0
                self.quad_traj[key].der_ineq = self.der_ineq[key]
                self.quad_traj[key].delta = 0

        # Freespace planning additions
        if not hasattr(self, 'nodes'):
            self.nodes = None
            self.der_fixed_nodes = None
            self.der_ineq_nodes = None
        elif not hasattr(self, 'der_fixed_nodes'):
            if self.nodes is None:
                self.der_fixed_nodes = None
                self.der_ineq_nodes = None
            else:
                self.der_fixed_nodes = utils.default_der_fixed_polytraj(self.nodes['x'].shape[1]-2,self.order)
                self.der_ineq_nodes = dict()
                for key in self.der_fixed_nodes.keys():
                    self.der_ineq_nodes[key] = np.zeros(self.der_fixed_nodes[key].shape,dtype=bool)
        if not hasattr(self, 'l_max'):
            self.l_max = np.atleast_1d([5.0])
        else:
            self.l_max = np.atleast_1d(self.A_max)
        if not hasattr(self, 'A_max'):
            self.A_max = np.atleast_1d([12.0])
        else:
            self.A_max = np.atleast_1d(self.A_max)
        if not hasattr(self,'restrict_freespace'):
            self.restrict_freespace = False

        if not hasattr(self,'closed_loop'):
            self.closed_loop = False
            for key in self.quad_traj.keys():
                self.quad_traj[key].closed_loop = False

        if not hasattr(self,'set_yaw_constant'):
            self.set_yaw_constant = None
            self.set_yaw_to_traj = True


    def cost_function(self, times, times_changed=None):
        if times_changed is None:
            times_changed = np.r_[0:self.n_seg]

        cost = (self.quad_traj['x'].get_cost(times) +
                self.quad_traj['y'].get_cost(times) +
                self.quad_traj['z'].get_cost(times) +
                self.time_penalty * np.sum(times))

        # TODO(mereweth@jpl.nasa.gov) - is this the right way to handle yaw cost?
        # If we are not changing any of the yaw waypoints to match the minimum
            # snap waypoints, then include yaw in the cost term
        if not np.any(self.face_front):
            cost = cost + self.quad_traj['yaw'].get_cost(times)

        return cost

    def relative_time_opt(self, **kwargs):
        # Inequality constraints are introduced here - to keep time more than zero
        # TODO(bmorrell@jpl.nasa.gov) increase the lower bound on the time with this constraint (TIME_NZ_EPS)
        modified = False

        cons = ({'type': 'ineq',
                 'fun': lambda x: (np.array(x) -
                                   np.ones((np.size(x))) * minsnap.settings.TIME_NZ_EPS)})

        kwargs['constraints'] = cons

        # NOTE(mereweth@jpl.nasa.gov) - qualitatively, wrapping the cost function
        # to only recalculate segments where the time changed doesn't help much
        # with speed of numerical gradient evaluation

        start_timer = time.time()

        cost_wrapper = utils.InputWrapper(self.cost_function)
        res = sp.optimize.minimize(cost_wrapper.wrap_func, self.times, **kwargs)

        self.outer_opt_time = time.time() - start_timer
        #res = sp.optimize.minimize(self.cost_function, self.times, **kwargs)

        self.opt_nfev = res.nfev
        self.opt_succes = res.success

        if (np.all(np.greater(res.x, np.zeros(np.shape(res.x)))) and
            self.cost_function(res.x) < self.cost_function(self.times)):
            # check if cost function is lower than at start
            # and check that no times are zero or negative
            self.times = res.x
            modified = True
        else:
            print("Optimization failed to reduce cost function\n")
            modified = False

        start_timer2 = time.time()

        self.quad_traj['x'].update_times(self.times)
        self.quad_traj['y'].update_times(self.times)
        self.quad_traj['z'].update_times(self.times)

        if np.any(self.face_front):
            self.set_yaw_des_from_traj(self.set_yaw_to_traj)
            if not self.set_yaw_to_traj:
                self.quad_traj['yaw'].update_waypoints(self.waypoints['yaw'],
                                                    defer=True)
        if not self.set_yaw_to_traj:
            self.quad_traj['yaw'].update_times(self.times)

        self.opt_time = time.time() - start_timer2

        return (res, modified)

    def fill_poly_traj_msg(self, msg):
        msg.n_segs = np.size(self.quad_traj['x'].piece_poly.x) - 1

        msg.n_x_coeffs = np.shape(self.quad_traj['x'].piece_poly.c)[0] # number of coefficients per segment for x
        msg.n_y_coeffs = np.shape(self.quad_traj['y'].piece_poly.c)[0] # number of coefficients per segment for y
        msg.n_z_coeffs = np.shape(self.quad_traj['z'].piece_poly.c)[0] # number of coefficients per segment for z
        msg.n_yaw_coeffs = np.shape(self.quad_traj['yaw'].piece_poly.c)[0] # number of coefficients per segment for yaw

        # polynomial coefficients, segment by segment, starting at highest order coefficient
        msg.x_coeffs[0:np.size(self.quad_traj['x'].piece_poly.c)] = np.asarray(self.quad_traj['x'].piece_poly.c.T).flatten().tolist()
        msg.y_coeffs[0:np.size(self.quad_traj['y'].piece_poly.c)] = np.asarray(self.quad_traj['y'].piece_poly.c.T).flatten().tolist()
        msg.z_coeffs[0:np.size(self.quad_traj['z'].piece_poly.c)] = np.asarray(self.quad_traj['z'].piece_poly.c.T).flatten().tolist()
        msg.yaw_coeffs[0:np.size(self.quad_traj['yaw'].piece_poly.c)] = np.asarray(self.quad_traj['yaw'].piece_poly.c.T).flatten().tolist()
        # transition times, segment by segment
        msg.t_trans[0:np.size(self.quad_traj['x'].piece_poly.x) - 1] = self.quad_traj['x'].piece_poly.x[1:]

        print("Message sent with {} segs, {} coefficients, {} times\n".format(msg.n_segs,msg.n_x_coeffs,np.size(msg.t_trans)))

    def get_full_state_trajectory(self,n_steps):
        # times
        trans_times = utils.seg_times_to_trans_times(self.times)
        t = np.linspace(trans_times[0], trans_times[-1], n_steps)
        x = self.quad_traj['x'].piece_poly(t)
        y = self.quad_traj['y'].piece_poly(t)
        z = self.quad_traj['z'].piece_poly(t)

        yaw = self.quad_traj['yaw'].piece_poly(t)
        yaw_dot = self.quad_traj['yaw'].piece_poly.derivative()(t)
        yaw_ddot = self.quad_traj['yaw'].piece_poly.derivative().derivative()(t)

        acc = np.array([self.quad_traj['x'].piece_poly.derivative().derivative()(t),
                        self.quad_traj['y'].piece_poly.derivative().derivative()(t),
                        self.quad_traj['z'].piece_poly.derivative().derivative()(t)])

        jerk = np.array([self.quad_traj['x'].piece_poly.derivative().derivative().derivative()(t),
                         self.quad_traj['y'].piece_poly.derivative().derivative().derivative()(t),
                         self.quad_traj['z'].piece_poly.derivative().derivative().derivative()(t)])

        snap = np.array([self.quad_traj['x'].piece_poly.derivative().derivative().derivative().derivative()(t),
                         self.quad_traj['y'].piece_poly.derivative().derivative().derivative().derivative()(t),
                         self.quad_traj['z'].piece_poly.derivative().derivative().derivative().derivative()(t)])

        self.t = t
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.yaw_dot = yaw_dot
        self.yaw_ddot = yaw_ddot
        self.acc = acc
        self.jerk = jerk
        self.snap = snap


    def waypoints_from_nodes(self,l_max,A_max):
        """
        Takes current input of waypoints as 'nodes' of a collision free path,
        then generates waypoints, with inequality constrinats, maximum velocity and accceleration
        to forec path to stay withing l_max of the input path

        Following tehniques of Campos Macias 2017, "A Hybrid Method for Online Trajectory
        Planning of Mobile Robots in Cluttered Environments"

        Uses:
            self.
            waypoints:
            der_fixed:
            der_ineq:

        Modifies:
            self.
            waypoints:, der_fixed:, der_ineq:, delta:, times:, nodes:

        Args:
            l_max: an array with length equal to the number of segments. Gives the minimum
                distance to an obstacle for each segment (used to set the bound for the
                trajectory from the reference path - defined by the nodes)
            A_max: an array with length equal to the number of segments. Gives the
                maximum allowable acceleration in each axis, for each segment.

        Returns:

        Raises:
        """

        if np.sum(l_max<=0.0) > 0:
            print("Error: Path is in collision with environment. Not computing waypoints")
            return
        print("Forming waypointsfrom super waypoints")
        # Nodes represent the reference path - take from the input waypoints
        if not hasattr(self,'nodes'):
            nodes = self.waypoints.copy()
            der_fixed_nodes = self.der_fixed.copy()
            der_ineq_nodes = self.der_ineq.copy()
        else:
            if self.nodes is None:
                nodes = self.waypoints.copy()
                der_fixed_nodes = self.der_fixed.copy()
                der_ineq_nodes = self.der_ineq.copy()
            else:
                nodes = self.nodes
                if hasattr(self,'der_fixed_nodes'):
                    der_fixed_nodes = self.der_fixed_nodes
                    der_ineq_nodes = self.der_ineq_nodes
                else:
                    # Initialise as default
                    der_fixed_nodes = utils.default_der_fixed_polytraj(nodes['x'].shape[1]-2,self.order)
                    der_ineq_nodes = dict()
                    for key in der_fixed_nodes.keys():
                        der_fixed_nodes[key][0,1:-2] = False
                        der_ineq_nodes[key] = np.zeros(der_fixed_nodes[key].shape,dtype=bool)

        # Intilialise waypoints dictionary
        waypoints = dict()
        der_fixed = dict()
        der_ineq = dict()
        delta = dict()
        times = np.array([])
        # wayp_per_seg = np.zeros()

        if type(A_max).__module__ != np.__name__ or np.size(A_max) == 1:
            A_max = np.repeat(A_max,(nodes['x'].shape[1]-1))
        elif np.size(A_max) != nodes['x'].shape[1]-1:
            # Repeat the first value
            A_max = np.repeat(A_max[0],(nodes['x'].shape[1]-1))
        if type(l_max).__module__ != np.__name__ or np.size(l_max) == 1:
            l_max = np.repeat(l_max,(nodes['x'].shape[1]-1))
        elif np.size(l_max) != nodes['x'].shape[1]-1:
            # Repeat the first value
            l_max = np.repeat(l_max[0],(nodes['x'].shape[1]-1))

        for key in nodes.keys():
            waypoints[key] = np.array([]).reshape([nodes[key].shape[0],0])
            der_fixed[key] = np.array([],dtype=bool).reshape([nodes[key].shape[0],0])
            der_ineq[key] = np.array([],dtype=bool).reshape([nodes[key].shape[0],0])
            delta[key] = np.array([]).reshape([nodes[key].shape[0],0])

        if np.sum(l_max<=0.0) > 0:
            print("Error: Path is in collision with environment. Setting to minimum but path is invalid")
            l_max[l_max<=0.0] = 0.02

        # Loop for each segment between nodes
        for i in range(1,nodes['x'].shape[1]):
            # Compute l parameter, the spacing of the waypoints
            spacing = 2.0/3.0*l_max[i-1]/np.sqrt(3) # l
            # print(l_max[i-1])
            # Compute V_max and timestep h
            V_max = np.sqrt(spacing*A_max[i-1])
            h = np.sqrt(4*spacing/A_max[i-1])

            # Vector between two nodes in segment (between two nodes)
            n1 = np.array([nodes['x'][0,i-1],nodes['y'][0,i-1],nodes['z'][0,i-1]])
            n2 = np.array([nodes['x'][0,i],nodes['y'][0,i],nodes['z'][0,i]])
            vec_between = n2-n1

            # Number of waypoints in segment with the given spacing (at a minimum, 2)
            n_waypoints = max(int(np.ceil(np.linalg.norm(vec_between)/spacing)),2)

            # times
            times = np.concatenate([times,h*np.ones(n_waypoints)])

            # Create waypoints
            for key in nodes.keys():
                # Initialise matrices
                w_mat = np.zeros([nodes[key].shape[0],n_waypoints]) # waypoints
                df_mat = np.zeros([nodes[key].shape[0],n_waypoints],dtype=bool) # der_fixed
                di_mat = np.zeros([nodes[key].shape[0],n_waypoints],dtype=bool)# der_ineq
                delt_mat = np.zeros([nodes[key].shape[0],n_waypoints]) # delta

                # Set the end points
                w_mat[:,0] = nodes[key][:,i-1]
                w_mat[:,-1] = nodes[key][:,i]

                # Manually fix the first and last waypoints, leaving others with no constraints
                if i == 1:
                    df_mat[:,0] = True

                if i == nodes['x'].shape[1] - 1:
                    df_mat[:,-1] = True

                # TODO (bmorrell@jpl.nasa.gov) check what usability we want for this - do we want to fix waypoints?
                # df_mat[:,0] = der_fixed_nodes[key][:,i-1]
                # df_mat[:,-1] = der_fixed_nodes[key][:,i]
                # di_mat[:,0] = der_ineq_nodes[key][:,i-1]
                # di_mat[:,-1] = der_ineq_nodes[key][:,i]

                # Linearly interpolate for the internal points
                w_mat[0,:] = np.linspace(nodes[key][0,i-1],nodes[key][0,i],n_waypoints)
                # Keep all velocities and accelerations to zero if not at the fixed nodes
                # Inequalities on all positions
                di_mat[0:3,1:-1] = True
                # TODO(bmorrell@jpl.nasa.gov): how to deal with the set nodes for vel and accel - keep as input?
                # inequalities on all velocities and accelerations
                # di_mat[1:3,:] = True

                # Set the boundaries
                delt_mat[0,:] = spacing
                delt_mat[1,:] = V_max
                delt_mat[2,:] = A_max[i-1]

                # Store add to matrix for the current dimension
                waypoints[key] = np.concatenate([waypoints[key],w_mat],1)
                der_fixed[key] = np.concatenate([der_fixed[key],df_mat],1)
                der_ineq[key] = np.concatenate([der_ineq[key],di_mat],1)
                delta[key] = np.concatenate([delta[key],delt_mat],1)

        # Take off last time step
        times = np.delete(times,times.size-1)



        self.waypoints = waypoints
        self.times = times
        self.der_fixed = der_fixed
        self.der_ineq = der_ineq
        self.delta = delta
        self.nodes = nodes
        self.l_max = l_max
        self.der_fixed_nodes = der_fixed_nodes
        self.der_ineq_nodes = der_ineq_nodes

        # Update data with the change in number of waypoints
        self.update_number_of_waypoints()

        if any(self.face_front):
            self.face_front = [True] * (self.n_seg + 1)

    def update_number_of_waypoints(self):
        """
        To update the range of data required when there are a a different number of waypoints
        """

        self.n_seg = self.waypoints['x'].shape[1]-1

        # Change in each joint optmize object
        if hasattr(self, 'quad_traj'):
            for key in self.waypoints.keys():
                self.quad_traj[key].waypoints = self.waypoints[key]
                self.quad_traj[key].der_fixed = self.der_fixed[key]
                self.quad_traj[key].der_ineq = self.der_ineq[key]
                self.quad_traj[key].delta = self.delta[key]
                self.quad_traj[key].times = self.times
                self.quad_traj[key].n_seg = self.n_seg
                # Reset matrices
                self.quad_traj[key].A = None
                self.quad_traj[key].Q = None
                self.quad_traj[key].M = None
                self.quad_traj[key].R = None
                # TODO(bmorrell@jpl.nasa.gov) Look into a better way to do this to re-use parts of a matrix of only one waypoint is added

    def close_trajectory_loop(self):
        if self.closed_loop and hasattr(self,"quad_traj"):
            if self.quad_traj['x'].closed_loop:
                # Already a closed loop
                print("Loop is already closed. No need to close again")
                return
        else:
            self.closed_loop = True

        # Add a waypoint at the end that is fixed to the first
        for key in self.der_fixed.keys():
            n_der = utils.n_coeffs_free_ders(self.order[key])[1]
            # Make higher order derivatives free at the current end points`
            self.der_fixed[key][1:,[0,-1]] = False
            # Repeat the first waypoint at the end
            self.waypoints[key] = np.append(self.waypoints[key],self.waypoints[key][:,0].reshape(n_der,1),axis=1)
            self.der_fixed[key] = np.append(self.der_fixed[key],self.der_fixed[key][:,0].reshape(n_der,1),axis=1)
            self.der_ineq[key]  = np.append(self.der_ineq[key],self.der_ineq[key][:,0].reshape(n_der,1),axis=1)
            # self.delta[key]  = np.append(self.delta[key],self.delta[key][0])

        self.n_seg += 1
        self.times = np.append(self.times,self.times[0])

        if hasattr(self,"quad_traj"): # If already initialised

            # If TACO being run, then change the nodes
            if self.restrict_freespace and hasattr(self,"nodes"):
                # Generate the waypoints from the nodes
                new_node = dict()
                new_der_fixed = dict()
                for key in self.nodes.keys():
                    n_der = utils.n_coeffs_free_ders(self.order[key])[1]
                    new_node[key] = self.nodes[key][:,0].copy()
                    new_der_fixed[key] = np.array([True]+[False]*int(n_der-1))

                self.insert_node(self.nodes['x'].shape[1],new_node,new_der_fixed)
                print("updated super waypoints")
                # Don't do joint optimise yet (need obstacle information)
                return

            # reinitialise joint optimise polynomials for each dimension
            self.quad_traj = dict()

            for key in self.waypoints.keys():
                # if any of the waypoints are specified to point along the trajectory,
                    # need to do yaw after first pass
                if not (key == 'yaw' and any(self.face_front)):
                    self.quad_traj[key] = joint_optimize.PolyTraj(self.waypoints[key],
                                                                  self.order[key],
                                                                  self.costs[key],
                                                                  self.der_fixed[key],
                                                                  self.times,
                                                                  self.der_ineq[key],
                                                                  self.delta[key],
                                                                  closed_loop=self.closed_loop)

            if any(self.face_front):
                self.set_yaw_des_from_traj(set_yaw_to_traj=False)
                self.quad_traj['yaw'] = joint_optimize.PolyTraj(self.waypoints['yaw'],
                                                                self.order['yaw'],
                                                                self.costs['yaw'],
                                                                self.der_fixed['yaw'],
                                                                self.times,
                                                                self.der_ineq['yaw'],
                                                                self.delta['yaw'],
                                                                closed_loop=self.closed_loop)
                if self.set_yaw_to_traj:
                    self.set_yaw_des_from_traj(self.set_yaw_to_traj)


    def open_trajectory_loop(self):

        if self.closed_loop is False:
            print("Loop is already open. No need to open")
            return
        else:
            self.closed_loop = False

        # Remove waypoint at the end that is fixed to the first
        for key in self.der_fixed.keys():
            n_der = utils.n_coeffs_free_ders(self.order[key])[1]
            # Fix the higher order derivatives at the new end points`
            self.der_fixed[key][1:,[0,-2]] = True
            # Remove the last waypoint
            self.waypoints[key] = np.delete(self.waypoints[key],self.n_seg,axis=1)
            self.der_fixed[key] = np.delete(self.der_fixed[key],self.n_seg,axis=1)
            self.der_ineq[key]  = np.delete(self.der_ineq[key],self.n_seg,axis=1)
            # self.delta[key]  = np.append(self.delta[key],self.delta[key][0])

        # Reduce segment count
        self.n_seg -= 1
        self.times = np.delete(self.times,self.n_seg-1)

        # Update joint optimize trajectory
        # If TACO being run,delete from nodes
        if self.restrict_freespace:
            # Delete last node
            self.delete_node(self.nodes['x'].shape[1]-1)
            return

        # reinitialise joint optimise polynomials for each dimension
        self.quad_traj = dict()

        for key in self.waypoints.keys():
            # if any of the waypoints are specified to point along the trajectory,
                # need to do yaw after first pass
            if not (key == 'yaw' and any(self.face_front)):
                self.quad_traj[key] = joint_optimize.PolyTraj(self.waypoints[key],
                                                              self.order[key],
                                                              self.costs[key],
                                                              self.der_fixed[key],
                                                              self.times,
                                                              self.der_ineq[key],
                                                              self.delta[key],
                                                              closed_loop=self.closed_loop)

        if any(self.face_front):
            self.set_yaw_des_from_traj(set_yaw_to_traj=False)
            self.quad_traj['yaw'] = joint_optimize.PolyTraj(self.waypoints['yaw'],
                                                            self.order['yaw'],
                                                            self.costs['yaw'],
                                                            self.der_fixed['yaw'],
                                                            self.times,
                                                            self.der_ineq['yaw'],
                                                            self.delta['yaw'],
                                                            closed_loop=self.closed_loop)
            if self.set_yaw_to_traj:
                self.set_yaw_des_from_traj(self.set_yaw_to_traj)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-f', '--file', default="share/sample_data/waypoints.yaml",
        help='filename to load waypoints from')
    parser.add_argument(
        '-o', '--outfile', default="share/sample_output/traj.pickle",
        help='filename to pickle problem to')
    parser.add_argument('time_penalty', type=float,
                        help='Time penalty coefficient')

    args = parser.parse_args()
    args.file = os.path.expanduser(args.file)
    args.outfile = os.path.expanduser(args.outfile)

    waypoints = utils.load_waypoints(args.file)
    # qr_polytraj = QRPolyTraj(waypoints, args.time_penalty)


    # qr_polytraj.inner_loop_callback = PRINT_TOTAL_COST

    # res = qr_polytraj.relative_time_opt(method='COBYLA', options=dict(disp=3, maxiter=200, tol=0.1))
    #
    # print(res)
    # try:
    #     with open(args.outfile, 'wb') as f:
    #         print("Pickling to {}".format(args.outfile))
    #         # Pickle the 'data' dictionary using protocol 2 for compatibility.
    #         pickle.dump(qr_polytraj, f, 2)
    # except OSError:
    #     print("Path {} does not exist".format(args.outfile))
    #     args.outfile = os.path.split(args.outfile)[1]
    #     with open(args.outfile, 'wb') as f:
    #         print("Pickling to {}".format(args.outfile))
    #         # Pickle the 'data' dictionary using protocol 2 for compatibility.
    #         pickle.dump(qr_polytraj, f, 2)


    # new_waypoint = dict()
    # for key in waypoints.keys():
    #     new_waypoint[key] = 2.0
    #
    #
    # qr_polytraj.update_xyz_yaw_partial_waypoint(2, new_waypoint)
    #
    # for key in waypoints.keys():
    #     new_waypoint[key] = 1.5
    #
    # time1 = qr_polytraj.times
    # new_times = [time1[1]/2,time1[1]/2]
    # a1 = qr_polytraj.quad_traj['x'].free_ders
    #
    # # import pdb; pdb.set_trace()
    # qr_polytraj.insert(2, new_waypoint, new_times)
    # a2 = qr_polytraj.quad_traj['x'].free_ders
    #
    # qr_polytraj.delete(2, time1[1])
    # a3 = qr_polytraj.quad_traj['x'].free_ders
    #
    # time2 = qr_polytraj.times
    #
    # # import pdb; pdb.set_trace()
    # np.allclose(a1,a3)
    #
    # l_max = np.array(1.0)
    # A_max = np.array(3.0)
    # # import pdb; pdb.set_trace()
    # qr_polytraj.waypoints_from_nodes(l_max,A_max)

    #
    # qr_polytraj = QRPolyTraj(waypoints, args.time_penalty,
    #                                      restrict_freespace=True)
    #
    # new_waypoint = dict()
    # for key in waypoints.keys():
    #     new_waypoint[key] = 2.0
    #
    # import pdb; pdb.set_trace()
    # qr_polytraj.insert_node(2,new_waypoint)
    # import pdb; pdb.set_trace()
    # qr_polytraj.delete_node(2)
    # import pdb; pdb.set_trace()

    qr_polytraj = QRPolyTraj(waypoints, args.time_penalty,
                                         restrict_freespace=True)

    import pdb; pdb.set_trace()
    qr_polytraj = QRPolyTraj(waypoints, 10,restrict_freespace=True,
                                         closed_loop=True)

    import pdb; pdb.set_trace()
    qr_polytraj.close_trajectory_loop()
    import pdb; pdb.set_trace()
    qr_polytraj.open_trajectory_loop()
    import pdb; pdb.set_trace()
    qr_polytraj.close_trajectory_loop()
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()
