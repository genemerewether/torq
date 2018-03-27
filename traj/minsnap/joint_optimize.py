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
import time

import numpy
np = numpy

import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import scipy.interpolate
import scipy.optimize # Need to to this to access optimize
sp = scipy

from minsnap import utils
from minsnap import selector
from minsnap import cost
from minsnap import constraint
from minsnap import exceptions

from cvxopt import matrix
from cvxopt import solvers

"""Note! The function and form of this PolyTraj class are different than
that used to evaluate piecewise polynomials inside the PX4 flight stack."""


class PolyTraj(object):
    """Waypoints & times for a single spatial dimension"""

    def __init__(self, waypoints, order, costs, der_fixed, times=None, der_ineq=None, delta=None, print_output=False, closed_loop=False):
        if np.shape(waypoints) != np.shape(der_fixed):
            raise exceptions.InputError(waypoints,
                    "Mismatch between size of waypoints"
                    + " array and size of derivative fixed"
                    + "array")



        # if der_ineq is None:
        # no inequalities set
        der_ineq = np.array(der_fixed)
        der_ineq[:,:] = False
        # elif np.shape(der_ineq) != np.shape(der_fixed):
        #     raise exceptions.InputError(der_ineq,
        #             "Mismatch between size of der_ineq array and size of derivative fixed array")
        # elif (der_ineq[der_ineq]==der_fixed[der_ineq]).any():
        #     raise exceptions.InputError(der_ineq,"Invalid inequality and fixed input arrays;\n Have conflicting selections of constraints i.e. both are true for the same derivative")

        # constants
        self.waypoints = np.array(waypoints).copy()
        self.order = order
        self.n_der = utils.n_coeffs_free_ders(order)[1]
        self.costs = np.array(costs).copy()
        self.der_fixed = np.array(der_fixed).copy()
        self.der_ineq = np.array(der_ineq).copy()
        self.delta = delta
        self.n_seg = np.shape(waypoints)[1] - 1
        self.print_output = print_output
        self.closed_loop = closed_loop
        self.opt_time = 0.0

        self.Q = None
        self.M = None
        self.A = None
        self.R = None
        self.pinv_A = None
        self.free_ders = None
        self.fixed_ders = None
        self.piece_poly = None
        self.coeffs = None

        # don't need segment times to create the object
        # allow deferral of joint optimization
        if times is not None:
            self.times = np.array(times).copy()
            self.get_free_ders()
            self.get_poly_coeffs()
            self.get_piece_poly()


    def __eq__(self, other):
        # TODO(mereweth@jpl.nasa.gov) - check piecepoly too?
        if not (np.allclose(self.waypoints, other.waypoints)      and
                (self.order == other.order)                       and
                (self.n_der == other.n_der)                       and
                np.allclose(self.costs, other.costs)              and
                np.all(np.equal(self.der_fixed, other.der_fixed)) and
                (self.n_seg == other.n_seg)                       and
                np.allclose(self.Q.todense(), other.Q.todense())  and
                np.allclose(self.M.todense(), other.M.todense())  and
                np.allclose(self.A.todense(), other.A.todense())  and
                np.allclose(self.R.todense(), other.R.todense())  and
                np.allclose(self.pinv_A.todense(),
                            other.pinv_A.todense())               and
                np.allclose(self.free_ders, other.free_ders)      and
                np.allclose(self.fixed_ders, other.fixed_ders)    and
                np.allclose(self.times, other.times)              and
                np.allclose(self.coeffs, other.coeffs)):
            return False

        return True


    def insert(self, new_index, new_waypoint, new_times, new_der_fixed, defer=False):
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

            Everything modified by the following functions:
                self.get_cost()
                self.get_piece_poly()

        Args:
            new_index: index that new waypoints should start at after insertion
            new_waypoint: numpy array
            new_times:
            new_der_fixed:

        Returns:

        Raises:
        """

        # Check input
        if new_der_fixed is not None:
            if np.shape(new_der_fixed)[0] != np.shape(self.der_fixed)[0]:
                msg = "Mismatch between length of new derivative fixed array "
                msg = msg + "and size of 2nd dimension of stored derivative "
                msg = msg + "fixed array"
                raise exceptions.InputError(new_der_fixed, msg)
            der_fixed_changed = np.array(new_index).copy()
        else:
            der_fixed_changed = np.array([])

        if np.shape(new_waypoint)[0] != np.shape(self.der_fixed)[0]:
            msg = "Mismatch between length of new waypoint array and size of"
            msg = msg + "2nd dimension of stored derivative fixed array"
            raise exceptions.InputError(new_waypoint, msg)

        # modify waypoints
        self.waypoints = np.insert(self.waypoints,new_index,np.array(new_waypoint),axis=1)

        # modify der_fixed
        self.der_fixed = np.insert(self.der_fixed,new_index,np.array(new_der_fixed),axis=1)
        # Correct der fixed around it
        if new_index == 0:
            self.der_fixed[1:,1] = False
        elif new_index == (self.n_seg + 1):
            self.der_fixed[1:,-2] = False

        # modify n_seg - increase by 1
        self.n_seg += 1

        # modify times - add a time and change time for segments on either side of new point
        if new_index > self.times.size:
            self.times = np.append(self.times,new_times[0])
        else:
            self.times = np.insert(self.times,new_index,new_times[1])

        if new_index != 0 and new_index <= self.times.size-1:
            self.times[new_index-1] = new_times[0]

        # modify Q and A matrices
        self.Q = cost.insert(self.Q,new_index=new_index,new_times=new_times,costs=self.costs,order=self.order)
        self.A = constraint.insert(self.A,new_index=new_index,new_times=new_times,order=self.order)

        # Get inverse A
        if (self.A.shape[0] == self.A.shape[1]):
            pinv_A = sp.sparse.linalg.inv(self.A)
        else:
            # is there some way to keep this sparse? sparse SVD?
            pinv_A = np.asmatrix(sp.linalg.pinv(self.A.todense()))

        self.pinv_A = pinv_A

        if not defer:
            self.get_free_ders(waypoints_changed=None,
                               der_fixed_changed=None,
                               times_changed=np.array([]))
            self.get_poly_coeffs()
            self.get_piece_poly()


    def delete(self, delete_index, new_time, defer=False):
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

            Everything modified by the following functions:
                self.get_cost()
                self.get_piece_poly()

        Args:
            delete_indices: indices of waypoints to delete
            new_time: duration of segment that joins the waypoints on each side
                of the deletion

        Returns:

        Raises:
        """

        # Check input

        n_seg = self.n_seg

        # Take out waypoint and der_fixed
        mask = np.ones(n_seg+1,dtype=bool)
        mask[delete_index] = False
        self.waypoints = self.waypoints[:,mask]
        self.der_fixed = self.der_fixed[:,mask]
        # Correct der_fixed around it
        if delete_index == 0:
            self.der_fixed[:,0] = True
        if delete_index == self.n_seg:
            self.der_fixed[:,-1] = True

        # modify n_seg - decrease by 1
        self.n_seg -= 1

        # modify times - add a time and change time for segments on either side of new point
        if delete_index < self.times.size:
            self.times = np.delete(self.times,delete_index)
        else:
            # Delete the last term
            self.times = np.delete(self.times,self.times.size-1)

        if delete_index != 0 and delete_index < self.times.size+1:
            self.times[delete_index-1] = new_time

        # modify Q and A matrices
        self.Q = cost.delete(self.Q,delete_index=delete_index,new_time=new_time,costs=self.costs,order=self.order)
        self.A = constraint.delete(self.A,delete_index=delete_index,new_time=new_time,order=self.order)

        # Get inverse A
        if (self.A.shape[0] == self.A.shape[1]):
            pinv_A = sp.sparse.linalg.inv(self.A)
        else:
            # is there some way to keep this sparse? sparse SVD?
            pinv_A = np.asmatrix(sp.linalg.pinv(self.A.todense()))

        self.pinv_A = pinv_A


        if not defer:
            self.get_free_ders(waypoints_changed=None,
                               der_fixed_changed=None,
                               times_changed=np.array([]))
            self.get_poly_coeffs()
            self.get_piece_poly()


    def prepend(self, new_poly):
        """Utility function for PPoly of different order"""
        times = np.r_[new_poly.times, self.times]
        trans_times = utils.seg_times_to_trans_times(times)

        order_delta = np.shape(self.coeffs)[0] - np.shape(new_poly.coeffs)[0]

        if order_delta < 0:
            coeffs = np.c_[new_poly.coeffs, np.pad(self.coeffs,
                           ((-order_delta, 0), (0, 0)),
                           'constant', constant_values=(0))]
        elif order_delta > 0:
            coeffs = np.c_[np.pad(new_poly.coeffs,
                           ((order_delta, 0), (0, 0)),
                           'constant', constant_values=(0)),
                           self.coeffs]
        else:
            coeffs = np.c_[new_poly.coeffs, self.coeffs]

        piece_poly = sp.interpolate.PPoly(coeffs, trans_times, extrapolate=False)

        return piece_poly


    def append(self, new_poly):
        """Utility function for PPoly of different order"""
        times = np.r_[self.times, new_poly.times]
        trans_times = utils.seg_times_to_trans_times(times)

        order_delta = np.shape(self.coeffs)[0] - np.shape(new_poly.coeffs)[0]

        if order_delta < 0:
            coeffs = np.c_[self.coeffs, np.pad(new_poly.coeffs,
                           ((-order_delta, 0), (0, 0)),
                           'constant', constant_values=(0))]
        elif order_delta > 0:
            coeffs = np.c_[np.pad(self.coeffs,
                           ((order_delta, 0), (0, 0)),
                           'constant', constant_values=(0)),
                           new_poly.coeffs]
        else:
            coeffs = np.c_[self.coeffs, new_poly.coeffs]

        piece_poly = sp.interpolate.PPoly(coeffs, trans_times, extrapolate=False)

        return piece_poly


    def get_cost(self, times, times_changed=None, defer=False):
        """Utility function to calculate trajectory cost

        Richter, Bry, Roy; Polynomial Trajectory Planning for Quadrotor Flight
        Equation 31

        Uses:
            Everything used by the following functions:
                self.get_free_ders()
                self.get_poly_coeffs()

        Modifies:
            self.
            times: time in which to complete each segment (not the transition
                times)

            Everything modified by the following functions:
                self.get_free_ders()
                self.get_poly_coeffs()

        Args:

        Returns:
            self.
            piece_poly_cost: cost for integral of squared norm
                of specified derivatives

        Raises:
            minsnap.exceptions.InputError if
                np.size(times) != np.shape(der_fixed)[1] - 1
        """

        if np.size(times) != np.shape(self.der_fixed)[1] - 1:
            raise exceptions.InputError(times,
                                        "Mismatch between number of segment times"
                                        + " and number of segments")

        self.times = np.array(times).copy()

        # for reuse by update_times()
        if not defer:
            self.get_free_ders(waypoints_changed=np.array([]),
                               der_fixed_changed=np.array([]),
                               times_changed=times_changed)
            self.get_poly_coeffs()

            return self.piece_poly_cost

    def get_cost_gradient(self, times, have_cost=False):
        # did we just calculate cost?
        if not have_cost:
            self.times = times
            self.get_free_ders()
            self.get_poly_coeffs()

        for i in range(0, self.n_seg):

            pass


    def update_waypoint(self, index, new_waypoint, new_der_fixed=None, defer=False):
        """Set new waypoint value and whether derivatives are fixed

        Uses:

        Modifies:
            self.
            waypoints:
            der_fixed:

        Args:
            new_waypoint: array of waypoint derivatives with length equal to
                n_der, the number of free derivatives given the polynomial order
            new_der_fixed: Boolean array of n_der (max potential number of free
                derivatives per segment) x 1 (one column per waypoint).
                Fixing a derivative at a waypoint is performed by setting the
                corresponding entry to True.

        Returns:

        Raises:
            minsnap.exceptions.InputError if
                np.size(waypoint) != np.shape(self.waypoints)[1]

            minsnap.exceptions.InputError if
                np.size(der_fixed) != np.shape(self.der_fixed)[1]
        """

        if new_der_fixed is not None:
            if np.size(new_der_fixed) != np.shape(self.der_fixed)[0]:
                msg = "Mismatch between length of new derivative fixed array "
                msg = msg + "and size of 2nd dimension of stored derivative "
                msg = msg + "fixed array"
                raise exceptions.InputError(new_der_fixed, msg)
            der_fixed_changed = np.array(index).copy()
        else:
            der_fixed_changed = np.array([])

        if np.size(new_waypoint) != np.shape(self.der_fixed)[0]:
            msg = "Mismatch between length of new waypoint array and size of"
            msg = msg + "2nd dimension of stored derivative fixed array"
            raise exceptions.InputError(new_waypoint, msg)

        if np.size(der_fixed_changed):
            self.der_fixed[:, index] = np.array(new_der_fixed).copy()
        self.waypoints[:, index] = np.array(new_waypoint).copy()

        if not defer:
            self.get_free_ders(waypoints_changed=np.array(index),
                               der_fixed_changed=der_fixed_changed,
                               times_changed=np.array([]))
            self.get_poly_coeffs()
            self.get_piece_poly()


    def update_waypoints(self, new_waypoints, new_der_fixed=None, defer=False):
        """Set new waypoint values and whether derivatives are fixed

        Uses:

        Modifies:
            self.
            waypoints:
            der_fixed:

        Args:
            new_waypoints: full array of waypoints and derivatives with the same
                size as the internal der_fixed array
            new_der_fixed: Boolean array of n_der (max potential number of free
                derivatives per segment) x n_seg + 1 (one column per waypoint).
                Fixing a derivative at a waypoint is performed by setting the
                corresponding entry to True.

        Returns:

        Raises:
            minsnap.exceptions.InputError if
                np.shape(waypoints) != np.shape(self.der_fixed)
        """

        if new_der_fixed is not None:
            if np.shape(new_der_fixed) != np.shape(self.der_fixed):
                raise exceptions.InputError(new_der_fixed,
                        "Mismatch between shape of new derivative fixed"
                        + " array and shape of stored derivative fixed"
                        + "array")
        else:
            # this is for convenience only (we are not updating der_fixed)
            new_der_fixed = self.der_fixed

        if np.shape(new_waypoints) != np.shape(new_der_fixed):
            raise exceptions.InputError(new_waypoints,
                    "Mismatch between shape of new waypoints"
                    + " array and shape of new derivative fixed"
                    + "array")

        self.der_fixed = np.array(new_der_fixed).copy()
        self.waypoints = np.array(new_waypoints).copy()

        if not defer:
            # update all data structures
            self.get_free_ders()
            self.get_poly_coeffs()
            self.get_piece_poly()


    def update_times(self, times, defer=False, times_changed=None):
        """Generate new PPoly based on new segment times

        Richter, Bry, Roy; Polynomial Trajectory Planning for Quadrotor Flight
        Equation 31

        Uses:
            Everything used by the following functions:
                self.get_cost()
                self.get_piece_poly()

        Modifies:
            self.
            times: time in which to complete each segment (not the transition
                times)

            Everything modified by the following functions:
                self.get_cost()
                self.get_piece_poly()

        Args:
            times: time in which to complete each segment (not the transition
                times)

        Returns:

        Raises:
            minsnap.exceptions.InputError if
                np.size(times) != np.shape(der_fixed)[1] - 1
        """

        self.get_cost(times, defer=defer, times_changed=times_changed)

        if not defer:
            self.get_piece_poly()


    #TODO(mereweth@jpl.nasa.gov) - implement update methods for cost
    # and constraint; change these to be lists of indices
    def get_free_ders(self, waypoints_changed=None,
                      der_fixed_changed=None,
                      times_changed=None):
        """Solve for free derivatives of polynomial segments

        Richter, Bry, Roy; Polynomial Trajectory Planning for Quadrotor Flight
        Equation 31

        Uses:
            self.
            waypoints: Numpy array of the waypoints (including derivatives)
            times: time in which to complete each segment (not the transition
                times)
            cost: weight in the sum of Hessian matrices (Equation 15)
            order: the order of the polynomial segments
            der_fixed: Boolean array of n_der (max potential number of free
                derivatives per segment) x n_seg + 1 (one column per waypoint).
                Fixing a derivative at a waypoint is performed by setting the
                corresponding entry to True.

        # TODO(mereweth@jpl.nasa.gov) - decide how many of these fields to store vs
        # recompute
        Modifies:
            self.
            free_ders: Numpy matrix (column vector) of the free derivatives
            fixed_ders: Numpy matrix (column vector) of the fixed derivatives
            Q: the cost matrix in terms of polynomial coefficients
            M: the selector matrix mapping the ordering of derivatives to/from
                the form where free and fixed derivatives are partitioned
            A: the constraint matrix for segment boundaries
            pinv_A: the (pseudo-)inverse of A
            R: the cost matrix in terms of the free derivatives

        Args:

        Returns:

        Raises:
            minsnap.exceptions.InputError if
                np.size(times) != np.shape(der_fixed)[1] - 1
        """


    # def get_free_ders_setup_matrices(self, waypoints_changed=None,
    #                   der_fixed_changed=None,
    #                   times_changed=None):
        start_timer = time.time()

        if times_changed is None:
            times_changed = np.r_[0:self.n_seg]
        if der_fixed_changed is None:
            der_fixed_changed = np.r_[0:self.n_seg + 1]
        if waypoints_changed is None:
            waypoints_changed = np.r_[0:self.n_seg + 1]

        waypoints = self.waypoints
        times = self.times
        costs = self.costs
        order = self.order
        der_fixed = self.der_fixed
        der_ineq = self.der_ineq
        delta = self.delta
        n_seg = self.n_seg
        n_der = utils.n_coeffs_free_ders(order)[1]
        n_ineq = sum(sum(der_ineq))

        if np.size(times) != np.shape(der_fixed)[1] - 1:
            raise exceptions.InputError(times,
                                        "Mismatch between number of segment times"
                                        + " and number of segments")

        if np.shape(waypoints) != np.shape(der_fixed):
            raise exceptions.InputError(waypoints,
                                        "Mismatch between size of waypoints"
                                        + " array and size of derivative fixed"
                                        + "array")

        waypoints = np.matrix(waypoints)

        if n_ineq > 0:
            #TODO(mereweth@jpl.nasa.gov & bmorrell@jpl.nasa.gov) - investigate other ways to prevent singular matrices here
            limit = 0.03
            if sum(np.array(times)<limit)>0:
                print("Warning: changing times because a lower limit has been reached")
                temp = np.array(times)
                temp[temp<limit] = limit
                times = np.array(temp)
                self.times = times


        # See README.md on MATLAB vs Octave vs Numpy linear equation system solving
        # Porting this code: R{i} = M{i} / A{i}' * Q{i} / A{i} * M{i}';
        # With even polynomial order, the odd number of coefficients is not equal
        # to twice the number of free derivatives (floor of (odd # / 2)).
        # Therefore, A is not square.

        # Based on some quick checks, the inverse of A is still quite sparse,
        # especially when only the 0th derivative is fixed at internal
        # waypoints.

        # convert COO sparse output to CSC for fast matrix arithmetic
        if np.size(times_changed) or self.Q is None or self.A is None:
            if (self.Q is None or self.A is None): #or np.size(times_changed) > 1):
                Q = cost.block_cost(times, costs, order).tocsc(copy=True)
                A = constraint.block_constraint(times, order).tocsc(copy=True)
            else:
                # if we know which segment times have changed, only update
                # the corresponding blocks in the block matrix
                cost.update(self.Q, times_changed, times[times_changed], costs, order)
                constraint.update(self.A, times_changed, times[times_changed], order)
                Q = self.Q
                A = self.A

            if (A.shape[0] == A.shape[1]):
                pinv_A = sp.sparse.linalg.inv(A)
                # TODO(mereweth@jpl.nasa.gov) - could this be made to outperform the line above?
                #pinv_A = constraint.invert(A, order)
            else:
                # is there some way to keep this sparse? sparse SVD?
                pinv_A = np.asmatrix(sp.linalg.pinv(A.todense()))
        else:
            # TODO(mereweth@jpl.nasa.gov) - is this the cleanest way?
            Q = self.Q
            A = self.A
            pinv_A = self.pinv_A

        if np.size(der_fixed_changed) or self.M is None:
            M = selector.block_selector(der_fixed,closed_loop=self.closed_loop).tocsc(copy=True)
        else:
            # TODO(mereweth@jpl.nasa.gov) - is this the cleanest way?
            M = self.M

        if np.size(times_changed) or np.size(der_fixed_changed) or self.R is None:
            # all are matrices; OK to use * for multiply
            # R{i} = M{i} * pinv(full(A{i}))' * Q{i} * pinv(full(A{i})) * M{i}';
            R = M * pinv_A.T * Q * pinv_A * M.T

            # partition R by slicing along columns, converting to csr, then slicing
            # along rows
            if self.closed_loop:
                num_der_fixed = np.sum(der_fixed[:,:-1])
            else:
                num_der_fixed = np.sum(der_fixed)

            # Equation 29
            try:
                R_free = R[:, num_der_fixed:].tocsr()
            except AttributeError:
                # oops - we are a numpy matrix
                R_free = R[:, num_der_fixed:]
            R_fixed_free = R_free[:num_der_fixed, :]
            R_free_free = R_free[num_der_fixed:, :]

        else:
            R = self.R

            if hasattr(self,'R_fixed_free'):
                R_fixed_free = self.R_fixed_free
                R_free_free = self.R_free_free
            else:
                # partition R by slicing along columns, converting to csr, then slicing
                # along rows
                if self.closed_loop:
                    num_der_fixed = np.sum(der_fixed[:,:-1])
                else:
                    num_der_fixed = np.sum(der_fixed)

                # Equation 29
                try:
                    R_free = R[:, num_der_fixed:].tocsr()
                except AttributeError:
                    # oops - we are a numpy matrix
                    R_free = R[:, num_der_fixed:]
                R_fixed_free = R_free[:num_der_fixed, :]
                R_free_free = R_free[num_der_fixed:, :]


        # TODO(mereweth@jpl.nasa.gov) - transpose waypoints and der_fixed at input
        if not self.closed_loop:
            fixed_ders = waypoints.T[np.nonzero(der_fixed.T)].T# Equation 31
        else:
            fixed_ders = waypoints[:,:-1].T[np.nonzero(der_fixed[:,:-1].T)].T# Equation 31

        """ Solve """
        # the fixed derivatives, D_F in the paper, are just the waypoints for which
        # der_fixed is true
        # DP{i} = -RPP \ RFP' * DF{i};

        if n_ineq == 0:
            # Run for unconstrained case:
            # Solve for the free derivatives
            # Solve the unconstrained system
            free_ders = np.asmatrix(sp.sparse.linalg.spsolve(R_free_free,
                                                             -R_fixed_free.T * fixed_ders)).T

        # Run for Constrained cases
        elif n_ineq>0: # If there are inequality constraints
            # Inequalities
            # Isolate the waypoints for which there is an inequality
            ineq_way_p = waypoints.T[np.nonzero(der_ineq.T)].T

            if type(delta)!=float:
                # Take out the delta for the inequality constrained parts
                ineq_delta = delta.T[np.nonzero(der_ineq.T)].reshape([ineq_way_p.shape[0],1])
            else:
                # Just a constant
                ineq_delta = delta

            W = np.concatenate(((ineq_way_p-ineq_delta),(-ineq_way_p-ineq_delta),),axis=0)

            # Selector matix
            if np.size(der_fixed_changed) or not hasattr(self,'E'):
                E = constraint.block_ineq_constraint(der_fixed,der_ineq).tocsc(copy=True)
            elif self.E is None:
                E = constraint.block_ineq_constraint(der_fixed,der_ineq).tocsc(copy=True)
            else:
                E = self.E

            ### Solve with CVXOPT
            # Setup matrices
            P = matrix(2*R_free_free.toarray(),tc='d')
            q_vec = matrix(np.array(2*fixed_ders.T*R_fixed_free).T,tc='d')
            G = matrix(-E.toarray().astype(np.double),tc='d')
            h_vec = matrix(np.array(-W),tc='d')

            # Checks on the problem setup
            if np.linalg.matrix_rank(np.concatenate((P,G))) < P.size[0]:
                print('Warning: rank of [P;G] {} is less than size of P: {}'.format(np.linalg.matrix_rank(np.concatenate((P,G))),P.size[0]))
            else:
                if self.print_output:
                    print('Rank of A is {} size is {}\nRank for Q is {}, size is {}'.format(np.linalg.matrix_rank(A.toarray()),A.shape,np.linalg.matrix_rank(Q.toarray()),Q.shape))
                    print('Rank of R is {}, size is {}'.format(np.linalg.matrix_rank(R.toarray()),R.shape))
                    print('Rank P is {}\nRank of G is: {}\nRank of [P;G] {} size of P: {}\n'.format(np.linalg.matrix_rank(P),np.linalg.matrix_rank(G),np.linalg.matrix_rank(np.concatenate((P,G))),P.size[0]))

            # To suppress output
            solvers.options['show_progress'] = False

            # Run cvxopt solver
            sol = solvers.qp(P,q_vec,G,h_vec)#,initvals=primalstart)

            # Solution
            free_ders = np.matrix(sol['x'])

            if self.print_output:
                print('cvx solution is:\n{}'.format(free_ders))
                print('cvx cost is:\n{}'.format(sol['primal objective']))
                print('Constraints are: \n{}'.format(E*free_ders - W))



        # self.fixed_terms = fixed_terms
        self.opt_time = time.time() - start_timer
        self.free_ders = free_ders
        self.fixed_ders = fixed_ders
        self.Q = Q
        self.M = M
        self.A = A
        self.pinv_A = pinv_A
        self.R = R
        self.R_fixed_free = R_fixed_free
        self.R_free_free = R_free_free


    def get_poly_coeffs(self):
        """Get the polynomial coefficients and the cost for a trajectory

        Uses:
            self.

            times: time in which to complete each segment (not the transition
                times)
            order: the order of the polynomial segments
            free_ders: Numpy array of the free derivatives
            fixed_ders: Numpy array of the fixed derivatives
            Q: the cost matrix in terms of polynomial coefficients
            M: the selector matrix mapping the ordering of derivatives to/from
                the form where free and fixed derivatives are partitioned
            A: the constraint matrix for segment boundaries
            R: the cost matrix in terms of the free derivatives

        Modifies:
            self.

            coeffs: piecewise polynomial coefficients
            piece_poly_cost: The Hessian cost function evaluated for the polynomial

        Args:

        Returns:

        Raises:
        """

        Q = self.Q
        M = self.M
        A = self.A
        pinv_A = self.pinv_A
        R = self.R

        order = self.order
        free_ders = self.free_ders
        fixed_ders = self.fixed_ders

        if (A.shape[0] == A.shape[1]):
            # inv(A{i}) * M{i}' * [DF{i}; DP{i}];
            # TODO(mereweth@jpl.nasa.gov) - do we need to use pinv_A here?
            coeffs = np.asmatrix(sp.sparse.linalg.spsolve(A,
                                                          M.T * np.r_[fixed_ders, free_ders])).T
        else:
            # p = pinv(A{i}) * M{i}' * [DF{i}; DP{i}];
            coeffs = np.asmatrix(pinv_A * M.T * np.r_[fixed_ders, free_ders])

        self.piece_poly_cost = (coeffs.T * Q * coeffs)[0,0]

        # highest order coefficient first; horizontal axis is segments

        self.coeffs = np.flip(np.asmatrix(np.reshape(coeffs, (-1, order + 1))).T, 0)


    def get_piece_poly(self):
        """Get the piecewise polynomial from the coefficients

        Uses:
            self.

            coeffs: piecewise polynomial coefficients
            times: time in which to complete each segment (not the transition
                times)

        Modifies:
            self.

            piece_poly: An instance of scipy.interpolate.PPoly

        Args:

        Returns:

        Raises:
        """

        trans_times = utils.seg_times_to_trans_times(self.times)
        self.piece_poly = sp.interpolate.PPoly(
            self.coeffs, trans_times, extrapolate=False)


    def check_continuity(self, eps=1e-6, equal_eps=1e-3):
        """Checks that the piecewise polynomial and its derivatives are continuous

        Uses:
            self.

            piece_poly: piecewise-continuous polynomial of segments
            times: time in which to complete each segment (not the transition
                times)

        Modifies:
            self.

            piece_poly: An instance of scipy.interpolate.PPoly

        Args:
            eps: epsilon value specifying how far on each side of each
                transition time to set the data point for continuity checking
            equal_eps: epsilon value specifying how close the values have to be
                in order to be considered equal

        Returns:

        Raises:
        """

        temp_ppoly = self.piece_poly

        # ppoly has no data before and after first transition points, so we
        # can't check those
        trans_times = utils.seg_times_to_trans_times(self.times)[1:-1]

        for i in range(self.n_der):
            if not np.allclose(temp_ppoly(trans_times - eps),
                               temp_ppoly(trans_times + eps), equal_eps):
                return False
            temp_ppoly = temp_ppoly.derivative()

        return True


    def create_n_laps(self,n_laps,entry_ID,exit_ID):
        """

        Behaviour: Will no n_laps, where the last lap will not be a complete loop,
        instead it will finish at the exit_ID waypoint

        """
        coeffs = self.coeffs.copy()
        times = self.times.copy()

        if entry_ID < exit_ID or (entry_ID==exit_ID and entry_ID == 0):
            # Entry ID earlier - repeat for number of laps
            new_coeffs = np.tile(coeffs,(1,n_laps))
            new_times = np.tile(times,n_laps)
        else:
            # Entry ID later, so need to add another lap
            new_coeffs = np.tile(coeffs,(1,n_laps+1))
            new_times = np.tile(times,n_laps+1)

        # new_coeffs = np.append(new_coeffs,new_coeffs[:,0],axis=1)

        # Delete from the start to get correct start ID
        new_coeffs = np.delete(new_coeffs,np.arange(entry_ID),axis=1)
        new_times = np.delete(new_times,np.arange(entry_ID))

        # Delete from the end to get the correct exit ID
        if exit_ID != 0:
            new_coeffs = np.delete(new_coeffs,np.arange(new_coeffs.shape[1]-(coeffs.shape[1]-exit_ID),new_coeffs.shape[1]),axis=1)
            new_times =  np.delete(new_times,np.arange(new_times.shape[0]-(times.shape[0]-exit_ID),new_times.shape[0]))
            # Do nothing if zero is the exit ID (zero is by default the final term)

        # Get the piecewise polynomial
        trans_times = utils.seg_times_to_trans_times(new_times)
        piece_poly = sp.interpolate.PPoly(new_coeffs, trans_times, extrapolate=False)
        #
        # # Check continuity

        return piece_poly


# cost, gradient, hessian
# equation 30 and derivatives, in case we want to do constrained optimization
# using another cost function in concert

def main():
    costs = [0, 0, 0, 0, 1]
    order = max(np.nonzero(costs)[0]) * 2 - 1
    n_der = utils.n_coeffs_free_ders(order)[1]

    num_internal = 1#98
    # float derivatives at internal waypoints and fix at beginning and end
    inner = [[True] + [False] * num_internal + [True]] * (n_der - 1)
    # fix 0th derivative at internal waypoints
    der_fixed = [[True] * (num_internal + 2)]
    der_fixed.extend(inner)

    # waypoints = [range(num_internal + 2)]
    waypoints = [[0,1,0]]
    waypoints.extend([[0] * (num_internal + 2)] * (n_der - 1))

    times = [1] * (num_internal + 1)
    print("\nTrajectory for:\nWaypoints:\n")
    pp(np.array(waypoints))
    print("\nSegment times:\n")
    pp(np.array(times))
    print("\ncost {}; poly order {};\n".format(costs, order))
    result = PolyTraj(waypoints, order, costs, der_fixed, times,closed_loop=True)
    print(result.free_ders)

    if not result.check_continuity():
        print("\nFailed continuity check")

    costs = [0, 0, 0, 0, 1]
    num_internal = 0

    # give the optimization enough free derivatives
    order = max(np.nonzero(costs)[0] + 1) * 2 - 1
    n_der = utils.n_coeffs_free_ders(order)[1]
    print(n_der)

    num_internal = 4
    # float derivatives at internal waypoints and fix at beginning and end
    inner = [[True] + [False] * num_internal + [True]] * (n_der - 1)
    # fix 0th derivative at internal waypoints
    der_fixed = [[True] * (num_internal + 2)]
    der_fixed.extend(inner)
    print(der_fixed)
    waypoints = [range(num_internal + 2)]
    waypoints.extend([[0] * (num_internal + 2)] * (n_der - 1))

    times = [0.2] * (num_internal + 1)
    print("\nTrajectory for:\nWaypoints:\n")
    pp(np.array(waypoints))
    print("\nSegment times:\n")
    pp(np.array(times))
    print("\ncost {}; poly order {};\n".format(costs, order))

    result = PolyTraj(waypoints, order, costs, der_fixed, times)
    print(result.free_ders)

    # CHANGE FOR INEQUALITY CONSTRAINTS
    delta = 0.2
    der_ineq = np.array(der_fixed)
    der_ineq[:,[0,-1]] = False
    # der_ineq[:,:] = False
    der_fixed = np.array(der_fixed)
    der_fixed[0,1:-1] = False
    print("fixed mat = \n{}\n".format(der_fixed))
    print("ineq mat = \n{}\n".format(der_ineq))
    result2 = PolyTraj(waypoints, order, costs, der_fixed, times, der_ineq, delta)
    result2.update_times(times)
    # print(result.free_ders)

    # costs = [0, 0, 0, 0, 1]
    # num_internal = 0
    #
    # # give the optimization enough free derivatives
    # order = max(np.nonzero(costs)[0] + 1) * 2 - 1
    # n_der = utils.n_coeffs_free_ders(order)[1]
    #
    # # float derivatives at internal waypoints and fix at beginning and end
    # inner = [[True] + [False] * num_internal + [True]] * (n_der - 1)
    # # fix 0th derivative at internal waypoints
    # der_fixed = [[True] * (num_internal + 2)]
    # der_fixed.extend(inner)
    #
    # waypoints = [range(num_internal + 2)]
    # waypoints.extend([[0] * (num_internal + 2)] * (n_der - 1))
    #
    # times = [1] * (num_internal + 1)
    # print("\nTrajectory for:\nWaypoints:\n")
    # pp(np.array(waypoints))
    # print("\nSegment times:\n")
    # pp(np.array(times))
    # print("\ncost {}; poly order {};\n".format(costs, order))
    #
    # result = PolyTraj(waypoints, order, costs, der_fixed)
    # result.update_times(times)
    # print(result.free_ders)

    # Test new functions
    costs = [0, 0, 0, 0, 1]
    order = max(np.nonzero(costs)[0]) * 2 - 1
    n_der = utils.n_coeffs_free_ders(order)[1]

    num_internal = 3
    # float derivatives at internal waypoints and fix at beginning and end
    inner = [[True] + [False] * num_internal + [True]] * (n_der - 1)
    # fix 0th derivative at internal waypoints
    der_fixed = [[True] * (num_internal + 2)]
    der_fixed.extend(inner)

    waypoints = np.zeros([n_der,num_internal+2])
    waypoints[0,:] = np.arange(0,num_internal + 2,1.0)

    times = [1.0] * (num_internal + 1)

    pol = PolyTraj(waypoints, order, costs, der_fixed, times)

    new_index = waypoints.shape[1]-1
    new_waypoint = pol.waypoints[:,new_index].copy()
    new_waypoint[0] = 1.5
    new_times = [result.times[new_index]/2,result.times[new_index]/2]
    new_der_fixed = pol.der_fixed[:,new_index].copy()


    a1 = pol.free_ders.copy()
    pol.insert(new_index+1, new_waypoint, new_times, new_der_fixed, defer=False)
    a2 = pol.free_ders.copy()
    pol.delete(new_index+1, 1.0, defer=False)
    a3 = pol.free_ders.copy()

    import pdb; pdb.set_trace()
    costs = [0, 0, 0, 1]
    order = max(np.nonzero(costs)[0]) * 2 - 1
    waypoints_n = np.zeros((3,5))
    waypoints_n[0,:] = [0,1.0,2.0,3.0,0]
    der_fixed_n = np.zeros(waypoints_n.shape,dtype=bool)
    # der_fixed_n[:,[0,2]] = True
    der_fixed_n[0,:] = True
    times = np.ones(waypoints_n.shape[1]-1)
    pol = PolyTraj(waypoints_n,order, costs, der_fixed_n, times, closed_loop=True)
    n_laps = 2
    entry_ID = 3
    exit_ID = 1
    import pdb; pdb.set_trace()
    out_pol = pol.create_n_laps(n_laps,entry_ID,exit_ID)


    import pdb; pdb.set_trace()
    if not result.check_continuity():
        print("\nFailed continuity check")


if __name__ == '__main__':
    main()
