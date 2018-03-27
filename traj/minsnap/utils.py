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

import yaml
import string
import os

import rdp
import numpy as np
import copy
import scipy.interpolate
sp = scipy

from minsnap import exceptions
from minsnap.settings import STATE_INFO_TOKEN, FIELD_ORDER_TOKEN, TANGO_MAP_T_BODY_FIELDS

try:
    from px4_msgs.msg import PolyTraj as PolyTraj_msg
    HAS_POLYTRAJ_MSG = True
except ImportError:
    HAS_POLYTRAJ_MSG = False

import transforms3d
from transforms3d import euler
from transforms3d import quaternions


# TODO(mereweth@jpl.nasa.gov) - refactor into general utils?
class InputWrapper(object):
    def __init__(self, func):
        self.x = None
        self.func = func

    def wrap_func(self, new_x):
        # determine numerically which x values have changed
        if self.x is None:
            # None as second argument must mean that all x values changed
            x_changed = None
        else:
            x_changed = np.isclose(self.x, new_x)

        self.x = new_x
        return self.func(self.x, x_changed)


# TODO(mereweth@jpl.nasa.gov) - refactor into general utils?
def spline_seed_traj(orig_waypoints):#, epsilon):
    #temp_waypoints = rdp(np.c_[orig_waypoints['x'],
    #                           orig_waypoints['y'],
    #                           orig_waypoints['z']], epsilon=epsilon)
    temp_waypoints = np.c_[orig_waypoints['x'],
                           orig_waypoints['y'],
                           orig_waypoints['z']]
    x = np.array(temp_waypoints[:, 0])
    y = np.array(temp_waypoints[:, 1])
    z = np.array(temp_waypoints[:, 2])

    #dist = np.sqrt((x[:-1] - x[1:]) ** 2 +
    #               (y[:-1] - y[1:]) ** 2 +
    #               (z[:-1] - z[1:]) ** 2)
    #path_length = np.r_[0, np.cumsum(dist)]
    #spline = sp.interpolate.splprep([x, y, z], u=path_length, s=0)

    spline, path_length = sp.interpolate.splprep([x, y, z])

    return (spline, path_length)


# TODO(mereweth@jpl.nasa.gov) - refactor into general utils?
def write_poly_coeffs(qr_polytraj, filename):
    if HAS_POLYTRAJ_MSG:
        fields = string.split(PolyTraj_msg._full_text, '\n')
        header = string.join(["# " + f for f in fields], '\n')
        msg = PolyTraj_msg()
        qr_polytraj.fill_poly_traj_msg(msg)

        with open(filename, 'w') as filp:
            filp.write(header + '\n' + str(msg))


# TODO(mereweth@jpl.nasa.gov) - refactor into general utils?
def load_waypoints_from_tango(filename):
    # TODO(mereweth@jpl.nasa.gov) - should this function raise an exception on
    # invalid format?

    data = np.loadtxt(filename, delimiter=',')

    waypoints = dict(yaw=None)
    waypoints['x'] = data[:, 1]
    waypoints['y'] = data[:, 2]
    waypoints['z'] = data[:, 3]
    # TODO(mereweth@jpl.nasa.gov) - submit PR for vectorized transforms3d
    # and use that here to generate yaw

    return waypoints


# TODO(mereweth@jpl.nasa.gov) - refactor into general utils?
def load_waypoints(filename):
    with open(filename, 'r') as filp:
        data = yaml.load(filp)

    attitude = dict()
    waypoints = dict(yaw=None)
    for i in range(len(data['states'][0]['x'])):
        field_tokens = data[STATE_INFO_TOKEN]['x'][FIELD_ORDER_TOKEN]
        if field_tokens[i] in ['x', 'y', 'z']:
            waypoints[field_tokens[i]] = [state['x'][i]
                                          for state in data['states']]

    waypoints['yaw'] = np.array([])
    for state in data['states']:
        for i in range(len(data['states'][0]['x'])):
            field_tokens = data[STATE_INFO_TOKEN]['x'][FIELD_ORDER_TOKEN]
            if field_tokens[i] == 'q_w':
                attitude = state['x'][i:(i+4)]
                yaw = transforms3d.euler.quat2euler(attitude,'rzyx')[0]
                waypoints['yaw'] = np.concatenate((waypoints['yaw'],[yaw]))


    return waypoints

# TODO(mereweth@jpl.nasa.gov) - refactor into general utils?
def save_waypoints(waypoints, filename):
    header = dict()
    header[STATE_INFO_TOKEN] = dict()
    header[STATE_INFO_TOKEN]['x'] = dict(units=["meters", "meters", "meters"])
    header[STATE_INFO_TOKEN]['x'][FIELD_ORDER_TOKEN] = ["x", "y", "z"]

    zipped = zip(waypoints['x'].tolist(), waypoints['y'].tolist(), waypoints['z'].tolist())
    data = [dict(x=list(point)) for point in zipped]

    with open(filename, 'w') as filp:
        filp.write(yaml.dump(header, default_flow_style=None))
        filp.write("states:\n")

        filp.write(yaml.dump(data, default_flow_style=None))


# TODO(mereweth@jpl.nasa.gov) - refactor into general utils?
def seg_times_to_trans_times(seg_times):
    """
    Get the transition times from per-segment times

    Args:
        seg_times: time to complete each segment

    Returns:
        trans_times: time at which to transition to each segment
            from the beginning of the trajectory

    Raises:
    """
    trans_times = np.r_[0, np.cumsum(seg_times)]
    return trans_times


# TODO(mereweth@jpl.nasa.gov) - refactor into general utils?
def trans_times_to_seg_times(trans_times):
    """
    Get the per-segment times from transition times

    Args:
        trans_times: time at which to transition to each segment
            from the beginning of the trajectory

    Returns:
        seg_times: time to complete each segment

    Raises:
    """
    seg_times = np.diff(trans_times)
    return seg_times


# TODO(mereweth@jpl.nasa.gov) - refactor into general utils?
def get_seed_times(poses, avg_vel_des):
    """
    Get the seed times for a given set of distances between points

    Args:
        poses: a dictionary with keys x, y, z and values that are lists
            for each coordinate
        avg_vel_des: the desired average velocity

    Returns:
        seed_times: the seed time for each segment before relative optimization

    Raises:
        minsnap.exceptions.InputError if the input values are of incorrect
            sizes or shapes
    """

    for key in poses.keys():
        if len(np.shape(poses[key])) != 1:
            raise exceptions.InputError(poses[key],
                                        "Only pass the 0th derivatives to"
                                        + " minsnap.utils.get_seed_times"
                                        + " as an array with 1 dimension")

    diff_norms = np.sqrt(np.diff(poses['x']) ** 2.0 +
                         np.diff(poses['y']) ** 2.0 + np.diff(poses['z']) ** 2.0)

    return diff_norms / avg_vel_des


# TODO(mereweth@jpl.nasa.gov) - refactor into general utils?
def join_poly(to_extend, extend_by):
    """
    Extend a polynomial trajectory

    Args:
        to_extend: a scipy.interpolate.PPoly that will be extended
        extend_by: a scipy.interpolate.PPoly to extend by

    Returns:

    Raises:
    """

    # TODO(mereweth@jpl.nasa.gov)
    # check that all derivatives up to the order/2 are continuous
    # at the polynomial boundary

    t_final = to_extend.c[-1]

    to_extend.extend(extend_by.c + t_final, extend_by.x, right=true)


# TODO(mereweth@jpl.nasa.gov) - refactor into general utils?
def nested_copy(references_2d):
    """
    Strip out references from the 1st dimension of a 2d nested list

    Args:
        references_2d: 2d nested list where some (or all) of the 1st
            dimension lists are references to the same object

    Returns:
        2d nested list with the same values, but with all references
            converted into copies.

    Raises:
    """
    # watch out for modifying nested lists created using the * operator
    # TODO(mereweth@jpl.nasa.gov) fix this for dimension > 2
    copied_2d = []
    ids = set()
    for i in range(len(references_2d)):
        if (id(references_2d[i]) in ids):
            # this is a reference, not a real object
            copied_2d.extend([copy.copy(references_2d[i])])
        else:
            copied_2d.extend([references_2d[i]])

        # id for unique object; OK to add twice
        ids.add(id(references_2d[i]))

    return copied_2d


# TODO(mereweth@jpl.nasa.gov) - refactor into general utils?
def dup_internal(to_dup, axis=0):
    """
    Duplicate all internal elements of the desired axis

    If to_dup is one dimensional, add a dimension so that the desired axis
        can be duplicated along.

    Args:
        to_dup: numpy array to duplicate
            Eg: [[0, 1, 2, 3],
                 [4, 5, 6, 7]]
        axis: axis to duplicate internal elements along

    Returns:
        A numpy array with every internal column duplicated.
            Eg: [[0, 1, 1, 2, 2, 3],
                 [4, 5, 5, 6, 6, 7]]

    Raises:
        minsnap.exceptions.InputError if to_dup has more than two dimensions
    """

    # to_dup should already be a numpy array, but make it one anyway
    to_dup = np.array(to_dup)

    # TODO(mereweth@jpl.nasa.gov) - is this the desired way to handle these two
    # cases?
    # 1d array can be saved by adding dimension
    if len(np.shape(to_dup)) == 1:
        if axis == 0:
            to_dup = np.expand_dims(to_dup, 1)
        if axis == 1:
            to_dup = np.expand_dims(to_dup, 0)
    # we choose not to handle more than 2 dimensions
    if len(np.shape(to_dup)) > 2:
        raise exceptions.InputError(to_dup, "Input has more than 2 dimensions")

    # if there are no internal elements along the desired axis:
    if np.shape(to_dup)[axis] <= 2:
        return to_dup

    dupped = np.repeat(to_dup, np.r_[1,
                                     2 * np.ones(np.shape(to_dup)[axis] - 2,
                                                 dtype=int),
                                     1],
                       axis=axis)

    # dupped = np.c_[to_dup[:, [0]],
    #               np.repeat(to_dup[:, 1:-1], 2, axis=1),
    #               to_dup[:, [-1]]]
    return dupped


def n_coeffs_free_ders(order):
    """Get the number of coefficients and free derivatives

    Args:
        order: the order of the polynomial segments

    Returns:
        A tuple with the number of coefficients and the number of free
            derivatives.
        Ex: (10, 5)

    Raises:
    """
    import numpy as np

    if order < 0:
        return (0, 0)

    if np.isnan(order):
        return (0, 0)

    # Take floor of floating point value
    order = int(order)

    # TODO(mereweth@jpl.nasa.gov) order has to be even? or is trunc and cast
    # ok?
    n_coeff = order + 1
    # TODO(mereweth@jpl.nasa.gov) document why half of derivatives must be free
    n_der = np.floor(n_coeff / 2).astype(int)
    return (n_coeff, n_der)


def ltri_mgrid(order, max_der=0):
    """Get a lower triangular meshgrid where the diagonals increase linearly

    These are the values of exponents during successive differentiations of a
        polynomial.

    Args:
        order: the order of the polynomial segment
        max_der: Optional variable (less than or equal to the number of free
            derivatives) specifying the max derivative to return. If absent,
            use n_der.

    Returns:
        A numpy array of size n_coeff x max_der with all zero entries on and
            above the main diagonal.

        Eg:
        [0, 0
         1, 0
         2, 1
         3, 2]

    Raises:
    """

    # TODO(mereweth@jpl.nasa.gov) could reuse this by returning it
    n_coeff, n_der = n_coeffs_free_ders(order)

    if (max_der <= 0) or np.isnan(max_der):
        max_der = n_der
    max_der = int(max_der)

    # : operator defaults to starting at 0
    [n, r] = np.mgrid[:n_coeff, :max_der]
    n_minus_r = n - r

    # n_minus_r < 0 means r < n, this value is automatically 0
    n_minus_r[n_minus_r < 0] = 0

    return n_minus_r


# TODO(mereweth@jpl.nasa.gov) can this be shared with other modules?
def poly_der_coeffs(order):
    """Get a matrix showing the coefficients of polynomials and derivatives

    Args:
        order: the order of the polynomial segment

    Returns:
        A tuple with the ltri_mgrid and a matrix where the column is the order
            of the derivative, the row is the position in the polynomial,
            and the value is the coefficient from successive differentiations.

        Values stop increasing after the exponent of a term has dropped to 1.

        We only calculate terms up to floor((order+1)/2), the number of free
            derivatives, because higher order derivative terms must be left
            free to allow continuity at segment boundaries.

        Eg:
        ([[0, 0, 0, 0, 0],
          [1, 0, 0, 0, 0],
          [2, 1, 0, 0, 0],
          [3, 2, 1, 0, 0],
          [4, 3, 2, 1, 0],
          [5, 4, 3, 2, 1],
          [6, 5, 4, 3, 2],
          [7, 6, 5, 4, 3],
          [8, 7, 6, 5, 4],
          [9, 8, 7, 6, 5]],
         [[   1,    1,    1,    1,    1],
          [   1,    1,    1,    1,    1],
          [   1,    2,    2,    2,    2],
          [   1,    3,    6,    6,    6],
          [   1,    4,   12,   24,   24],
          [   1,    5,   20,   60,  120],
          [   1,    6,   30,  120,  360],
          [   1,    7,   42,  210,  840],
          [   1,    8,   56,  336, 1680],
          [   1,    9,   72,  504, 3024]]))

    Raises:
    """

    # TODO(mereweth@jpl.nasa.gov) could reuse this by returning it
    n_coeff, n_der = n_coeffs_free_ders(order)

    n_minus_r = ltri_mgrid(order)

    # construct the product of (r-m) from m = 0 to r-1 as the columns of r_minus_m
    # can reuse mgrid from above
    # the c_ here means concatenate columns
    r_minus_m = np.tile(
        np.c_[np.ones((n_coeff, 1), dtype=np.int), n_minus_r[:, :-1]], (n_der, 1, 1))
    r_minus_m[r_minus_m < 1] = 1

    [j, i, k] = np.mgrid[0:n_coeff, 0:n_der, 0:n_der]
    one_ind = np.where(i < k)  # these values get changed to be 1
    r_minus_m[i[one_ind], j[one_ind], k[one_ind]] = 1

    # multiply across rows to get the product of decreasing exponents
    # 2d array with same dimensions of t_power for each segment is output
    prod = np.squeeze(np.transpose(
        np.prod(r_minus_m, axis=2, keepdims=True), axes=[2, 1, 0]), axis=0)

    return (n_minus_r, prod)


def poly_hessian_coeffs(order, der_order):
    """Get a matrix showing the outer product polynomial derivatives

    First, calculate the products of free polynomial derivative coefficients.
        Then, calculate the outer product with itself of a vector of those
        products. These are the factors in the Hessian coefficients that come
        from successive differentiations.

    The polynomial order is allowed to be greater than that necessary for a
        given order cost. Eg: 9th order polynomial allows cost terms on any
        derivative up to and including snap.

    Args:
        order: the order of the polynomial segment

    Returns:
        A tuple with the ltri_mgrid and the factors in the coefficients of the
            Hessian matrix obtained by taking the second partial derivatives of
            the cost term that incorporates the integral of squares of
            derivatives.
            (Equation 6)

        Eg:
        ([[0, 0, 0, 0],
          [1, 0, 0, 0],
          [2, 1, 0, 0],
          [3, 2, 1, 0],
          [4, 3, 2, 1],
          [5, 4, 3, 2],
          [6, 5, 4, 3],
          [7, 6, 5, 4]],
         [[0, 0, 0,    0,     0,     0,     0,      0,      0,      0],
          [0, 0, 0,    0,     0,     0,     0,      0,      0,      0],
          [0, 0, 0,    0,     0,     0,     0,      0,      0,      0],
          [0, 0, 0,   36,   144,   360,   720,   1260,   2016,   3024],
          [0, 0, 0,  144,   576,  1440,  2880,   5040,   8064,  12096],
          [0, 0, 0,  360,  1440,  3600,  7200,  12600,  20160,  30240],
          [0, 0, 0,  720,  2880,  7200, 14400,  25200,  40320,  60480],
          [0, 0, 0, 1260,  5040, 12600, 25200,  44100,  70560, 105840],
          [0, 0, 0, 2016,  8064, 20160, 40320,  70560, 112896, 169344],
          [0, 0, 0, 3024, 12096, 30240, 60480, 105840, 169344, 254016]])


    Raises:
    """

    i_minus_m = ltri_mgrid(order, der_order)

    # product of free polynomial derivative coefficients
    der_prod = np.prod(i_minus_m, axis=1, keepdims=True)

    # outer product with itself of products of free polynomial
    # derivative coefficients
    return (i_minus_m, der_prod * der_prod.transpose())

def form_waypoints_polytraj(waypoints,order):
    """ utility to form the zeros for the waypoint matrix where not defined """
    for key in waypoints.keys():
        n_der = n_coeffs_free_ders(order[key])[1]
        if waypoints[key] is not None:
            wayp_shape = np.shape(waypoints[key])

            if np.size(wayp_shape)==0:
                wayp = np.reshape(waypoints[key],(1,1))
                col = 1
                row = 1
            elif np.size(wayp_shape)==1:
                wayp = np.reshape(waypoints[key],(1,wayp_shape[0]))
                col = wayp_shape[0]
                row = 1
            else:
                row = wayp_shape[0]
                col = wayp_shape[1]
                wayp = waypoints[key]

            if row < n_der:
                waypoints[key] = np.append(wayp,
                                                [[0.0] * col] * (n_der - row), axis=0)
    return waypoints

def default_der_fixed_polytraj(num_internal,order):
    # Create fixed and inequality matrices
    n_der = n_coeffs_free_ders(order['x'])[1]
    # float derivatives at internal waypoints and fix at beginning and end
    inner = [[True] + [False] * num_internal + [True]] * (n_der - 1)
    # fix 0th derivative at internal waypoints
    der_fixed_temp = [[True] * (num_internal + 2)]
    der_fixed_temp.extend(inner)
    der_fixed_temp = np.array(der_fixed_temp)

    # Allocate for each dimension
    der_fixed = dict()
    for key in order.keys():
        n_der = n_coeffs_free_ders(order[key])[1]
        der_fixed[key] = der_fixed_temp[0:n_der,:].copy()

    return der_fixed

def create_laps_trajectory(n_laps, qr_p_lap, qr_p_entry, qr_p_exit, entry_ID, exit_ID, closed_loop=True):
    """
        Creates a trajectory with entry, exit and a number of laps of a given course
        Generic creation of polynomial to represent trajectory. If entry and exit are None, then
        the output is just a copy of the trajectory polynomial
        Works if loop is closed (will make laps) or open

        Args:
            n_laps: The number of laps to create
            qr_p_lap: a qr_polytraj object for the main trajectory
            qr_p_entry: a qr_polytaj object for the entry trajectory
            qr_p_exit: a qr_polytraj object for the exit trajectory
            entry_ID: the ID of the waypoint on the trajectory to enter
            exit_ID: the ID of the waypoint on the trajectory to exit
            close_loop: flag for whether or not the trajectory should be closed loop
    """

    if closed_loop:
        ppoly_laps = qr_p_lap.make_n_laps(n_laps,entry_ID,exit_ID)
    else:
        ppoly_laps = qr_p_lap.entry_exit_on_open_traj(entry_ID, exit_ID)
        # dict()
        # for key in qr_p_lap.quad_traj.keys():
        #     ppoly_laps[key] = qr_p_lap.quad_traj[key].piece_poly

    poly_out = dict()

    for key in qr_p_lap.quad_traj.keys():
        # Start with the entry
        if qr_p_entry is not None:
            poly_out[key] = qr_p_entry.quad_traj[key].piece_poly
        else:
            poly_out[key] = ppoly_laps[key]

        # Add laps
        if qr_p_entry is not None:
            # Compute the new time breakpoints to append
            x_1 = ppoly_laps[key].x[1:] + poly_out[key].x[-1]*np.ones(ppoly_laps[key].x.size-1)
            poly_out[key].extend(ppoly_laps[key].c,x_1)

        # Add exit
        if qr_p_exit is not None:
            # Compute the new time breakpoints to append
            x_2 = qr_p_exit.quad_traj[key].piece_poly.x[1:] + poly_out[key].x[-1]
            poly_out[key].extend(qr_p_exit.quad_traj[key].piece_poly.c,x_2)

    return poly_out

def create_TO_traj_LAND_trajectory(qr_p, qr_p_entry, qr_p_exit):
    """
        Creates a trajectory with entry, exit and a a trajectory that is not a closed loop
        Assumes entry is at the start and exit is at the end

        Args:
            laps: a qr_polytraj object for the main trajectory for
            entry: a qr_polytaj object for the entry trajectory
            exit: a qr_polytraj object for the exit trajectory
    """


    ppoly_traj = dict()

    poly_out = dict()

    for key in qr_p.quad_traj.keys():

        # Trajectory
        ppoly_traj[key] = qr_p.quad_traj[key].piece_poly

        # Start with the entry
        poly_out[key] = qr_p_entry.quad_traj[key].piece_poly

        # Add trajectory
        # Compute the new time breakpoints to append
        x_1 = ppoly_traj[key].x[1:] + poly_out[key].x[-1]*np.ones(ppoly_traj[key].x.size-1)
        poly_out[key].extend(ppoly_traj[key].c,x_1)

        # Add exit
        # Compute the new time breakpoints to append
        x_2 = qr_p_exit.quad_traj[key].piece_poly.x[1:] + poly_out[key].x[-1]
        poly_out[key].extend(qr_p_exit.quad_traj[key].piece_poly.c,x_2)

    return poly_out

def form_entry_or_exit_waypoints(entry_or_exit, qr_p_lap, add_on_waypoints, entry_ID, add_on_der_fixed=dict(x=None,y=None,z=None,yaw=None)):

    # if type(add_on_waypoints['x']).__module__ != np.__name__ or np.size(add_on_waypoints['x']) == 1:
    #     for key in add_on_waypoints.keys():
    #         add_on_waypoints[key] = np.array([add_on_waypoints[key]] + [0.0]*(lap_waypoints[key].shape[0]-1))

    # Get transition times for polytraj of the laps
    trans_times = seg_times_to_trans_times(qr_p_lap.times)

    # time to match to
    t = trans_times[entry_ID]

    # Get state
    lap_waypoint = dict()
    out_waypoints = dict()
    out_der_fixed = dict()

    for key in add_on_waypoints.keys():
        n_der = qr_p_lap.waypoints[key].shape[0]
        lap_waypoint[key] = np.reshape(np.zeros(n_der),(n_der,1))

        temp_poly = qr_p_lap.quad_traj[key].piece_poly

        # Get each derivative at the waypoint
        for i in range(n_der):
            lap_waypoint[key][i] = temp_poly(t)
            temp_poly = temp_poly.derivative()

        # Assign der_fixed if not defined
        if add_on_der_fixed[key] is None:
            add_on_der_fixed[key] = np.zeros((n_der,add_on_waypoints[key].shape[1]),dtype=bool)
            add_on_der_fixed[key][0,:] = True # Fix all positions
            if entry_or_exit is "entry":
                add_on_der_fixed[key][:,0] = True # Fix all starting derivatives.
            elif entry_or_exit is "exit":
                add_on_der_fixed[key][:,-1] = True # Fix all ending derivatives.

        if entry_or_exit is "entry":
            # Concatenate output waypoints and der_fixed
            out_waypoints[key] = np.concatenate([add_on_waypoints[key],lap_waypoint[key]],axis=1)
            # Force waypoint on lap to match all derivatives
            out_der_fixed[key] = np.concatenate((add_on_der_fixed[key],np.ones((n_der,1),dtype=bool)),axis=1)
        elif entry_or_exit is "exit":
            # Concatenate output waypoints and der_fixed
            out_waypoints[key] = np.concatenate([lap_waypoint[key],add_on_waypoints[key]],axis=1)
            # Force waypoint on lap to match all derivatives
            out_der_fixed[key] = np.concatenate((np.ones((n_der,1),dtype=bool),add_on_der_fixed[key]),axis=1)

    return out_waypoints, out_der_fixed

def get_state_at_waypoint(qr_p, ID):
    # Get transition times for polytraj of the laps
    trans_times = seg_times_to_trans_times(qr_p.times)

    # time to match to
    t = trans_times[ID]

    # Get state
    state_wp = dict()
    for key in qr_p.waypoints.keys():
        n_der = qr_p.waypoints[key].shape[0]
        state_wp[key] = np.zeros(n_der)

        temp_poly = qr_p.quad_traj[key].piece_poly

        # Get each derivative at the waypoint
        for i in range(n_der):
            state_wp[key][i] = temp_poly(t)
            temp_poly = temp_poly.derivative()

    return state_wp

def get_full_state_trajectory_ppoly(qr_p,ppoly,n_steps):
    """ Get the complete trajectory from a piecewise polynomial input """

    # times
    trans_times = ppoly['x'].x
    t = np.linspace(trans_times[0], trans_times[-1], n_steps)
    x = ppoly['x'](t)
    y = ppoly['y'](t)
    z = ppoly['z'](t)

    yaw = ppoly['yaw'](t)
    yaw_dot = ppoly['yaw'].derivative()(t)
    yaw_ddot = ppoly['yaw'].derivative().derivative()(t)

    acc = np.array([ppoly['x'].derivative().derivative()(t),
                    ppoly['y'].derivative().derivative()(t),
                    ppoly['z'].derivative().derivative()(t)])

    jerk = np.array([ppoly['x'].derivative().derivative().derivative()(t),
                     ppoly['y'].derivative().derivative().derivative()(t),
                     ppoly['z'].derivative().derivative().derivative()(t)])

    snap = np.array([ppoly['x'].derivative().derivative().derivative().derivative()(t),
                     ppoly['y'].derivative().derivative().derivative().derivative()(t),
                     ppoly['z'].derivative().derivative().derivative().derivative()(t)])

    # out = dict()
    #
    # out['t'] = t
    # out['x'] = x
    # out['y'] = y
    # out['z'] = z
    # out['yaw'] = yaw
    # out['yaw_dot'] = yaw_dot
    # out['yaw_ddot'] = yaw_ddot
    # out['acc'] = acc
    # out['jerk'] = jerk
    # out['snap'] = snap

    qr_p.t = t
    qr_p.x = x
    qr_p.y = y
    qr_p.z = z
    qr_p.yaw = yaw
    qr_p.yaw_dot = yaw_dot
    qr_p.yaw_ddot = yaw_ddot
    qr_p.acc = acc
    qr_p.jerk = jerk
    qr_p.snap = snap

    # return qr_p

def check_continuity(ppoly, n_der, eps=1e-6, rtol=1e-5, atol=1e-5):
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


    # ppoly has no data before and after first transition points, so we
    # can't check those
    trans_times = ppoly['x'].x[1:-1]

    for key in ppoly.keys():
        temp_ppoly = ppoly[key]
        for i in range(n_der):
            if not np.allclose(temp_ppoly(trans_times - eps),temp_ppoly(trans_times + eps), rtol=rtol, atol=atol):
                error = np.linalg.norm(temp_ppoly(trans_times - eps)-temp_ppoly(trans_times + eps))
                print("Failed continuity check with error {} on derivative {} of dimension {}\n".format(error, i, key))
                # return False
            temp_ppoly = temp_ppoly.derivative()

    return True

def fill_n_laps_poly_traj_msg(ppoly_laps, msg):
    msg.n_segs = np.size(ppoly_laps['x'].x) - 1

    msg.n_x_coeffs = np.shape(ppoly_laps['x'].c)[0] # number of coefficients per segment for x
    msg.n_y_coeffs = np.shape(ppoly_laps['y'].c)[0] # number of coefficients per segment for y
    msg.n_z_coeffs = np.shape(ppoly_laps['z'].c)[0] # number of coefficients per segment for z


    # trans_times = seg_times_to_trans_times(ppoly_laps.times)
    # t_total = trans_times[-1] - trans_times[0]
    # t = np.linspace(trans_times[0], trans_times[-1], t_total*1000)
    #
    # yaw = np.arctan2(ppoly_laps['y'].derivative()(t),ppoly_laps['x'].derivative()(t))
    #
    # spl = sp.interpolate.splrep(t,yaw)
    # ppoly_laps['yaw'] = np.interpolate.PPoly.from_spline(spl)


    msg.n_yaw_coeffs = np.shape(ppoly_laps['yaw'].c)[0] # number of coefficients per segment for yaw

    # polynomial coefficients, segment by segment, starting at highest order coefficient
    msg.x_coeffs[0:np.size(ppoly_laps['x'].c)] = np.asarray(ppoly_laps['x'].c.T).flatten().tolist()
    msg.y_coeffs[0:np.size(ppoly_laps['y'].c)] = np.asarray(ppoly_laps['y'].c.T).flatten().tolist()
    msg.z_coeffs[0:np.size(ppoly_laps['z'].c)] = np.asarray(ppoly_laps['z'].c.T).flatten().tolist()
    msg.yaw_coeffs[0:np.size(ppoly_laps['yaw'].c)] = np.asarray(ppoly_laps['yaw'].c.T).flatten().tolist()
    # transition times, segment by segment
    msg.t_trans[0:np.size(ppoly_laps['x'].x) - 1] = ppoly_laps['x'].x[1:]

    print(msg)

def quaternion_power(q,t):

    tf = transforms3d

    vec, theta = tf.quaternions.quat2axangle(q)

    q_out = tf.quaternions.axangle2quat(vec,theta*t)

    q_out /= tf.quaternions.qnorm(q_out)

    return q_out

def quaternion_SLERP(q1,q2,t):
    # t from 0 to 1
    tf = transforms3d

    q_a = tf.quaternions.qmult(tf.quaternions.qconjugate(q1),q2)
    q_a = quaternion_power(q_a,t)

    q_out = tf.quaternions.qmult(q1,q_a)

    return q_out


def main():
    order = 7
    der_order = 3
    print("Polynomial Hessian coefficients for poly order {}; derivative order {}\n".format(
        order, der_order))
    print(poly_hessian_coeffs(order, der_order))

    tf = transforms3d

    q1 = tf.quaternions.qeye()
    import pdb; pdb.set_trace()
    q2 = tf.quaternions.axangle2quat(np.array([1.0,0,0]),np.pi/4)
    t = 0.1
    q3 = quaternion_SLERP(q1,q2,t)


if __name__ == '__main__':
    main()
