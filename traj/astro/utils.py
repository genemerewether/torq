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

import yaml
import string
import os
import rdp

import numpy as np
import copy
import scipy.interpolate
sp = scipy

from minsnap import exceptions
from minsnap.settings import STATE_INFO_TOKEN, FIELD_ORDER_TOKEN
# from astro.settings import STATE_INFO_TOKEN, FIELD_ORDER_TOKEN

try:
    from px4_msgs.msg import PolyTraj as PolyTraj_msg
    HAS_POLYTRAJ_MSG = True
except ImportError:
    HAS_POLYTRAJ_MSG = False

import transforms3d
from transforms3d import euler

class InputWrapper(object):
    def __init__(self, func, x,defer=True):
        self.x = x
        self.func = func
        self.defer=defer

    def wrap_func(self, new_x):
        # determine numerically which x values have changed
        # indices = np.where(-np.isclose(self.x, new_x,atol=1e-10,rtol=1e-9))[0]
        # print("x is: {}".format(self.x))
        # print('x_new is {}\nindices are: {}'.format(new_x,indices))
        #
        # if len(indices)<1:
        #     return self.func([None], None,defer=self.defer)
        # elif len(indices) == len(self.x):
        #     # If all are changed then reset stored x
        #     self.x = new_x.copy()
        #     return self.func(indices, new_x,defer=self.defer)
        # else:
        return self.func(np.arange(0,len(new_x),1), new_x,defer=self.defer)

def load_waypoints(filename):
    with open(filename, 'r') as filp:
        data = yaml.load(filp)

    # TODO(mereweth@jpl.nasa.gov) - get yaw from quaternion
    # waypoints.yaw is only used if face_front is False
    der_fixed = dict()
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

def form_waypoints_polytraj(waypoints,n_der):
    """ utility to form the zeros for the waypoint matrix where not defined """

    for key in waypoints.keys():

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
            if row < n_der[key]:
                waypoints[key] = np.append(wayp,
                                                [[0.0] * col] * (n_der[key] - row), axis=0)
    return waypoints

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

    if not hasattr(qr_p_lap,'state'):
        qr_p_lap.get_trajectory()

    for key in add_on_waypoints.keys():
        n_der = qr_p_lap.n_der[key]
        lap_waypoint[key] = np.reshape(np.zeros(n_der),(n_der,1))

        # Get each derivative at the waypoint
        if entry_ID == qr_p_lap.state[key].shape[2]:
            # Last waypoint
            lap_waypoint[key][:] = qr_p_lap.state[key][:,-1,-1].reshape((n_der,1))
        else:
            # Other waypoints
            lap_waypoint[key][:] = qr_p_lap.state[key][:,0,entry_ID].reshape((n_der,1))
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

    # Get state
    state_wp = dict()

    if not hasattr(qr_p,'state'):
        qr_p.get_trajectory()

    for key in qr_p.waypoints.keys():

        state_wp[key] = np.zeros(qr_p.n_der[key])
        # Get each derivative at the waypoint

        # Get each derivative at the waypoint
        if ID == qr_p.state[key].shape[2]:
            # Last waypoint
            state_wp[key][:] = qr_p.state[key][:,-1,-1]
        else:
            # Other waypoints
            state_wp[key][:] = qr_p.state[key][:,0,ID]

    return state_wp

def n_der_from_costs(costs):
    n_der = dict()
    # Loop for each dimension
    for key in costs.keys():
        n_der_tmp = np.where(costs[key]==1)
        n_der[key] = int(n_der_tmp[0]+1)
    return n_der

def default_der_fixed_polytraj(num_internal,n_der):
    # Create fixed and inequality matrices
    # Allocate for each dimension
    der_fixed = dict()
    for key in n_der.keys():
        # float derivatives at internal waypoints and fix at beginning and end
        inner = [[True] + [False] * num_internal + [True]] * (n_der[key] - 1)
        # fix 0th derivative at internal waypoints
        der_fixed_temp = [[True] * (num_internal + 2)]
        der_fixed_temp.extend(inner)
        der_fixed_temp = np.array(der_fixed_temp)
        der_fixed[key] = der_fixed_temp[0:n_der[key],:].copy()

    return der_fixed

def update_n_coeffs(order,n_seg):
    n_coeff = 0
    # Loop for each dimension
    for key in order.keys():
        n_coeff += order[key] + 1
    n_coeff *= n_seg
    return n_coeff

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

def load_obstacles(filename):
    with open(filename, 'r') as filp:
        data = yaml.load(filp)

    # TODO(mereweth@jpl.nasa.gov) - get yaw from quaternion
    # waypoints.yaw is only used if face_front is False
    position = dict()
    orientation = dict()
    scale = dict()
    for i in range(len(data['states'][0]['x'])):
        field_tokens = data[STATE_INFO_TOKEN]['x'][FIELD_ORDER_TOKEN]
        if field_tokens[i] in ['x', 'y', 'z']:
            position[field_tokens[i]] = [state['x'][i]
                                          for state in data['states']]
        elif field_tokens[i] in ['q_w', 'q_x', 'q_y', 'q_z']:
            orientation[field_tokens[i]] = [state['x'][i]
                                          for state in data['states']]
        elif field_tokens[i] in ['scale_x', 'scale_y', 'scale_z']:
            scale[field_tokens[i]] = [state['x'][i]
                                          for state in data['states']]
    weight = [state['w'] for state in data['states']]

    n_obstacles = np.size(weight)

    params = []

    for i in range(n_obstacles):
        params.append(dict())
        params[i]['constraint_type'] = "ellipsoid"
        params[i]['weight'] = weight[i]
        params[i]['keep_out'] = True
        params[i]['der'] = 0
        params[i]['x0'] = np.array([position['x'][i],position['y'][i],position['z'][i]])
        A = np.matrix(np.identity(3))
        A[0,0] = 1/scale['scale_x'][i]**2
        A[1,1] = 1/scale['scale_y'][i]**2
        A[2,2] = 1/scale['scale_z'][i]**2
        rot_mat = transforms3d.quaternions.quat2mat(np.array([orientation['q_w'][i],orientation['q_x'][i],orientation['q_y'][i],orientation['q_z'][i]]))
        params[i]['rot_mat'] = rot_mat
        params[i]['A'] = np.array(np.matrix(rot_mat).T*A*np.matrix(rot_mat))


    return params


def create_complete_trajectory(qr_p_traj, qr_p_entry, qr_p_exit,entry_ID,exit_ID):
    """
        Combines entry, trajectory and exit polynomials

        Args:
            qr_p_traj: a qr_polytraj object for the main trajectory
            qr_p_entry: a qr_polytaj object for the entry trajectory
            qr_p_exit: a qr_polytraj object for the exit trajectory
            entry_ID: the ID of the waypoint on the trajectory to enter
            exit_ID: the ID of the waypoint on the trajectory to exit
            close_loop: flag for whether or not the trajectory should be closed loop
    """

    ppoly_laps = qr_p_traj.create_callable_ppoly()

    if entry_ID > exit_ID:
        print("Error with entry and exit IDs")
        return
    else:
        for key in ppoly_laps.keys():
            new_coeffs = np.delete(ppoly_laps[key].c,np.arange(entry_ID),axis=1)
            times = trans_times_to_seg_times(ppoly_laps[key].x)
            new_times = np.delete(times,np.arange(entry_ID))
            new_coeffs = np.delete(new_coeffs,np.arange(new_coeffs.shape[1]-(ppoly_laps[key].c.shape[1]-exit_ID),new_coeffs.shape[1]),axis=1)
            new_times =  np.delete(new_times,np.arange(new_times.shape[0]-(times.shape[0]-exit_ID),new_times.shape[0]))
            trans_times = seg_times_to_trans_times(new_times)
            ppoly_laps[key] = sp.interpolate.PPoly(new_coeffs,trans_times,extrapolate=False)

    poly_out = dict()

    if qr_p_entry is not None:
        ppoly_entry = qr_p_entry.create_callable_ppoly()
    if qr_p_exit is not None:
        ppoly_exit = qr_p_exit.create_callable_ppoly()

    for key in ppoly_laps.keys():
        # Start with the entry
        if qr_p_entry is not None:
            poly_out[key] = ppoly_entry[key]
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
            x_2 = ppoly_exit[key].x[1:] + poly_out[key].x[-1]
            poly_out[key].extend(ppoly_exit[key].c,x_2)

    return poly_out

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
