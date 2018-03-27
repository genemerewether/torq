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
__email__ = "bmorrell@jpl.nasa.gov"

import argparse
import os
import time

import numpy
from numpy import linalg
np = numpy
import scipy
sp = scipy

from diffeo import settings
from diffeo import exceptions

import transforms3d
from transforms3d import euler
from transforms3d import quaternions


def check_furthest_axis(x_c,y_c,z_b):
    """
    checks whether the z_b axis is closer to x_c or y_x, returns flag to indicate use of the vector furthest from z_b

    Args:
        x_c: dimension 3 array of the x vector in the iterim yaw frame
        y_c: dimension 3 array of the y vector in the iterim yaw frame
        z_b: dimension 3 array of the z body axis

    Returns:
        flag: 'x' or 'y' indicating which vector to use to extract orientation

    Raises
    """

    # Cross products
    theta_x = np.arccos(np.inner(x_c,z_b))
    theta_y = np.arccos(np.inner(y_c,z_b))

    # Want angle closest to 90 degrees
    val_x = np.abs(theta_x - np.pi/2)
    val_y = np.abs(theta_y - np.pi/2)

    # choose vector with smallest value
    if val_x <= val_y:
        return "x"
    else:
        return "y"

def check_closest_axis(x_b_1,x_b_2,x_b_current):
    """
    checks whether the current x_b axis is closer to x_b from x_c or y_c, returns which option to use

    Args:
        x_b_1: dimension 3 array of the x body vector from using x_c
        x_b_2: dimension 3 array of the x_body vector from using y_c
        x_b_current: dimension 3 array of the current or previous x_b

    Returns:
        flag: 'x' or 'y' indicating which vector to use to extract orientation

    Raises
    """

    # Cross products
    theta_1 = np.arccos(np.inner(x_b_1,x_b_current))
    theta_2 = np.arccos(np.inner(x_b_2,x_b_current))

    # Check using Tan theta:
    np.arctan2(np.linalg.norm(np.cross(x_b_current,x_b_1)),np.inner(x_b_1,x_b_current))

    if np.isnan(theta_1):
        theta_1 = np.pi
        print("Warning: nan angle between trial body vector 1 and current body vector")
    if np.isnan(theta_2):
        theta_2 = np.pi
        print("Warning: nan angle between trial body vector 2 and current body vector")

    # choose vector with smallest value
    if theta_1 <= theta_2:
        return 1
    else:
        return 2

def check_closest_axes_set(x_b_1,y_b_1,x_b_2=None,y_b_2=None,x_b_current=None,y_b_current=None):#,use_body_set=False):
    """
    checks which set of axes to use as which are closest to the current x and y body axes
    Also checkes the negative of the two axes sets

    Args:
        x_b_1: dimension 3 array of the x body vector from using x_c
        y_b_1: dimension 3 array of the y body vector from using x_c
        x_b_2: dimension 3 array of the x_body vector from using y_c
        y_b_2: dimension 3 array of the y_body vector from using y_c
        x_b_current: dimension 3 array of the current or previous x_b
        y_b_current: dimension 3 array of the current or previous y_b

    Returns:
        flag: '+x_c' or '-x_c' or '+y_c' or '-y_c' indicating which vector to use to extract orientation

    Raises
    """
    #
    measure = np.zeros(4)
    # measure_check = np.zeros(4)

    start_1 = time.time()
    # Check angles for each axes set
    # x_b_1 positive
    measure[0] = get_angle_between(x_b_1,x_b_current)**2 + get_angle_between(y_b_1,y_b_current)**2
    # x_b_1 negative
    measure[1] = get_angle_between(-x_b_1,x_b_current)**2 + get_angle_between(-y_b_1,y_b_current)**2

    if x_b_2 is None or y_b_2 is None:
        measure[2] = 50.0
        measure[3] = 50.0
    else:
        # x_b_2 positive
        measure[2] = get_angle_between(x_b_2,x_b_current)**2 + get_angle_between(y_b_2,y_b_current)**2
        # x_b_2 negative
        measure[3] = get_angle_between(-x_b_2,x_b_current)**2 + get_angle_between(-y_b_2,y_b_current)**2
        method1_t = time.time() - start_1

    # start_2 = time.time()
    # measure_check[0] = get_angle_between(x_b_1,x_b_current)
    # measure_check[1] = get_angle_between(-x_b_1,x_b_current)
    # measure_check[2] = get_angle_between(x_b_2,x_b_current)
    # measure_check[3] = get_angle_between(-x_b_1,x_b_current)
    # method2_t = time.time() - start_2

    # time_comp = method2_t/method1_t
    # print("time difference is {}".format(time_comp))

    # Get minimum measure
    index = np.where(measure==np.min(measure))[0][0]
    # index_check = np.where(measure_check==np.min(measure_check))[0][0]
    # if index != index_check:
    #     print("Using only x gives different result for finding closest angle")
    #     print("x_b_1: {}, y_b_1: {}\nx_b_2: {}, y_b_2: {}".format(x_b_1,y_b_1,x_b_2,y_b_2))

    # if np.linalg.norm(x_b_1-x_b_2)>0.05 and np.linalg.norm(x_b_1-x_b_current)>0.0005:
    #     print("NOTE: x_b_1 and x_b_2 are different by {}\n{}\n{}".format(np.linalg.norm(x_b_1-x_b_2),x_b_1,x_b_2))

    # List of flag outputs
    flag_list = ['+x_c','-x_c','+y_c','-y_c']

    # Return flag with minimum measure
    return flag_list[index], measure

def get_angle_between(vec1,vec2):

    dot_prod = np.inner(vec1,vec2)

    tol = 1e-5

    if (1.0-dot_prod) < tol:
        theta = 0.0
    elif (1.0+dot_prod) < tol:
        theta = np.pi
    else:
        theta = np.arccos(dot_prod)

    # Check using Tan theta:
    # TODO pick the most robust approach - but the cos approach is quicker
    # a = np.inner(vec1,vec2)
    # if a < 0:
    #     import pdb; pdb.set_trace()
    # theta = np.arctan2(np.linalg.norm(np.cross(vec1,vec2)),np.inner(vec1,vec2))

    # if not np.allclose(theta,theta_c):
    #     print("WARNING: computation between dot product ({}) and tan approach ({}) is different".format(theta_c,theta))
    #     # print("time diff in get angle is {}".format(dt2/dt1))

    if theta < 0.0:
        print("WARNING: angle computed {} is negative".format(theta))

    return theta

def get_quat_pitch_roll(z_b):
    """ Computes the quaternion describing the rotation required in purely pitch
    and roll to achieve the desired quaternion
    """

    # Body axis in the body frame (equivalent to Z global axis)
    z_b_b = np.array([0,0,1])

    # Angle between:
    alpha = get_angle_between(z_b_b,z_b)

    eps = 1e-10
    if alpha <= eps:
        # Set to identity
        quat = np.array([1,0,0,0.0])
    else:
        # Use cross product to compute the quaternion
        q_vec = np.cross(z_b_b,z_b)
        q_vec = q_vec/np.linalg.norm(q_vec)*np.sin(alpha/2)
        # form the quaternion
        quat = np.append(np.cos(alpha/2),q_vec)

    # Transform the quaternion into a rotation matrix
    R = transforms3d.quaternions.quat2mat(quat)

    return R[:,0], R[:,1]


def get_thrust(accel):
    """
    Calculates the thrust acceleration from an input acceleration and gravity

    Args:
        accel: length 3 np array for the acceleration (x, y, z)

    Returns:
        thrust: the thrust acceleration vector (length 3 np array: x, y, z)
        thrust_mag: the net thrust acceleration (float)

    Raises
    """
    # hardcoded gravity
    gravity = settings.GRAVITY

    # Thrust (adding gravity)
    thrust = accel + np.array([0,0,gravity])
    thrust_mag = np.linalg.norm(thrust)

    return thrust, thrust_mag

def get_z_body(accel):
    """
    Calculates the z body vector from a given acceleration.

    Method is with the differentially flat derivation from:
    'Minimum Snap Trajectory Generation and Control for Quadrotors'
    Mellinger and Kumar 2011, equation 6

    Args:
        accel: length 3 np array for the acceleration (x, y, z)

    Returns:
        z_b: the z body axis (length 3 np array: x, y, z)

    Raises
    """
    # Get thrust
    thrust, thrust_mag = get_thrust(accel)

    #TODO(bmorrell@jpl.nasa.gov) implement a check for when thrust_mag == 0

    # Body z vector (same as thrust direction)
    z_b = thrust/thrust_mag

    return z_b

def get_x_y_body(yaw, z_b, x_b_current=None, y_b_current=None, deriv_type='check_x_b_current'):
    """
    Calculates the x and y body vectors from a given z body axis and yaw angle
    The x_b_current is to correct the derivation to be nearest to the current oreintation

    Method is with the differentially flat derivation from:
    'Minimum Snap Trajectory Generation and Control for Quadrotors'
    Mellinger and Kumar 2011, page 2521

    Args:
        yaw: yaw value (in radians)
        z_b: the z body axis (length 3 np array: x, y, z)

    Returns:
        x_b: the x body axis (length 3 np array: x, y, z)
        y_b: the y body axis (length 3 np array: x, y, z)

    Raises
    """
    data = dict()
    # intermediate yaw frame x and ys
    if (deriv_type == 'use_x_b_current' or deriv_type == 'combined_x_b_current') and x_b_current is not None:
        # Using the current body axis for x_c
        x_c = x_b_current
        # x_b_current = np.array([np.cos(yaw),np.sin(yaw),0])
    else:
        x_c = np.array([np.cos(yaw),np.sin(yaw),0])

    if x_b_current is None:
        x_b_current = x_c.copy()

    if deriv_type == 'combined_x_b_current' and y_b_current is not None:
        # Using the current body axis for y_c
        y_c = y_b_current
        # y_b_current = np.array([-1*np.sin(yaw),np.cos(yaw),0])
    else:
        y_c = np.array([-1*np.sin(yaw),np.cos(yaw),0])

    if y_b_current is None:
        y_b_current = y_c.copy()


    # Use old method if no x_b_current is set
    if deriv_type == "yaw_only" or deriv_type == 'use_x_b_current':
        # Simplest method, using only x_c
        y_b = np.cross(z_b,x_c)
        y_b /= np.linalg.norm(y_b)
        x_b = np.cross(y_b,z_b)
    elif deriv_type == 'x_or_y_furthest':
        # use x_c or y_c - whichever is furthest (closest to 90 deg)
        # Check which axis to uses
        axis_flag = check_furthest_axis(x_c,y_c,z_b)

        # Compute the body axes
        if axis_flag == 'x':
            y_b = np.cross(z_b,x_c)
            y_b /= np.linalg.norm(y_b)
            x_b = np.cross(y_b,z_b)
        elif axis_flag == 'y':
            x_b = np.cross(y_c,z_b)
            x_b /= np.linalg.norm(x_b)
            y_b = np.cross(z_b,x_b)
    elif deriv_type == 'check_x_b_current' or deriv_type[:8] == "combined":
        # Work with both and then select the closes to the current orientation
        # if z_b[2]<0:
        #     x_c = -x_c
        #     y_c = -y_c

        # initialise
        x_b_1 = np.zeros(z_b.shape)
        x_b_2 = np.zeros(z_b.shape)
        y_b_1 = np.zeros(z_b.shape)
        y_b_2 = np.zeros(z_b.shape)

        # Compute the body axes using x_c
        y_b_1 = np.cross(z_b,x_c)
        y_b_1 /= np.linalg.norm(y_b_1)
        x_b_1 = np.cross(y_b_1,z_b)

        # Compute the body axes using y_c
        x_b_2 = np.cross(y_c,z_b)
        x_b_2 /= np.linalg.norm(x_b_2)
        y_b_2 = np.cross(z_b,x_b_2)

        if deriv_type == 'check_x_b_current':
            # Check for the closest axis
            axis = check_closest_axis(x_b_1,x_b_2,x_b_current)

            if axis == 1:
                x_b = x_b_1
                y_b = y_b_1
            elif axis == 2:
                x_b = x_b_2
                y_b = y_b_2

        elif deriv_type[:8] == "combined":
            axes_flag, measure = check_closest_axes_set(x_b_1,y_b_1,x_b_2,y_b_2,x_b_current,y_b_current)
            data['xc_yc'] = measure
            if deriv_type == "combined" :#and np.min(measure) > (80*np.pi/180):
                # Pass if deriv_type = "combined_multi_ax"
                print("using current body")
                # If change more than 10 degrees
                # Use x_b_current
                # Compute the body axes using x_c
                y_b_12 = np.cross(z_b,x_b_current)
                y_b_12 /= np.linalg.norm(y_b_12)
                x_b_12 = np.cross(y_b_12,z_b)

                # Compute the body axes using y_c
                x_b_22 = np.cross(y_b_current,z_b)
                x_b_22 /= np.linalg.norm(x_b_22)
                y_b_22 = np.cross(z_b,x_b_22)

                # if np.linalg.norm(x_b_1-x_b_2) > 0.05 and np.linalg.norm(x_b_1+x_b_2) > 0.05:
                #     import pdb; pdb.set_trace()

                axes_flag2, measure = check_closest_axes_set(x_b_12,y_b_12,x_b_22,y_b_22,x_b_current,y_b_current)#,use_body_set=True)
                data['xb_yb'] = measure
                # if not np.allclose(measure[0],measure[2],atol=1e-4):
                #     print("x_b and y_b result is different")
                #     import pdb; pdb.set_trace()
            else:
                 print("using x_c, y_c")
                 data['xb_yb'] = np.zeros(4)
            print("Combined axes selection is: "+axes_flag)
            if axes_flag == "+x_c":
                x_b = x_b_1
                y_b = y_b_1
            elif axes_flag == "-x_c":
                x_b = -x_b_1
                y_b = -y_b_1
            elif axes_flag == "+y_c":
                x_b = x_b_2
                y_b = y_b_2
            elif axes_flag == "-y_c":
                x_b = -x_b_2
                y_b = -y_b_2
            else:
                print("ERROR: check closest axes set failed")


    elif deriv_type == 'check_negative_set':
        # Compute body axes
        y_b = np.cross(z_b,x_c)
        y_b /= np.linalg.norm(y_b)
        x_b = np.cross(y_b,z_b)

        # Check for the closest axis
        axis = check_closest_axis(x_b,-x_b,x_b_current)

        if axis == 2:
            x_b = -x_b
            y_b = -y_b

    elif deriv_type[:11] == 'neg_or_quat':

        # Faessler's Method
        # Compute the body axes using y_c
        SIGMA = 0.000001

        if np.fabs(z_b[2]) > SIGMA:
            x_b = np.cross(y_c,z_b)

            if z_b[2]<0:
                x_b *= -1

        else:
            # Thrust in xy plan, point x down
            x_b = np.array([0,0,-1.0])

        eps = 1e-10

        if np.linalg.norm(x_b) <= eps:
            # singularity
            x_b, y_b = get_quat_pitch_roll(z_b)
        else:
            x_b /= np.linalg.norm(x_b)
            y_b = np.cross(z_b,x_b)

    elif deriv_type == "quat_pq_only":

        x_b, y_b = get_quat_pitch_roll(z_b)

    else:
        raise exceptions.InputError(deriv_type,"Invalid deriv type or missing information")


    return x_b, y_b, data

def get_x_y_body_second_angle(yaw, z_b,x_b_current=None,y_b_current=None,deriv_type="second_angle"):
    """
    Calculates the x and y body vectors from a given z body axis and yaw angle

    Method is with the differentially flat derivation from:
    'Aggressive Flight With Quadrotors for Perching on Inclined Surfaces'
    Thomas, Pope, Lopianno, ... Kumar 2016. UPenn GRASP Lab

    Args:
        yaw: yaw value (in radians)
        z_b: the z body axis (length 3 np array: x, y, z)

    Returns:
        x_b: the x body axis (length 3 np array: x, y, z)
        y_b: the y body axis (length 3 np array: x, y, z)

    Raises
    """

    gamma_lim = 30.*np.pi/180

    # intermediate yaw frame x and y
    x_c = np.array([np.cos(yaw),np.sin(yaw),0])
    y_c = np.array([-1*np.sin(yaw),np.cos(yaw),0])

    if x_b_current is None:
        x_b_current = x_c.copy()

    if y_b_current is None:
        y_b_current = y_c.copy()

    # initialise
    x_b = np.zeros(z_b.shape)
    y_b = np.zeros(z_b.shape)

    if deriv_type[-1] == "2":
        # Use quaternion method
        # Get projection of z_b onto x_c, z plane
        n_c = np.cross(x_c,np.array([0,0,1]))
        z_c = z_b - np.dot(z_b,y_c)*y_c

        # Rotate from z_c by 30 degrees
        ang = 30*np.pi/180
        quat = np.append(np.cos(ang/2),y_c*np.sin(ang/2))
        x_c = transforms3d.quaternions.rotate_vector(z_c,quat)

        # Compute body axes
        y_b = np.cross(z_b,x_c)
        y_b /= np.linalg.norm(y_b)
        x_b = np.cross(y_b,z_b)

        axes_flag, measure = check_closest_axes_set(x_b,y_b,x_b_2=None,y_b_2=None,x_b_current=x_b_current,y_b_current=y_b_current)

        if axes_flag == '-x_c':
            x_b = -x_b
            y_b = -y_b

    else:

        # Compute the angle from x_c to z_b
        theta_x = np.arccos(np.inner(x_c,z_b))
        sign = 1

        # normalise to be between 0 and pi/2
        if theta_x > np.pi/2:
            theta_x = np.pi-theta_x




        if np.abs(theta_x) < gamma_lim:# or np.isnan(theta_x):
            # Activate the gamma out of plane angle
            gamma = sign*gamma_lim*2

            # Recalculate x_c, with gamma rotation about y
            x_c = np.array([np.cos(yaw)*np.cos(gamma),np.sin(yaw),-np.cos(yaw)*np.sin(gamma)])


            # Compute body axes
            y_b = np.cross(z_b,x_c)
            y_b /= np.linalg.norm(y_b)
            x_b = np.cross(y_b,z_b)

            if np.arccos(np.inner(y_b,y_c)) > np.pi/2:
                y_b = -y_b
                x_b = np.cross(y_b,z_b)

        else:
            # Compute body axes
            y_b = np.cross(z_b,x_c)
            y_b /= np.linalg.norm(y_b)
            x_b = np.cross(y_b,z_b)

    return x_b, y_b

def get_x_y_body_analyse(yaw, z_b, x_b_current=None, y_b_current=None, deriv_type='check_x_b_current'):
    """
    Calculates the x and y body vectors from a given z body axis and yaw angle
    The x_b_current is to correct the derivation to be nearest to the current oreintation

    Method is with the differentially flat derivation from:
    'Minimum Snap Trajectory Generation and Control for Quadrotors'
    Mellinger and Kumar 2011, page 2521

    Args:
        yaw: yaw value (in radians)
        z_b: the z body axis (length 3 np array: x, y, z)

    Returns:
        x_b: the x body axis (length 3 np array: x, y, z)
        y_b: the y body axis (length 3 np array: x, y, z)

    Raises
    """
    data = dict()

    # intermediate yaw frame x and y
    x_c = np.array([np.cos(yaw),np.sin(yaw),0])
    y_c = np.array([-1*np.sin(yaw),np.cos(yaw),0])

    if x_b_current is None:
        x_b_current = x_c.copy()

    if y_b_current is None:
        y_b_current = y_c.copy()

    # COMPUTE RANGE OF AXES
    # Using  x_c
    y_b_1 = np.cross(z_b,x_c)
    y_b_1 /= np.linalg.norm(y_b_1)
    x_b_1 = np.cross(y_b_1,z_b)

    # Using y_c
    x_b_2 = np.cross(y_c,z_b)
    x_b_2 /= np.linalg.norm(x_b_2)
    y_b_2 = np.cross(z_b,x_b_2)

    # Measure for these angles
    axes_flag, measure = check_closest_axes_set(x_b_1,y_b_1,x_b_2,y_b_2,x_b_current,y_b_current)
    # Second angle approach
    # Use quaternion method
    # Get projection of z_b onto x_c, z plane
    z_c_1 = z_b - np.dot(z_b,y_c)*y_c
    z_c_1 = z_c_1/np.linalg.norm(z_c_1)

    # Get projection of z_b onto y_c, z plane
    z_c_2 = z_b - np.dot(z_b,x_c)*x_c
    z_c_2 = z_c_2/np.linalg.norm(z_c_2)

    # Rotate from z_c by 30 degrees
    ang = 30*np.pi/180
    quat_1 = np.append(np.cos(ang/2),y_c*np.sin(ang/2))
    quat_2 = np.append(np.cos(ang/2),x_c*np.sin(ang/2))
    x_c_sa = transforms3d.quaternions.rotate_vector(z_c_1,quat_1)
    y_c_sa = transforms3d.quaternions.rotate_vector(z_c_2,quat_2)

    # Compute body axes with x_c second angle
    y_b_3 = np.cross(z_b,x_c_sa)
    y_b_3 /= np.linalg.norm(y_b_3)
    x_b_3 = np.cross(y_b_3,z_b)

    # compute body axes with y_c second angle
    x_b_4 = np.cross(y_c_sa,z_b)
    x_b_4 /= np.linalg.norm(x_b_4)
    y_b_4 = np.cross(z_b,x_b_4)

    # Measure for these angles
    axes_flag_sa, measure_sa = check_closest_axes_set(x_b_3,y_b_3,x_b_4,y_b_4,x_b_current,y_b_current)

    data['axes_errors'] = np.concatenate([measure,measure_sa])

    if np.min(measure) <= np.min(measure_sa):
        axes_choose = axes_flag
    else:
        axes_choose = axes_flag_sa
        x_b_1 = x_b_3
        y_b_1 = y_b_3
        x_b_2 = x_b_4
        y_b_2 = y_b_4

    if axes_flag == "+x_c":
        x_b = x_b_1
        y_b = y_b_1
    elif axes_flag == "-x_c":
        x_b = -x_b_1
        y_b = -y_b_1
    elif axes_flag == "+y_c":
        x_b = x_b_2
        y_b = y_b_2
    elif axes_flag == "-y_c":
        x_b = -x_b_2
        y_b = -y_b_2
    else:
        print("ERROR: check closest axes set failed")

    return x_b, y_b, data

def body_frame_from_yaw_and_accel(yaw, accel,out_format='all',deriv_type='yaw_only',x_b_current=None, y_b_current=None):
    """
    Return the rotation of the body frame from desired yaw and accel (single input!)

    Method is with the differentially flat derivation from:
    'Minimum Snap Trajectory Generation and Control for Quadrotors'
    Mellinger and Kumar 2011, page 2521

    Args:
        yaw: a single yaw value
        accel: numpy array of accelerations (3 in length - x, y, z)
        out_format: format of the outupt desred.
            Options are 'euler', 'quaternion', 'matrix' and 'all' (euler, quaternion, matrix)
        deriv_type: setting to select different methods. Can be 'yaw_only' (from Mellinger), or 'second_angle' (from Thomas)

    Returns:
        # TODO(mereweth@jpl.nasa.gov) - what data type & library for SO3 and SE3?
        eul: length 3 np array of the euler angles along the trajectory
        quat: length 4 np array of the quaternions along the trajectory
        R: 3x3 np array of the rotation matrices along the trajectory

    Raises:
    """

    # Make the input arrays
    yaw = np.array(yaw) # Ensure yaw has at least 1 dimenions

    accel = np.array(accel)

    data = None

    # Get the Z body axis
    z_b = get_z_body(accel)

    # Get the other body axes
    if deriv_type[:12] == 'second_angle':
        x_b, y_b = get_x_y_body_second_angle(yaw,z_b,x_b_current=x_b_current,y_b_current=y_b_current,deriv_type=deriv_type)
    elif deriv_type == 'analyse':
        x_b, y_b, data = get_x_y_body_analyse(yaw,z_b,x_b_current=x_b_current,y_b_current=y_b_current)
    else:
        x_b, y_b, data = get_x_y_body(yaw,z_b,x_b_current=x_b_current,y_b_current=y_b_current,deriv_type=deriv_type)

    # Form the rotation matrix
    R = np.array(np.vstack((x_b,y_b,z_b)).T)

    # return statements - change for different input options
    if out_format == 'euler':
        # Extract the Euler angles
        eul = np.zeros(z_b.shape)
        eul[2],eul[1],eul[0] = transforms3d.euler.mat2euler(R,'rzyx')
        return eul, data
    elif out_format == 'quaternion':
        # Convert to quaternions
        quat = np.array(transforms3d.quaternions.mat2quat(R))
        return quat, data
    elif out_format == 'matrix':
        return R, data
    elif out_format == 'all':
        # Extract the Euler angles
        eul = np.zeros(z_b.shape)
        eul[2],eul[1],eul[0] = transforms3d.euler.mat2euler(R,'rzyx')
        # Convert to quaternions
        quat = np.array(transforms3d.quaternions.mat2quat(R))
        return eul, quat, R, data

def body_frame_from_q3_and_accel(q3, accel, out_format):
    """
    Return the rotation of the body frame from desired q3 and accel (single input!)

    Args:
        q3: a single q3 value (with quaternion convention w, x, y, z)
        accel: numpy array of accelerations (3 in length - x, y, z)
        out_format: format of the outupt desred.
            Options are 'euler', 'quaternion', 'matrix' and 'all' (euler, quaternion, matrix)

    Returns:
        eul: length 3 np array of the euler angles along the trajectory
        quat: length 4 np array of the quaternions along the trajectory
        R: 3x3 np array of the rotation matrices along the trajectory

    Raises:
    """
    gravity = settings.GRAVITY

    # Make the input arrays
    q3 = np.array(q3) # Ensure yaw has at least 1 dimenions

    accel = np.array(accel)

    # Get thrust acceleration magnitude
    thrust, thrust_mag = get_thrust(accel)

    # q0
    # q02 = np.sqrt((accel[0]**2+accel[1]**2)/(2*thrust_mag**2*(1-(accel[2]+gravity)/thrust_mag))-q3**2)
    q0 = np.sqrt(0.5*(1+(accel[2]+gravity)/thrust_mag)-q3**2)
    # if q0!=q02:
    #     import pdb; pdb.set_trace()
    # q0 = np.ones(q3.shape)*np.cos(np.pi/8)

    #q1
    q1 = (1/(2*thrust_mag))*(1/(q0**2+q3**2))*(accel[0]*q3-accel[1]*q0)

    #q2
    q2 = (1/(2*thrust_mag))*(1/(q0**2+q3**2))*(accel[0]*q0+accel[1]*q3)

    # Form together
    quat = np.array([q0,q1,q2,q3]).T

    # normalise
    quat = quat/np.linalg.norm(quat)

    # return statements - change for different input options
    if out_format == 'euler':
        # Conver to Euler angles
        eul = np.zeros([3,])
        eul[2],eul[1],eul[0] = transforms3d.euler.quat2euler(quat,'rzyx')
        return eul
    elif out_format == 'quaternion':
        return quat
    elif out_format == 'matrix':
        # Convert to rotation matrix
        R = np.array(transforms3d.quaternions.quat2mat(quat))
        return R
    elif out_format == 'all':
        # Extract the Euler angles
        eul = np.zeros([3,])
        eul[2],eul[1],eul[0] = transforms3d.euler.quat2euler(quat,'rzyx')
        # Convert to rotation matrix
        R = np.array(transforms3d.quaternions.quat2mat(quat))
        return eul, quat, R

def accel_yaw_from_attitude(attitude,thr_mag):
    """
    Takes in an attitude and accelerration magnitude and outputs the
    corresponding acceleration vector and yaw

    Method is with the differentially flat derivation from:
    'Minimum Snap Trajectory Generation and Control for Quadrotors'
    Mellinger and Kumar 2011, page 2521

    Args:
        attitude: representaion of the attitude (Euler, matrix of quaternions). numpy arrays
        thr_mag: magnitude of thrust

    Returns:
        accel_vec: length 3 np array of the acceleration vector
        yaw: a float of the yaw

    Raises:
        InputError: invalid attitude input
    """
    #TODO(bmorrell@jpl.nasa.gov) test and check whether there are cases where this fails
    # extract acceleration direction from attitude
    if attitude.size == 3:
        #E uler angles
        # Convert to matrix
        R = transforms3d.euler.euler2mat(attitude[2],attitude[1],attitude[0],'rzyx')
        # Last column of rotation matrix is the thrust
        thr_dir = R[:,2]
        # Get yaw
        yaw = attitude[2]
    elif attitude.size == 4:
        # quaternions
        thr_dir = transforms3d.quaternions.rotate_vector([0.,0.,1.],attitude)
        # Get yaw
        yaw, pitch, roll  = transforms3d.euler.quat2euler(attitude,'rzyx')
    elif attitude.size == 9:
        # Rotation matrix
        thr_dir = attitude[:,2]
        # Get yaw
        yaw, pitch, roll  = transforms3d.euler.mat2euler(attitude,'rzyx')
    else:
        raise exceptions.InputError(attitude,
                "invalid attitude input. Needs to be dimenions 3, 4 or (3,3)")

    # Compute acceleration
    accel_vec = thr_mag*thr_dir - np.array([0,0,settings.GRAVITY])

    return accel_vec, yaw

def define_attitude_waypoint(waypoint,der_fixed,attitude,thr_mag):
    """
    Takes in an attitude and accelerration magnitude and creates a waypoint for
    each flat input to force the attitude and thrust magnitude

    Args:
        attitude: representaion of the attitude (Euler, matrix of quaternions). numpy arrays
        thr_mag: magnitude of thrust
        waypoint: the existing waypoint, a dict with fields 'x', 'y', 'z' and 'yaw'
                    each with an array of

    Returns:
        waypoint: a dict with fields 'x', 'y', 'z' and 'yaw' for the new waypoint
        der_fixed:

    Raises:
    """

    # TODO(bmorrell@jpl.nasa.gov) include input checks

    # Extract the acceleration and yaw
    accel_vec, yaw = accel_yaw_from_attitude(attitude,thr_mag)

    # Set acceleration constraints
    waypoint['x'][2] = accel_vec[0]
    waypoint['y'][2] = accel_vec[1]
    waypoint['z'][2] = accel_vec[2]

    # Set yaw constraint
    waypoint['yaw'][0] = yaw

    # Update der_fixed
    der_fixed['x'][2] = True
    der_fixed['y'][2] = True
    der_fixed['z'][2] = True
    der_fixed['yaw'][0] = True

    return waypoint, der_fixed

def assign_attitude_to_waypoints(waypoints,der_fixed,attitude,thr_mag):
    # Modify waypoints for acceleration
    for index in range(0,waypoints['x'].shape[1]):
        wayp_mod = dict()
        der_fixed_mod = dict()
        for key in waypoints.keys():
            wayp_mod[key] = waypoints[key][:,index]
            der_fixed_mod[key] = der_fixed[key][:,index]
        # Set waypoint for the given attitude and thrust magnitude
        waypoint_new, der_fixed_new = define_attitude_waypoint(wayp_mod,der_fixed_mod,attitude[:,index],thr_mag[index])
        for key in waypoints.keys():
            waypoints[key][:,index] = waypoint_new[key].copy()
            der_fixed[key][:,index] = der_fixed_new[key].copy()

    return waypoints, der_fixed

def main():
    from minsnap import utils
    # Load a trajectory
    import pickle
    f = open('share/sample_output/traj.pickle','rb')
    qr_p = pickle.load(f)
    f.close

    # Compute times
    trans_times = utils.seg_times_to_trans_times(qr_p.times)
    t1 = np.linspace(trans_times[0], trans_times[-1], 100)

    # To test single input:
    t = t1[15]#:4]

    # Yaw
    yaw = 0.#qr_p.quad_traj['yaw'].piece_poly(t)
    print(yaw)

    # accel
    accel = np.array([qr_p.quad_traj['x'].piece_poly.derivative().derivative()(t),qr_p.quad_traj['y'].piece_poly.derivative().derivative()(t),
                            qr_p.quad_traj['z'].piece_poly.derivative().derivative()(t)])
    print(accel.shape)

    eul, quat, R, data = body_frame_from_yaw_and_accel(yaw, accel,'all')

    eul2, quat2, R2 , data = body_frame_from_q3_and_accel(yaw, accel, 'all')






    #
    # print("\n EULER DERIVATION \n")
    # print("Rotation Matrix is\n {}".format(R))
    # print("Euler angles are: \n{}".format(eul))
    # print("Quaternion is: \n{}".format(quat))
    #
    # print("\n QUATERNION DERIVATION \n")
    # print("Rotation Matrix is\n {}".format(R2))
    # print("Euler angles are: \n{}".format(eul2))
    # print("Quaternion is: \n{}".format(quat2))
    #
    #
    # #_--------------------------------------------#
    # # Test singularity cases
    # accel = np.array([0.,0.,1.0])#np.random.randn(3)
    # yaw = np.array(0.0)
    # eul, quat, R, data = body_frame_from_yaw_and_accel(yaw, accel,'all')
    #
    # # for i in range(20):
    # #     accel = np.random.randn(3,1)
    # eul2, quat2, R2, data = body_frame_from_q3_and_accel(yaw, accel, 'all')
    #
    # print("\n EULER DERIVATION \n")
    # print("Rotation Matrix is\n {}".format(R))
    # print("Euler angles are: \n{}".format(eul))
    # print("Quaternion is: \n{}".format(quat))
    #
    # print("\n QUATERNION DERIVATION \n")
    # print("Rotation Matrix is\n {}".format(R2))
    # print("Euler angles are: \n{}".format(eul2))
    # print("Quaternion is: \n{}".format(quat2))


    method_list = ['yaw_only','check_negative_set','second_angle','x_or_y_furthest','check_x_b_current','quaternions']

    x_b_current = np.array([0.0995,0,-0.995])
    x_b_current = np.array([-0.0995,0,0.995])

    print('Zero Thrust \n')
    accel = np.array([0,0,-settings.GRAVITY])
    yaw = np.array(np.pi)
    q3 = np.sin(yaw/2)
    for item in method_list:
        if item == 'quaternions':
            R, data = body_frame_from_q3_and_accel(q3, accel, 'matrix')
        else:
            R, data = body_frame_from_yaw_and_accel(yaw, accel,'matrix',deriv_type=item,x_b_current=x_b_current)
        print("Method: {}\n{}\n".format(item,R))

    print('Testing singularities \n')
    print('Yaw singularities \n')
    accel = np.array([1.0,0,-settings.GRAVITY])
    # yaw = np.array(0.0)
    q3 = np.sin(yaw/2)
    for item in method_list:
        if item == 'quaternions':
            R, data = body_frame_from_q3_and_accel(q3, accel, 'matrix')
        else:
            R, data = body_frame_from_yaw_and_accel(yaw, accel,'matrix',deriv_type=item,x_b_current=x_b_current)
        print("Method: {}\n{}\n".format(item,R))

    print('Before Yaw singularities \n')
    accel = np.array([1.0,0,0.1-settings.GRAVITY])
    # yaw = np.array(0.0)
    q3 = np.sin(yaw/2)
    for item in method_list:
        if item == 'quaternions':
            R, data = body_frame_from_q3_and_accel(q3, accel, 'matrix')
        else:
            R, data = body_frame_from_yaw_and_accel(yaw, accel,'matrix',deriv_type=item,x_b_current=x_b_current)
        print("Method: {}\n{}\n".format(item,R))

    print('After Yaw singularities \n')
    accel = np.array([1.0,0,-0.1-settings.GRAVITY])
    # yaw = np.array(0.0)
    q3 = np.sin(yaw/2)
    for item in method_list:
        if item == 'quaternions':
            R, data = body_frame_from_q3_and_accel(q3, accel, 'matrix')
        else:
            R, data = body_frame_from_yaw_and_accel(yaw, accel,'matrix',deriv_type=item,x_b_current=x_b_current)
        print("Method: {}\n{}\n".format(item,R))

    print('Quaternion singularity \n')
    accel = np.array([0,0,-1.0-settings.GRAVITY])
    # yaw = np.array(0.0)
    q3 = np.sin(yaw/2)
    for item in method_list:
        if item == 'quaternions':
            R, data = body_frame_from_q3_and_accel(q3, accel, 'matrix')
        else:
            R = body_frame_from_yaw_and_accel(yaw, accel,'matrix',deriv_type=item,x_b_current=x_b_current)
        print("Method: {}\n{}\n".format(item,R))


    # print('\naccel and yaw from attitude and thrust mag\n')
    # # attitude = np.array([np.sqrt(0.5),0,0,np.sqrt(0.5)])
    # attitude = np.array([np.pi/4,np.pi/3,3*np.pi/4])
    # # yaw, roll, pitch = np.array(transforms3d.euler.quat2euler(attitude,'rzyx'))
    # # attitude = np.array([roll, pitch,yaw])
    # # attitude = transforms3d.quaternions.quat2mat(attitude)
    # #import pdb; pdb.set_trace()
    # thr_mag = 5.0
    # accel_vec, yaw = accel_yaw_from_attitude(attitude,thr_mag)
    # print("Accel Vec: {}, yaw {}".format(accel_vec,yaw))


if __name__ == '__main__':
    main()
