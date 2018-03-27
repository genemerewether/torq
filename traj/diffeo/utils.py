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

import argparse
import os

import numpy
np = numpy
import scipy
sp = scipy

#TODO(mereweth@jpl.nasa.gov) - decide what to do about ROS1 vs ROS2 vs no URI
# lookup; this also affects stl import if we want meshes later for aerodynamics

# urdf_parser_py is for parsing URDF description of quadrotor
# from urdf_parser_py.urdf import URDF
#
#
# def composite_mass_and_cg_from_urdf(urdf_object, dof_values):
#     """
#     Return the total mass and center of gravity given a urdfdom_py URDF object
#
#     Args:
#         urdf_object: urdfdom_py URDF object representing the quadrotor
#         dof_values: values of any degrees of freedom present
#
#     Returns:
#         A tuple with:
#         mass: floating point value for total mass
#         cg: vector from URDF root to center of gravity
#
#     Raises:
#         TypeError if conversion from this URDF structure is not implemented yet
#         AttributeError if conversion references nonexistent field
#     """
#
#     invalid = False
#     # TODO(mereweth@jpl.nasa.gov) - handle multiple, rigidly attached links
#     if len(urdf_object.links) != 1:
#         invalid = True
#
#     if invalid:
#         raise TypeError(urdf_object,
#                         "Extracting mass and center of gravity from"
#                         + " this URDF is not supported yet")
#
#     mass = urdf_object.links[0].inertial.mass
#     if urdf_object.links[0].inertial.origin is None:
#         cg = np.array([0, 0, 0])
#     else:
#         cg = urdf_object.links[0].inertial.origin.xyz
#     return (mass, cg)
#
#
# def composite_inertia_from_urdf(urdf_object, dof_values):
#     """
#     Return the inertia matrix given a urdfdom_py URDF object
#
#     Args:
#         urdf_object: urdfdom_py URDF object representing the quadrotor
#         dof_values: values of any degrees of freedom present
#
#     Returns:
#         inertia: floating point inertia matrix
#
#     Raises:
#         TypeError if conversion from this URDF structure is not implemented yet
#         AttributeError if conversion references nonexistent field
#     """
#
#     invalid = False
#     # TODO(mereweth@jpl.nasa.gov) - handle multiple, rigidly attached links
#     if len(urdf_object.links) != 1:
#         invalid = True
#
#     # TODO(mereweth@jpl.nasa.gov) - handle cg not at origin
#     if urdf_object.links[0].origin is not None:
#         invalid = True
#     if urdf_object.links[0].inertial.origin is not None:
#         invalid = True
#
#     if invalid:
#         raise TypeError(urdf_object,
#                         "Extracting mass and center of gravity from"
#                         + " this URDF is not supported yet")
#
#     inertia = np.mat(urdf_object.links[0].inertial.inertia.to_matrix())
#     return inertia


def first_order_mixer_from_urdf(urdf_object):
    """
    Return a matrix mapping from rotor velocities squared to force and moment

    Args:
        urdf_object: urdfdom_py URDF object representing the quadrotor

    Returns:
        A numpy matrix mapping from a vector of squared rotor velocities to
            exerted force and moment in the body frame

    Raises:
        TypeError if conversion from this URDF structure is not implemented yet
        AttributeError if conversion references nonexistent field
    """


def load_urdf(filename):
    """
    Load a representation of the quadrotor from a URDF

    Args:
        filename: already-resolved relative or absolute path to URDF file

    Returns:
        urdfdom_py URDF object representing the quadrotor

    Raises:
    """

    return URDF.from_xml_file(filename)


def quaternion_rates(ang_vel,quat):
    """
    Compute quaternion rates from input angular velocity and quaternion

    Args:
        ang_vel: numpy array of angular velocities (3 in length - p, q, r)
        quat: the quaternions of the orientation (w, x, y, z)

    Returns:
        q_dot: vector of quaternion rates (in format w, x, y, z)

    Raises:
    """

    omega_mat = np.array([[0,-ang_vel[0], -ang_vel[1], -ang_vel[2]],
                         [ang_vel[0], 0, ang_vel[2], -ang_vel[1]],
                         [ang_vel[1], -ang_vel[2], 0, ang_vel[0]],
                         [ang_vel[2], ang_vel[1], -ang_vel[0], 0]])

    q_dot = 0.5*np.matmul(omega_mat,quat)

    return q_dot

def quaternion_accel(ang_accel,quat,quat_dot):
    """
    Compute quaternion rates from input angular velocity and quaternion

    Args:
        ang_accel: numpy array of angular accelerations (3 in length - p_dot, q_dot, r_dot)
        quat: the quaternions of the orientation (w, x, y, z)

    Returns:
        quat_ddot: vector of quaternion rates (in format w, x, y, z)

    Raises:
    """

    omega_mat = np.array([[0,-ang_accel[0], -ang_accel[1], -ang_accel[2]],
                         [ang_accel[0], 0, ang_accel[2], -ang_accel[1]],
                         [ang_accel[1], -ang_accel[2], 0, ang_accel[0]],
                         [ang_accel[2], ang_accel[1], -ang_accel[0], 0]])

    quat_ddot = 0.5*np.matmul(omega_mat,quat) - np.inner(quat_dot,quat_dot)*quat

    return quat_ddot


def main():
    # parser = argparse.ArgumentParser(
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument(
    #     '-f', '--file', default="~/snappy/traj/share/sample_data/torq250.urdf",
    #     help='filename to load URDF from')
    #
    # args = parser.parse_args()
    # args.file = os.path.expanduser(args.file)
    # quadrotor_urdf = load_urdf(args.file)
    # print(quadrotor_urdf)
    import transforms3d
    from transforms3d import euler
    from transforms3d import quaternions

    ang_vel = np.array([0.4,0.0,0.0])
    quat = transforms3d.euler.euler2quat(0.0,0.0,0.0,'rzyx')

    quat_dot = quaternion_rates(ang_vel,quat)

    ang_accel = np.array([0.1,0.0,0.0])
    quat_ddot = quaternion_accel(ang_accel,quat,quat_dot)

    print("My method for quat_dot gives: \n{}".format(quat_dot))

    print("For quat_ddot gives: \n{}".format(quat_ddot))

if __name__ == '__main__':
    main()
