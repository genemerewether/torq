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


import numpy
np = numpy

from diffeo import utils

def get_p_and_q(thrust_mag,jerk,R):
    """
    Extracts the p and q body rates from the thrust, jerk and body axes

    Method is with the differentially flat derivation from:
    'Minimum Snap Trajectory Generation and Control for Quadrotors'
    Mellinger and Kumar 2011, page 2521

    Args:
        thrust_mag: a single value for the thrust acceleration magnitude
        jerk: numpy array of jerk (3 in length - x, y, z)
        R: Rotation matrix (captures x, y and z body axes). Rotation matrix from body to global

    Returns:
        p, q: the angular velocities about the x and y axes

    Raises:
    """
    # Take out body axes
    x_b = R[:,0]
    y_b = R[:,1]
    z_b = R[:,2]
    z_w = [0,0,1] # global z vector

    # Compute h_w vector - used for next steps
    h_w = 1/thrust_mag*(jerk-np.dot(z_b,jerk)*z_b)

    # Compute body rates
    p = -np.dot(h_w,y_b)
    q = np.dot(h_w,x_b)

    return p, q

def get_angular_vel(thrust_mag,jerk,R,yaw_dot):
    """
    Extracts angular rates from the thrust, jerk, yaw rate, and body axes

    Method is with the differentially flat derivation from:
    'Minimum Snap Trajectory Generation and Control for Quadrotors'
    Mellinger and Kumar 2011, page 2521

    Args:
        thrust_mag: a single value for the thrust acceleration magnitude
        jerk: numpy array of jerk (3 in length - x, y, z)
        yaw_dot: first derivative of yaw (single value)
        R: Rotation matrix (captures x, y and z body axes). Rotation matrix from body to global

    Returns:
        ang_vel: the angular velocities. numpy array of length 3 (p, q, r)

    Raises:
    """

    # Compute body rates
    p, q = get_p_and_q(thrust_mag,jerk,R)

    # Get yaw rate
    z_b = R[:,2]
    z_w = [0,0,1] # global z vector
    r = yaw_dot*np.dot(z_w,z_b)

    ang_vel = np.array([p,q,r])

    return ang_vel

def get_angular_vel_quat(thrust_mag,jerk,R,q3_dot,quat):
    """
    Extracts angular rates from the thrust, jerk, q3 rate, body axes and quaternions

    Args:
        thrust_mag: a single value for the thrust acceleration magnitude
        jerk: numpy array of jerk (3 in length - x, y, z)
        q3_dot: first derivative of q3 (single value)
        R: Rotation matrix (captures x, y and z body axes). Rotation matrix from body to global
        quat: the quaternions of the orientation (w, x, y, z)

    Returns:
        ang_vel: the angular velocities. numpy array of length 3 (p, q, r)

    Raises:
    """

    # Compute body rates
    p, q = get_p_and_q(thrust_mag,jerk,R)

    # Get yaw rate
    r = 1/quat[0]*(2*q3_dot + p*quat[2]-q*quat[1])

    ang_vel = np.array([p,q,r])

    return ang_vel

def get_pdot_and_qdot(thrust_mag,jerk,snap,R,ang_vel):
    """
    Extracts angular accelerations (p_dot and q_dot) from the thrust, jerk, snap, angular rates and body axes

    Method is with the differentially flat derivation from:
    'Minimum Snap Trajectory Generation and Control for Quadrotors'
    Mellinger and Kumar 2011, page 2521

    Args:
        thrust_mag: a single value for the thrust acceleration magnitude
        jerk: numpy array of jerk (3 in length - x, y, z)
        snap: numpy array of snap (3 in length - x, y, z)
        R: Rotation matrix (captures x, y and z body axes). Rotation matrix from body to global
        ang_vel: numpy array of angular velocities (3 in length - p, q, r)

    Returns:
        p_dot, q_dot: the angular accelerations about x and y

    Raises:
    """
    # Take out body axes
    x_b = R[:,0]
    y_b = R[:,1]
    z_b = R[:,2]
    z_w = [0,0,1] # global z vector

    # Compute h_alpha matrix
    h_a = 1/thrust_mag*(snap - np.dot(z_b,snap)*z_b
              + thrust_mag*(np.dot(z_b,np.cross(ang_vel,np.cross(ang_vel,z_b))))*z_b
              - thrust_mag*np.cross(ang_vel,np.cross(ang_vel,z_b))
              - 2*(np.cross(ang_vel,(np.dot(z_b,jerk))*z_b)))

    # Get angular accelerations
    p_dot = -1.0*np.dot(h_a,y_b)
    q_dot = np.dot(h_a,x_b)

    return p_dot, q_dot


def get_angular_accel(thrust_mag,jerk,snap,R,ang_vel,yaw_ddot):
    """
    Extracts angular accelerations from the thrust, jerk, snap, angular rates, body axes and yaw acceleration

    Method is with the differentially flat derivation from:
    'Minimum Snap Trajectory Generation and Control for Quadrotors'
    Mellinger and Kumar 2011, page 2521

    Args:
        thrust_mag: a single value for the thrust acceleration magnitude
        jerk: numpy array of jerk (3 in length - x, y, z)
        snap: numpy array of snap (3 in length - x, y, z)
        R: Rotation matrix (captures x, y and z body axes). Rotation matrix from body to global
        ang_vel: numpy array of angular velocities (3 in length - p, q, r)
        yaw_ddot: second derivative of yaw (single value)


    Returns:
        ang_accel: the angular accelerations. numpy array of length 3 (p, q, r)

    Raises:
    """

    # get x and y angular accelerations
    p_dot, q_dot = get_pdot_and_qdot(thrust_mag,jerk,snap,R,ang_vel)

    # Get yaw acceleration
    z_b = R[:,2]
    z_w = [0,0,1] # global z vector
    r_dot = yaw_ddot*np.dot(z_w,z_b)

    ang_accel = np.array([p_dot,q_dot,r_dot])

    return ang_accel

def get_angular_accel_quat(thrust_mag,jerk,snap,R,ang_vel,q3_ddot,quat):
    """
    Extracts angular accelerations from the thrust, jerk, snap, angular rates, body axes q3 acceleration


    Args:
        thrust_mag: a single value for the thrust acceleration magnitude
        jerk: numpy array of jerk (3 in length - x, y, z)
        snap: numpy array of snap (3 in length - x, y, z)
        R: Rotation matrix (captures x, y and z body axes). Rotation matrix from body to global
        ang_vel: numpy array of angular velocities (3 in length - p, q, r)
        q3_ddot: second derivative of q3 (single value)
        quat: the quaternions of the orientation (w, x, y, z)

    Returns:
        ang_accel: the angular accelerations. numpy array of length 3 (p, q, r)

    Raises:
    """

    # get x and y angular accelerations
    p_dot, q_dot = get_pdot_and_qdot(thrust_mag,jerk,snap,R,ang_vel)

    # Get quaternion rates
    quat_dot = utils.quaternion_rates(ang_vel,quat)

    # Combine for equation
    q_dot_prod = np.inner(quat_dot,quat_dot)

    # Get yaw acceleration
    r_dot = 1.0/quat[0]*(2*q3_ddot - quat[1]*q_dot + quat[2]*p_dot + 2*q_dot_prod*quat[3])

    # Combine together
    ang_accel = np.array([p_dot,q_dot,r_dot])

    return ang_accel


def main():
    from diffeo import body_frame
    from minsnap import utils
    # Load a trajectory
    import pickle
    f = open('share/sample_output/traj.pickle','rb')
    qr_p = pickle.load(f)
    f.close

    mass = 1.0

    # Compute times
    trans_times = utils.seg_times_to_trans_times(qr_p.times)
    t1 = np.linspace(trans_times[0], trans_times[-1], 100)

    # To test single input:
    t = t1[4]#:4]

    # Yaw
    yaw = qr_p.quad_traj['yaw'].piece_poly(t)
    yaw_dot = qr_p.quad_traj['yaw'].piece_poly.derivative()(t)
    yaw_ddot = qr_p.quad_traj['yaw'].piece_poly.derivative().derivative()(t)

    # accel
    accel = np.array([qr_p.quad_traj['x'].piece_poly.derivative().derivative()(t),
                      qr_p.quad_traj['y'].piece_poly.derivative().derivative()(t),
                      qr_p.quad_traj['z'].piece_poly.derivative().derivative()(t)])

    # jerk
    jerk = np.array([qr_p.quad_traj['x'].piece_poly.derivative().derivative().derivative()(t),
                     qr_p.quad_traj['y'].piece_poly.derivative().derivative().derivative()(t),
                     qr_p.quad_traj['z'].piece_poly.derivative().derivative().derivative()(t)])


    # snap
    snap = np.array([qr_p.quad_traj['x'].piece_poly.derivative().derivative().derivative().derivative()(t),
                     qr_p.quad_traj['y'].piece_poly.derivative().derivative().derivative().derivative()(t),
                     qr_p.quad_traj['z'].piece_poly.derivative().derivative().derivative().derivative()(t)])


    # Get rotation matrix
    euler, quat, R = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'all')

    # Thrust
    thrust, thrust_mag = body_frame.get_thrust(accel)

    print("\n EULER DERIVATION")
    # Angular rates
    ang_vel = get_angular_vel(thrust_mag,jerk,R,yaw_dot)
    print("angular rates are: \n{}".format(ang_vel))

    # Angular accelerations
    ang_accel = get_angular_accel(thrust_mag,jerk,snap,R,ang_vel,yaw_ddot)
    print("angular accels are: \n{}".format(ang_accel))


    print("\n QUATERNION DERIVATION")
    # Angular rates
    q3_dot = yaw_dot
    q3_ddot = yaw_ddot
    ang_vel = get_angular_vel_quat(thrust_mag,jerk,R,q3_dot,quat)
    print("angular rates are: \n{}".format(ang_vel))

    # Angular accelerations
    ang_accel = get_angular_accel_quat(thrust_mag,jerk,snap,R,ang_vel,q3_ddot,quat)
    print("angular accels are: \n{}".format(ang_accel))



if __name__ == '__main__':
    main()
