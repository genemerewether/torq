# -*- coding: utf-8 -*-

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

from pyquaternion import Quaternion
import numpy as np
import math

def quadDynamics(x, Theta, rpm_now):

    pos = x[0:3]
    vel = x[3:6]
    omega = x[10:13]
    quat = x[6:10]
    quat = Quaternion(quat)
    quat = quat.normalised
    try:
        quat_inv = quat.inverse
    except ZeroDivisionError:
        quat_inv = Quaternion([1,0,0,0])
        quat = Quaternion([1,0, 0, 0])
        print('Warning: Quaternion could not be inverted')

    theta = Theta['value']
    m   = theta['m']
    Ixx = theta['Ixx']
    Iyy = theta['Iyy']
    Izz = theta['Izz']
    Cq = theta['Cq']
    Cdh = theta['Cdh']
    Cdxy = theta['Cdxy']
    Cdy = theta['Cdy']
    Cdz = theta['Cdz']
    deltaDx = theta['deltaDx']
    deltaDy = theta['deltaDy']
    Dx = theta['Dx']
    Dy = theta['Dy']

    # Get the thrust produced by each propeller in this state
    thrusts, flagErr = getThrust(x, Theta, rpm_now)
#    thrusts = thrusts + np.random.normal(0, 0.4, 4).reshape(np.shape(thrusts))

    # The derivative of position is velocity
    pos_dot = vel

    # Get the rate of change of the quaterion
    quat_dot = quat.derivative(omega)
    quat_dot = np.array(quat_dot.elements)

    # Thrust in the inertial frameor
    thrust_BF = np.array([0, 0, -sum(thrusts)])
    thrust_IF = quat.rotate(thrust_BF)

    # Parasitic drag in the inertial frame
    vel_BF = quat_inv.rotate(vel)
    airframeDrag_BF = -1*np.array([Cdxy, Cdxy, Cdz])*vel_BF*abs(vel_BF);
    airframeDrag_IF = quat.rotate(airframeDrag_BF)

    # Blade drag in the inertial frame
    bladeDrag_BF = -1*abs(sum(thrusts))*np.array([Cdh*vel_BF[0], Cdh*vel_BF[1], 0])
    bladeDrag_IF = quat.rotate(bladeDrag_BF)

    # Gravity in inertial frame
    weight_IF = np.array([0, 0, 9.81*m])

    # Acceleration in inertial frame
    vel_dot = (thrust_IF+weight_IF+airframeDrag_IF+bladeDrag_IF)/m

    # Torque
    torque =   np.array([[-thrusts[0]*(Dy-deltaDy)+thrusts[1]*(Dy+deltaDy)+thrusts[2]*(Dy+deltaDy)-thrusts[3]*(Dy-deltaDy)],
                         [thrusts[0]*(Dx-deltaDx)-thrusts[1]*(Dx+deltaDx)+thrusts[2]*(Dx-deltaDx)-thrusts[3]*(Dx+deltaDx)],
                          [Cq*rpm_now[0]**2+Cq*rpm_now[1]**2-Cq*rpm_now[2]**2-Cq*rpm_now[3]**2]])
    torque = np.squeeze(torque)

    # Calc the angular accel
    inertia_mat = np.diag([Ixx, Iyy, Izz])
    omega_dot = np.dot(np.linalg.inv(inertia_mat), (torque.reshape(3,1) - np.cross(omega, np.dot(inertia_mat, omega), axis=0)))
    omega_dot = np.squeeze(omega_dot)

    # Put in column vector
    x_dot = np.vstack((pos_dot.reshape(3, 1),vel_dot.reshape(3, 1), quat_dot.reshape(4, 1), omega_dot.reshape(3, 1)))
    return x_dot, flagErr

def getThrust(x, Theta, rpm_now):
    theta_vals = Theta['value']
    k1 = theta_vals['k1']
    k2 = theta_vals['k2']
    k4 = theta_vals['k4']
    Dx = theta_vals['Dx']
    Dy = theta_vals['Dy']

    vel_IF = x[3:6];
    omega = x[10:13];
    quat = x[6:10];
    quat = Quaternion(quat)
    quat = quat.normalised
    quat_inv = quat.inverse

    # Rotate the velocity into the body frame
    vel_BF = quat_inv.rotate(vel_IF)

    # Calculate the body frame velocity for each prop
    vel1 = np.squeeze(vel_BF - np.cross(omega.reshape(1,3), np.array([Dx, Dy, 0])))
    vel2 = np.squeeze(vel_BF - np.cross(omega.reshape(1,3), np.array([-Dx, -Dy, 0])))
    vel3 = np.squeeze(vel_BF - np.cross(omega.reshape(1,3), np.array([Dx, -Dy, 0])))
    vel4 = np.squeeze(vel_BF - np.cross(omega.reshape(1,3), np.array([-Dx, Dy, 0])))

    # Negate the z component as for the thrust model z is positive upward
    # x and y sign don't matter as they are squared
    vel1[2] = -vel1[2];
    vel2[2] = -vel2[2];
    vel3[2] = -vel3[2];
    vel4[2] = -vel4[2];

    # Reshape the velocities into column vectors
    vel1 = vel1.reshape(3, 1)
    vel2 = vel2.reshape(3, 1)
    vel3 = vel3.reshape(3, 1)
    vel4 = vel4.reshape(3, 1)

    # Concatenate
    vels = np.concatenate((vel1, vel2, vel3, vel4), axis = 1)
    flagErr = False
    thrust = np.zeros([4, 1])
    # Loop through each prop
    for propNo in range(0, 4):

        # Extract the velocity for this prop
        thisVel = vels[:, propNo]
        thisRPM = rpm_now[propNo]

        # Total horizontal velocity
        v_xy = math.sqrt(thisVel[0]**2+thisVel[1]**2)
        v_z = thisVel[2];

        # Coefficients for the quartic
        p4 = k4**2
        p3 = 2*v_z*k4**2
        p2 = k4**2*v_z**2 - k2**2*thisRPM**2 + k4**2*v_xy
        p1 = -(2*k2**2*thisRPM**2*v_z + 2*k1*k2*thisRPM**3)
        p0 = -(k1**2*thisRPM**4 + k2**2*thisRPM**2*v_z**2 + 2*k1*k2*thisRPM**3*v_z)

        # Find all roots of the quartic
        try:
            allRoots = np.roots([p4, p3, p2, p1, p0])

             # We need the postive and real root
            for i in range(0, 4):
                root = allRoots[i]
                if np.isreal(root) and np.real(root)>0:
                    vi = np.real(root)
                    break

            # If we couldn't find a real root set it reasonably
            if not 'vi' in locals():
                print('Warning: real root not found')
                vi = 5
                flagErr = True

            thrust[propNo] = k1*thisRPM**2+k2*thisRPM*(v_z+vi)

        except np.linalg.linalg.LinAlgError as err:
            vi = 5
            thrust[propNo] = 1.5
            print('Warning: could not find solve quartic for induced velocity. Thrust:')
            print thrust
            flagErr = True


    return thrust, flagErr
