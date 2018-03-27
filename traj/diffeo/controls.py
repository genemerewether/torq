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
from numpy import linalg
np = numpy



def get_torques(ang_vel, ang_accel, params):
    """
    Extracts torques from given angular velocity, acceleration and quadrotor inertia matrix

    Method is with the differentially flat derivation from:
    'Minimum Snap Trajectory Generation and Control for Quadrotors'
    Mellinger and Kumar 2011, equation 4

    Args:
        ang_vel: angular velocity, length 3 np array (about x, y, z)
        ang_accel: angular acceleration, length 3 np array (about x, y, z)
        Interia: a 3x3 np array of the quadrotors inertia

    Returns:

    Raises:
    """
    #TODO(bmorrell@jpl.nasa.gov) check on interia input
    torques = np.matmul(params['Inertia'],ang_accel) + np.cross(ang_vel,np.matmul(params['Inertia'],ang_vel))

    return torques

def get_rotor_speeds(torques,thrust_mag,params):
    """
    Extracts rotor speeds from thrust and quadrotor properties

    Method is with the differentially flat derivation from:
    'Minimum Snap Trajectory Generation and Control for Quadrotors'
    Mellinger and Kumar 2011, eqn 2

    Args:
        torques: the torques around each axis, a length-3 np array
        thrust_mag: The magnitude of the thrust acceleration
        k_f: Thrust coefficent: F = k_f*rpm^2
        k_m: moment coefficient M = k_m*rpm^2
        L:   offset of motors from CG (moment arm)

    Returns:
        rpm: the revolutions per minute of the 4 rotors (a length-4 np array)

    Raises:
    """

    C_t = params['Ct']

    C_q = params['Cq']

    # Form the mapping matrix
    k_mat = np.array([[-C_t,-C_t,-C_t,-C_t],
                      [-params['Dy']*C_t,params['Dy']*C_t,params['Dy']*C_t,-params['Dy']*C_t],
                      [params['Dx']*C_t,-params['Dx']*C_t,params['Dx']*C_t,-params['Dx']*C_t],
                      [C_q,C_q,-C_q,-C_q]])

    # Create the input vector
    u_vec = -np.insert(torques,0,thrust_mag)
    u_vec[1] *= -1.0 # ENU to NED Conversion

    # Extract the rpm squared
    rpm_sq = np.matmul(np.linalg.inv(k_mat),u_vec)

    rpm = np.sqrt(rpm_sq)

    return rpm

def load_params(filename):
    """
    Load parameters describing physical traits of drone in filename (a yaml file)
    """
    import yaml

    with open(filename, 'r') as stream:
        try:
            params = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    params_out = dict()
    params_out['Inertia'] = np.diag([float(params['Ixx']['value']),float(params['Iyy']['value']),float(params['Izz']['value'])])
    params_out['mass'] = float(params['m']['value'])
    params_out['Cq'] = float(params['Cq']['value'])
    k1 = float(params['k1']['value'])
    k2 = float(params['k2']['value'])
    k4 = float(params['k4']['value'])
    params_out['Dx'] = float(params['Dx']['value'])+float(params['deltaDx']['value'])
    params_out['Dy'] = float(params['Dy']['value'])+float(params['deltaDy']['value'])

    # For simplicity convert to static thrust coefficient
    v_xy = 0
    v_z = 0
    rpm = 10000

    # Calculate the induced velocity
    p4 = k4**2
    p3 = 2*v_z*k4**2
    p2 = k4**2*v_z**2 - k2**2*rpm**2 + k4**2*v_xy
    p1 = -(2*k2**2*rpm**2*v_z + 2*k1*k2*rpm**3)
    p0 = -(k1**2*rpm**4 + k2**2*rpm**2*v_z**2 + 2*k1*k2*rpm**3*v_z)
    allRoots = np.roots([p4, p3, p2, p1, p0])
    for i in range(0, 4):
        root = allRoots[i]
        if np.isreal(root) and np.real(root)>0:
            vi = np.real(root)
            break

    # Calculate thrust coefficient at hover
    hover_ct = (k1*rpm**2+k2*rpm*(v_z+vi))/(rpm**2)
    params_out['Ct'] = hover_ct

    return params_out


def main():
    from diffeo import body_frame
    from diffeo import angular_rates_accel
    from minsnap import utils


    params = load_params("TestingCode/test_load_params.yaml")

    import pdb; pdb.set_trace()

    # Load a trajectory
    import pickle
    f = open('share/sample_output/traj.pickle','rb')
    qr_p = pickle.load(f)
    f.close

    mass = 0.48446 # kg
    Lxx = 1131729.47
    Lxy = -729.36
    Lxz = -5473.45
    Lyx = -729.36
    Lyy = 1862761.14
    Lyz = -2056.74
    Lzx = -5473.45
    Lzy = -2056.74
    Lzz = 2622183.34

    unit_conv = 1*10**-9
    Inertia = np.array([[Lxx,Lxy,Lxz],[Lyx,Lyy,Lyz],[Lzx,Lzy,Lzz]])*unit_conv

    # Thrust coefficeint
    k_f = 2.25*10**-6 # dow 5045 x3 bullnose prop
    k_m = k_f*10**-2
    L = 0.088 # meters

    # Compute times
    trans_times = utils.seg_times_to_trans_times(qr_p.times)
    t1 = np.linspace(trans_times[0], trans_times[-1], 100)

    # To test single input:
    t = t1[9]#:4]

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
    R, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'matrix')

    # Thrust
    thrust, thrust_mag = body_frame.get_thrust(accel)

    # Angular rates
    ang_vel = angular_rates_accel.get_angular_vel(thrust_mag,jerk,R,yaw_dot)

    # Angular accelerations
    ang_accel = angular_rates_accel.get_angular_accel(thrust_mag,jerk,snap,R,ang_vel,yaw_ddot)

    #--------------------------------------------------------------------------#
    # Get torque
    print("ang vel:\n {}\n ang_accel: \n{}\n Inertia: \n{}".format(ang_vel, ang_accel, params['Inertia']))
    torques = get_torques(ang_vel, ang_accel, params)
    print("Torques: {}\n".format(torques))
    print('Net Force: {}\n'.format(thrust_mag*params['mass']))
    # Get rotor speeds
    rpm = get_rotor_speeds(torques,thrust_mag*params['mass'],params)
    print("Rotor speeds:\n{}".format(rpm))
    import pdb; pdb.set_trace()


    test = dict()
    test['x'] = np.flip(np.array([[2.5000, -0.0000,  0.0000, -0.0000, -0.0000, -210.0247,  579.8746, -636.6504,  328.7759,  -66.9753],[-2.5000,  0.0000, 22.8251, -0.0000,  -59.3519,  127.8808, -296.8831,  417.5559, -274.0022, 66.9753]]).T,0)
    test['y'] = np.flip(np.array([[5.0000, -0.0000,  0.0000, -0.0000,  0.0000, 70.0364, -165.7848,  166.1452,  -81.4554, 16.0586], [10.0000, 11.3739, -0.0000,  -12.8287, -0.0000, 26.2608,  -65.4048, 92.6125,  -63.0722, 16.0586]]).T,0)
    test['z'] = np.zeros([10,2])
    test['z'][-1,:] = 3.0
    test['yaw'] = np.zeros([10,2])

    test_case = 1

    for key in qr_p.quad_traj.keys():
        if test_case == 1:
            qr_p.quad_traj[key].piece_poly.x = np.array([0.0,1.0,2.0])
            qr_p.quad_traj[key].piece_poly.c = test[key]
        elif test_case == 2:
            qr_p.quad_traj[key].piece_poly.x = qr_p.quad_traj[key].piece_poly.x[0:3]
            qr_p.quad_traj[key].piece_poly.c = qr_p.quad_traj[key].piece_poly.c[:,0:2]

    f_in = open("/media/sf_shared_torq_vm/Results/test_diffeo_in_good.txt",'w')
    f_in.write("Input:\nC_x: {}\nC_y: {}\nC_z: {}\n".format(qr_p.quad_traj['x'].piece_poly.c,qr_p.quad_traj['y'].piece_poly.c,qr_p.quad_traj['z'].piece_poly.c))
    f_in.write("Time at waypoints: {}".format(qr_p.quad_traj['x'].piece_poly.x))
    f_in.close()

    f = open("/media/sf_shared_torq_vm/Results/test_diffeo_good.txt",'w')
    # t = np.linspace(trans_times[0], trans_times[-1], n_steps)
    t = 0.5
    z = qr_p.quad_traj['z'].piece_poly(t)
    x = qr_p.quad_traj['x'].piece_poly(t)
    y = qr_p.quad_traj['y'].piece_poly(t)

    yaw = qr_p.quad_traj['yaw'].piece_poly(t)
    yaw_dot = qr_p.quad_traj['yaw'].piece_poly.derivative()(t)
    yaw_ddot = qr_p.quad_traj['yaw'].piece_poly.derivative().derivative()(t)

    f.write("t: {}\nX: {},Y: {},Z: {}\n".format(t,x,y,z))

    Vel = np.array([qr_p.quad_traj['x'].piece_poly.derivative().derivative()(t),
                    qr_p.quad_traj['y'].piece_poly.derivative().derivative()(t),
                    qr_p.quad_traj['z'].piece_poly.derivative().derivative()(t)])

    accel = np.array([qr_p.quad_traj['x'].piece_poly.derivative().derivative()(t),
                    qr_p.quad_traj['y'].piece_poly.derivative().derivative()(t),
                    qr_p.quad_traj['z'].piece_poly.derivative().derivative()(t)])

    jerk = np.array([qr_p.quad_traj['x'].piece_poly.derivative().derivative().derivative()(t),
                     qr_p.quad_traj['y'].piece_poly.derivative().derivative().derivative()(t),
                     qr_p.quad_traj['z'].piece_poly.derivative().derivative().derivative()(t)])

    snap = np.array([qr_p.quad_traj['x'].piece_poly.derivative().derivative().derivative().derivative()(t),
                     qr_p.quad_traj['y'].piece_poly.derivative().derivative().derivative().derivative()(t),
                     qr_p.quad_traj['z'].piece_poly.derivative().derivative().derivative().derivative()(t)])

    f.write("Accel: {}\nJerk: {}\nSnap: {}\n".format(accel,jerk,snap))

    # Get rotation matrix
    R, data = body_frame.body_frame_from_yaw_and_accel(yaw, accel,'matrix')

    # Thrust
    thrust, thrust_mag = body_frame.get_thrust(accel)
    import pdb; pdb.set_trace()
    # Angular rates
    ang_vel = angular_rates_accel.get_angular_vel(thrust_mag,jerk,R,yaw_dot)

    # Angular accelerations
    ang_accel = angular_rates_accel.get_angular_accel(thrust_mag,jerk,snap,R,ang_vel,yaw_ddot)

    params = controls.load_params("TestingCode/test_load_params.yaml")
    # torques
    torques = get_torques(ang_vel, ang_accel, params)

    # rpm
    rpm = get_rotor_speeds(torques,thrust_mag*mass,params)

    f.write("R: {}\nang_vel: {}\nang_accel: {}\nThrust: {}\ntorques: {}\nrpm: {}\n".format(R,ang_vel,ang_accel,thrust*mass,torques,rpm))

    f.close()
    import pdb; pdb.set_trace()
    print("Done")

if __name__ == '__main__':
    main()
