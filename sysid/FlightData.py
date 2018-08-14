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

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
from pyquaternion import Quaternion

class FlightData:

    def __init__(self, input_dir, startRow, nRow, attCutoffHz, rpmCutoffHz, n):
        self.startRow = startRow
        self.nRow = nRow
        self.directory = input_dir

        att_fname = input_dir+"_vehicle_attitude_0.csv"
        posvel_fname = input_dir+"_vehicle_local_position_0.csv"
        rpm_fname = input_dir+"_actuator_feedbacks_0.csv"

        posvel_t_full = np.genfromtxt(posvel_fname, delimiter=",", usecols=(0), skip_header=startRow, max_rows=nRow)/1e6
        pos_full = np.genfromtxt(posvel_fname, delimiter=",", usecols=(5,6,7), skip_header=startRow, max_rows=nRow)
        vel_full = np.genfromtxt(posvel_fname, delimiter=",", usecols=(11,12,13), skip_header=startRow, max_rows=nRow)

        self.posvel_t = posvel_t_full[0:-1:n]
        self.pos = pos_full[0:-1:n, :]
        self.vel = vel_full[0:-1:n, :]
        self.obs_t = self.posvel_t

        self.omega_raw = np.genfromtxt(att_fname, delimiter=",", usecols=(1,2,3), skip_header=1)
        self.quat_raw = np.genfromtxt(att_fname, delimiter=",", usecols=(4,5,6,7), skip_header=1)
        self.att_raw_t = np.genfromtxt(att_fname, delimiter=",", usecols=(0), skip_header=1)/1e6

        self.rpm_raw = np.genfromtxt(rpm_fname, delimiter=",", usecols=(2,3,4,5), skip_header=1)
        self.rpm_t = np.genfromtxt(rpm_fname, delimiter=",", usecols=(0), skip_header=1)/1e6

        q0 = np.interp(self.posvel_t, self.att_raw_t, self.quat_raw[:, 0]).reshape(self.posvel_t.size, 1)
        q1 = np.interp(self.posvel_t, self.att_raw_t, self.quat_raw[:, 1]).reshape(self.posvel_t.size, 1)
        q2 = np.interp(self.posvel_t, self.att_raw_t, self.quat_raw[:, 2]).reshape(self.posvel_t.size, 1)
        q3 = np.interp(self.posvel_t, self.att_raw_t, self.quat_raw[:, 3]).reshape(self.posvel_t.size, 1)
        quat_array = np.concatenate((q0, q1, q2, q3), axis=1)

        # Need to normalise each quaternion
        for i in range(0, len(self.posvel_t)):
            quat = quat_array[i, :]/np.linalg.norm(quat_array[i, :])
            quat_array[i, :] = quat
        self.quat = quat_array

        # Create low pass filter for the attitude
        order = 8
        changeInTime = self.att_raw_t[-1]-self.att_raw_t[0]
        freq = self.att_raw_t.size/changeInTime
        nyquist = freq/2
        fracNyquist = attCutoffHz/nyquist

        # Filter the rates
        b, a = signal.butter(order, fracNyquist)
        roll_rate = signal.filtfilt(b, a, self.omega_raw[:, 0], axis=0, padlen = 0)
        pitch_rate = signal.filtfilt(b, a, self.omega_raw[:, 1], axis=0, padlen = 0)
        yaw_rate = signal.filtfilt(b, a, self.omega_raw[:, 2], axis=0, padlen = 0)
        roll_rate = np.interp(self.posvel_t, self.att_raw_t, roll_rate).reshape(self.posvel_t.size, 1)
        pitch_rate = np.interp(self.posvel_t, self.att_raw_t, pitch_rate).reshape(self.posvel_t.size, 1)
        yaw_rate = np.interp(self.posvel_t, self.att_raw_t, yaw_rate).reshape(self.posvel_t.size, 1)
        self.omega = np.concatenate((roll_rate, pitch_rate, yaw_rate), axis=1)

        # Create low pass filter for the rpms
        order = 8
        changeInTime = self.rpm_t[-1]-self.rpm_t[0]
        freq = self.rpm_t.size/changeInTime
        nyquist = freq/2
        fracNyquist = rpmCutoffHz/nyquist

        # Filter the RPM
        b, a = signal.butter(order, fracNyquist)
        rpm1 = signal.filtfilt(b, a, self.rpm_raw[:, 0], axis=0).reshape(self.rpm_t.size, 1)
        rpm2 = signal.filtfilt(b, a, self.rpm_raw[:, 1], axis=0).reshape(self.rpm_t.size, 1)
        rpm3 = signal.filtfilt(b, a, self.rpm_raw[:, 2], axis=0).reshape(self.rpm_t.size, 1)
        rpm4 = signal.filtfilt(b, a, self.rpm_raw[:, 3], axis=0).reshape(self.rpm_t.size, 1)
        self.rpm = np.concatenate((rpm1, rpm2, rpm3, rpm4), axis=1)
