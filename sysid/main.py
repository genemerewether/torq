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
from FlightData import FlightData
from Estimator import Estimator
import pandas as pd

Theta = pd.DataFrame(columns=['value', 'estimate'])

############################# USER INPUT #####################################

directory = "/home/marcrigter/Documents/SystemID_PYTHON/SystemID/FlightData/August30/sess096/log001"
directory = "/home/marcrigter/Documents/SystemID_PYTHON/SystemID/FlightData/2017-08-02AeroCharacterisationTests/sess021/log002"
directory = "/home/marcrigter/Documents/SystemID_PYTHON/SystemID/FlightData/2017-08-02AeroCharacterisationTests/sess025/log002"

# Not intended for only estimating the thrust coefficients
estimateThrust = False

# The starting row to begin using the data, and the number of rows to use
startRow = 210
nRows = 60

# For low pass filtering data
attCutoffHz = 20
rpmCutoffHz = 40

# Exit criterion for optimisation
tol = 1e-6
maxIter = 20

# Use every nth observation
n = 1

# Optimisation step. Set to 1 for normal Gauss Newton, smaller for more conservative
step = 0.3

# Force and moment disturbance covariances
Q = np.diag([1,1,1,4e-5,4e-5,4e-5])

# Covariance of sensors
covSens = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4])

# Flag this if we need to correct the position and velocity data to the cg of the quad manually.
# Note that this can also be done by flagging tangoDx, Dy, Dz to be estimated.
correctPosVel = False
cgToDatum = np.array([0.080, 0.0, 0.0])

# Inertial
Theta.loc['m']   = {'value': 0.603, 'estimate': False}
Theta.loc['Ixx'] = {'value': 0.005, 'estimate': True}
Theta.loc['Iyy'] = {'value': 0.005, 'estimate': True}
Theta.loc['Izz'] = {'value': 0.005, 'estimate': True}

# Geometric
Theta.loc['deltaDx'] = {'value': 0.000, 'estimate': True}
Theta.loc['deltaDy'] = {'value': 0.000, 'estimate': True}
Theta.loc['Dx'] = {'value': 0.0880, 'estimate': False} # Cannot estimate this, must be set to known value
Theta.loc['Dy'] = {'value': 0.0880, 'estimate': False} # Cannot estimate this, must be set to known value

# Aerodynamic
Theta.loc['Cq'] = {'value': 1.213e-10, 'estimate': False} # Cannot estimate this, must be set to known value
Theta.loc['k1'] = {'value': 2.1e-8, 'estimate': False}
Theta.loc['k2'] = {'value': -1.5e-5, 'estimate': False}
Theta.loc['k4'] = {'value': 0.025, 'estimate': False}	# Recommended to leave this set. Difficult to converge otherwise
Theta.loc['Cdh'] = {'value': 0.03, 'estimate': True}
Theta.loc['Cdx'] = {'value': 2e-2, 'estimate': False}
Theta.loc['Cdy'] = {'value': 2e-2, 'estimate': False}
Theta.loc['Cdz'] = {'value': 2e-2, 'estimate': False}

# Related to the pose sensors
Theta.loc['R0'] = {'value': 0.999, 'estimate': False}
Theta.loc['R1'] = {'value': -0.012, 'estimate': False}
Theta.loc['R2'] = {'value': -0.017, 'estimate': False}
Theta.loc['R3'] = {'value': 0.0, 'estimate': False} #Note: this is only observable when you are flying around with x and y velocity
Theta.loc['tangoDx'] = {'value': 0.0, 'estimate': False}
Theta.loc['tangoDy'] = {'value': 0.0, 'estimate': False}
Theta.loc['tangoDz'] = {'value': 0.0, 'estimate': False}

##############################################################################


data = FlightData(directory, startRow, nRows, attCutoffHz, rpmCutoffHz, n, correctPosVel, cgToDatum)
data.plot()


# Run estimator
estimator = Estimator(Theta, tol, maxIter, step, Q, estimateThrust, covSens)
estimator.run(data)
