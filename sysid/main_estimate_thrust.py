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

# clear memory

import numpy as np
from FlightData import FlightData
from Estimator import Estimator
import pandas as pd
from IO import saveResults
from Plot import plotResults
import matplotlib
import copy
from Estimator import DataStore
from Estimator import EstimatorStore
from IO import loadFile
from Plot import plotData
from Plot import plotResiduals

############################# USER INPUT #####################################

directories = ["Data/VerticalFlight/log001"]

startRows = [2300]
nRows = [1100]

# For low pass filtering data
attCutoffHz = 20
rpmCutoffHz = 40

# Exit criterion for optimisation
tol = 1e-5
breakEarly = True
maxIter = 10
plot = True

# Skip rpm values if you want improved runtime. Else set to 1.
rpmStep = 4

# Use every nth observation
n = 5

# Method can either be "WLS" or "fastLTS". If this option is neither, it runs fastLTS by default.
method = 'WLS'

# Optimisation step. Set to 1 for normal Gauss Newton, smaller for more conservative
step = 1.0

# Folder for saving results
resultsFolder = "Results"

# Params for fastLTS
nSeedSamples = 15
max_cstep = 10
nBest = 3
propToRemove = 0.4

# Force and moment disturbance covariances
Q = np.diag([1,1,1,1e-5,1e-5,1e-5])

# Covariance of sensors
covSens = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 2e-4, 2e-4, 2e-4])*0.1

# Load in the starting parameter structure
loadTheta = False
thetaDir = '/home/marcrigter/Documents/SystemID_PYTHON/SystemID/Results/sess096_log001_fastLTS_20171001-1118.yaml'

if loadTheta:
    Theta = loadFile(thetaDir)
else:
    Theta = pd.DataFrame(columns=['value', 'stddev', 'estimate'])

    # Inertial
    Theta.loc['m']   = {'value': 0.697, 'stddev':0.0, 'estimate': False}
    Theta.loc['Ixx'] = {'value': 0.003, 'stddev':0.0, 'estimate': False}
    Theta.loc['Iyy'] = {'value': 0.004, 'stddev':0.0, 'estimate': False}
    Theta.loc['Izz'] = {'value': 0.006, 'stddev':0.0, 'estimate': False}

    # Geometric
    Theta.loc['deltaDx'] = {'value': 0.000,'stddev':0.0,  'estimate': True}
    Theta.loc['deltaDy'] = {'value': 0.000, 'stddev':0.0, 'estimate': True}
    Theta.loc['Dx'] = {'value': 0.0880, 'stddev':0.0, 'estimate': False} # Cannot estimate this, must be set to known value
    Theta.loc['Dy'] = {'value': 0.0880, 'stddev':0.0, 'estimate': False} # Cannot estimate this, must be set to known value

    # Aerodynamic
    Theta.loc['Cq'] = {'value': 1.44e-10, 'stddev':0.0, 'estimate': False}
    Theta.loc['k1'] = {'value': 2.2e-08, 'stddev':0.0, 'estimate': True}
    Theta.loc['k2'] = {'value': -10.0e-06, 'stddev':0.0, 'estimate':True}
    Theta.loc['k4'] = {'value': 0.02367, 'stddev':0.0, 'estimate': False}	# Recommended to leave this set. Difficult to converge otherwise
    Theta.loc['Cdh'] = {'value': 0.0, 'stddev':0.0, 'estimate':False}
    Theta.loc['Cdxy'] = {'value': 0.00, 'stddev':0.0, 'estimate':False}
    Theta.loc['Cdy'] = {'value': 0, 'stddev':0.0, 'estimate': False}
    Theta.loc['Cdz'] = {'value': 0.00, 'stddev':0.0, 'estimate':False}

    # Related to the pose sensors
    Theta.loc['R0'] = {'value': 1.0, 'stddev':0.0, 'estimate': True}
    Theta.loc['R1'] = {'value': 0.0, 'stddev':0.0, 'estimate': True}
    Theta.loc['R2'] = {'value': 0.0, 'stddev':0.0, 'estimate': True}
    Theta.loc['R3'] = {'value': 0.0, 'stddev':0.0, 'estimate':False} #Note: this is only observable when you are flying around with x and y velocity
    Theta.loc['tangoDx'] = {'value': 0.05, 'stddev':0.0, 'estimate': False}
    Theta.loc['tangoDy'] = {'value': 0.0, 'stddev':0.0, 'estimate': False}
    Theta.loc['tangoDz'] = {'value': 0.0, 'stddev':0.0, 'estimate':False}

# Close open figures
matplotlib.pyplot.close("all")

I_SW = [0.00241, 0.00342, 0.005221]

# Run the estimator for all files
for i in range(0, len(directories)):

    # Extract the information specific to this run
    directory = directories[i]
    startRow = startRows[i]
    nRow = nRows[i]
    print("Opening: " + directory)

    # Extract the data to use
    data = FlightData(directory, startRow, nRow, attCutoffHz, rpmCutoffHz, n)

    # Run estimator
    estimator = Estimator(copy.deepcopy(Theta), tol, maxIter, step, Q, covSens, nSeedSamples, max_cstep, nBest, propToRemove, method)
    estimator.run(data, breakEarly, rpmStep)

    # Save results
    dataStore = DataStore(data)
    estimatorStore = EstimatorStore(estimator)
    saveResults(dataStore, estimatorStore, resultsFolder)

    # Plot the results
    if plot:
        plotData(dataStore)
        plotResults(estimatorStore, i, I_SW)
        plotResiduals(estimatorStore, i, I_SW)

    # If loading from file always start from that point
    if loadTheta:
        Theta = loadFile(thetaDir)
