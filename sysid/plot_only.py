#!/usr/bin/env python2
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

Created on Fri Oct 20 15:23:31 2017

@author: marcrigter
"""

from Plot import plotData
from FlightData import FlightData

# For low pass filtering data
attCutoffHz = 20
rpmCutoffHz = 40

# Exit criterion for optimisation
tol = 1e-5
maxIter = 12

# Skip rpm values if you want improved runtime. Else set to 1.
rpmStep = 5

# Use every nth observation
n = 2

startRow = 1
nRow = 20000

directory = "/home/marcrigter/Documents/SystemID_PYTHON/SystemID/FlightData/August30/sess096/log001"
# Extract the data to use
data = FlightData(directory, startRow, nRow, attCutoffHz, rpmCutoffHz, n)

plotData(data)
