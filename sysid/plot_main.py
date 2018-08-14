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

import pickle
from Plot import plotResults
import matplotlib
from Plot import plotData
from Plot import plotStdDev
from Plot import plotResiduals


# These are the Nightwing thrust results
files = ['/home/marcrigter/Documents/SystemID_PYTHON/SystemID/ResultsToKeep/Nightwing/Thrust/sess047_log001_WLS_20171023-1817.dat',
'/home/marcrigter/Documents/SystemID_PYTHON/SystemID/ResultsToKeep/Nightwing/Thrust/sess048_log001_WLS_20171023-1827.dat',
'/home/marcrigter/Documents/SystemID_PYTHON/SystemID/ResultsToKeep/Nightwing/Thrust/sess049_log001_WLS_20171023-1837.dat',
'/home/marcrigter/Documents/SystemID_PYTHON/SystemID/ResultsToKeep/Nightwing/Thrust/sess050_log001_WLS_20171023-1846.dat']

I_SW = [0.00241, 0.00342, 0.005221]

# Close open figures
matplotlib.pyplot.close("all")

for i in range(0, len(files)):

    with open(files[i], "rb") as f:
        data, estimator = pickle.load(f)

    plotResults(estimator, i, I_SW)
    plotStdDev(estimator, i, I_SW)
    plotResiduals(estimator, i, I_SW)
