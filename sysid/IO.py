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

import time
import pickle
import yaml
import collections
import pandas as pd

def saveResults(data, estimator, resultsFolder):

    sess = data.directory.rsplit('/', 2)[-2:]
    datestr = time.strftime("%Y%m%d-%H%M")
    data_dir = resultsFolder + '/' + sess[0] + '_'+ sess[1] + '_' + estimator.method + '_' + datestr + '.dat'
    yaml_dir = resultsFolder + '/' + sess[0] + '_'+ sess[1] + '_' + estimator.method + '_' + datestr + '.yaml'

    with open(data_dir, "wb") as file:
        pickle.dump([data, estimator], file)

    createYaml(estimator, yaml_dir)

def saveResults2(run, purpose, data, estimator, resultsFolder):

    sess = data.directory.rsplit('/', 2)[-2:]
    datestr = time.strftime("%Y%m%d-%H%M")
    data_dir = resultsFolder + '/' + str(run) + '_' + purpose + '_' + sess[0] + '_'+ sess[1] + '_' + estimator.method + '_' + datestr + '.dat'
    yaml_dir = resultsFolder + '/'+ str(run) + '_' + purpose + '_' + sess[0] + '_'+ sess[1] + '_' + estimator.method + '_' + datestr + '.yaml'

    with open(data_dir, "wb") as file:
        pickle.dump([data, estimator], file)

    createYaml(estimator, yaml_dir)

def createYaml(estimator, yaml_dir):

    # Convert the pandas dataframe to a dict
    theta_dict = collections.OrderedDict()
    stddevInd = 0

    # Loop through the theta dataframe
    for df_row in estimator.Theta.itertuples():

        # If this variable was estimated
        if df_row[3]:

            # Extract the standard deviation
            stddev = estimator.stddev[stddevInd]

            # Iterate to the next standard deviation
            stddevInd = stddevInd + 1

        # Otherwise set the standard deviation to zero
        else:

            # If is was not estimated call the standard deviation zero
            stddev = 0.0

        theta_dict[df_row[0]] = {'stddev': str(stddev), 'estimated': str(df_row[3]), 'value': str(df_row[1])}

    with open(yaml_dir, 'w') as yaml_file:
        yaml.dump(theta_dict, yaml_file, default_flow_style=False)

def loadFile(thetaDir):

    Theta = pd.DataFrame(columns=['value', 'stddev', 'estimate'])

    # Read YAML file
    with open(thetaDir, 'r') as stream:
        theta_dict = yaml.load(stream)

    # Loop through the ordered dict
    for key, value in theta_dict.items():
        Theta.loc[key] = {'value': float(value['value']), 'stddev':float(value['stddev']), 'estimate':str2bool(value['estimated'])}
    return Theta

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")
