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

import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np

def plotResults(estimator, fileIndex, I_SW):
    xs_meas = estimator.xs_meas
    xs_est = estimator.xs_est
    obs_t = estimator.obs_t
    xs_pred_start = estimator.xs_pred_start
    xs_pred_end = estimator.xs_pred_end
    ts_pred = estimator.ts_pred
    resid_end = estimator.resid_end
    resid_start = estimator.resid_start
    theta_history = estimator.theta_history

    # The tango offset from the geometric centre is the vector of centre to cg plus cg to tango
    theta_history[:,20:21] = theta_history[:,20:21] + theta_history[:,4:5]

    # Plot the evoluation of the cost
    fig = plt.figure("History of optimisation")
    plt.subplot(3, 5, 1)
    plt.plot(np.arange(len(estimator.cost_history)), np.squeeze(np.array(estimator.cost_history)), "k", linewidth=2.0)
    plt.xlabel('Iterations')
    plt.title('Cost Evolution')

    tango_offset = [0.088, 0.004]

    titles = ["Mass", "Inertias", "CG offset", "Cq", "k1", "k2", "Blade Drag", "Airframe drag", "Tango rotation", "Tango offset from geometric centre"]
    theta_indices = [[0], [1, 2, 3], [4, 5], [8], [9], [10], [12], [13, 15],  [17, 18, 19], [20, 21, 22]]
    ylabels = ["kg", "kg.m^2", "m", "", "", "", "Cd,h", "", "quaternion", "m"]
    labels = [[""], ["Ixx", "Iyy", "Izz"], ["deltaDx", "deltaDy"], [""], [""], [""], [""], ["Cd,xy", "Cd,z"], ["qx", "qy", "qz"], ["x", "y", "z"]]
    stddevInd = 0
    rowNames = list(estimator.Theta.index)
    for i in range(0, len(titles)):
        ax = fig.add_subplot(3,5,i+2)
        indices = theta_indices[i]
        label = labels[i]
        colors = "kbr"
        if i == 7:
            colors = "kr"

        for j in range(0, len(theta_indices[i])):


            plt.plot(np.arange(len(theta_history[:, 1])), theta_history[:, indices[j]], colors[j], label = label[j], linewidth=2.0)
            if len(label) > 1 and fileIndex == 0:
                plt.legend(loc = 3, fontsize = 8)

            # If this is the inertias plot SW
            if i == 1:
                plt.plot([0, len(theta_history)], [I_SW[j], I_SW[j]], colors[j]+'--', linewidth = 2.0)

            # If this is the tango one plot that
            if i == 9 and j <= 1:
                plt.plot([0, len(theta_history)], [tango_offset[j], tango_offset[j]], colors[j]+'--', linewidth = 2.0)

            # If we have standard deviations available, plot them
            if hasattr(estimator, 'stddev'):

                # If we were estimating this variable
                if estimator.Theta['estimate'][rowNames[indices[j]]]:
                    ax.errorbar((len(theta_history[:, 1])-1), theta_history[-1, indices[j]], estimator.stddev[stddevInd], marker='s', mfc=colors[j], mec=colors[j], ms=4, mew=2, elinewidth=2.0, ecolor=colors[j])
                    stddevInd = stddevInd + 1

        plt.xlabel('Iterations')
        plt.ylabel(ylabels[i])
        plt.title(titles[i])
        plt.tight_layout()

#    # Plot the residuals initially
#    fig = plt.figure()
#    st = fig.suptitle("Modelled vs actual values before optimisation", fontsize="x-large")
#    st.set_y(0.95)
#    fig.subplots_adjust(top=0.85)
    colors = ["kbr", "kbr", "gkbr", "kbr"]
    titles = ["position", "velocity", "quaternion", "rates"]
    indices = [[0,1, 2], [3, 4, 5], [6, 7, 8, 9], [10, 11, 12]]
    labels = [["x", "y", "z"], ["x", "y", "z"], ["q0", "q1", "q2", "q3"], ["x", "y", "z"]]
    ylabels = ["m", "m/s", "", "rad/s"]
#    for i in range(0, 4):
#        plt.subplot(2, 2, i+1)
#        index = indices[i]
#        label = labels[i]
#        color = colors[i]
#        for j in range(0, len(index)):
#            plt.plot(obs_t, xs_meas[:, index[j]], color[j], label=label[j], linewidth=2.0)
#            plt.plot(obs_t, xs_meas[:, index[j]], color[j]+"x",linewidth=2.0)
#            plt.plot(ts_pred, xs_pred_start[:,index[j]], color[j]+"o", linewidth=2.0, mew=1.5, ms=6)
#        plt.legend()
#        plt.title(titles[i])
#        plt.xlabel('Seconds')
#        plt.ylabel(ylabels[i])

def plotResiduals(estimator, fileIndex, I_SW):
    xs_meas = estimator.xs_meas
    xs_est = estimator.xs_est
    obs_t = estimator.obs_t
    xs_pred_start = estimator.xs_pred_start
    xs_pred_end = estimator.xs_pred_end
    ts_pred = estimator.ts_pred
    resid_end = estimator.resid_end
    resid_start = estimator.resid_start
    theta_history = estimator.theta_history

    # Plot the residuals initially
    fig = plt.figure()
    colors = ["kbr", "kbr", "gkbr", "kbr"]
    titles = ["position", "velocity", "quaternion", "rates"]
    indices = [[0,1, 2], [3, 4, 5], [6, 7, 8, 9], [10, 11, 12]]
    labels = [["x", "y", "z"], ["x", "y", "z"], ["q0", "q1", "q2", "q3"], ["x", "y", "z"]]
    ylabels = ["m", "m/s", "", "rad/s"]
    st = fig.suptitle("Modelled vs actual values after optimisation", fontsize="x-large")
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)

    for i in range(0, 4):
        plt.subplot(2, 2, i+1)
        index = indices[i]
        label = labels[i]
        color = colors[i]
        for j in range(0, len(index)):
            plt.plot(obs_t, xs_est[:, index[j]], color[j], label=label[j], linewidth=2.0)
            plt.plot(obs_t, xs_meas[:, index[j]], color[j]+"x",linewidth=2.0, mew=1.5, ms=4)
            plt.plot(ts_pred, xs_pred_end[:,index[j]], color[j]+"o", linewidth=2.0, mew=1.5, ms=4)
        plt.legend()
        plt.title(titles[i])
        plt.xlabel('Seconds')
        plt.ylabel(ylabels[i])

         # loop through the residuals not used
        if hasattr(estimator, 'model_resid_not_used'):
            for n in estimator.model_resid_not_used:

                # These are as nObs. To bring this back to an index for predition need to subtract 1
                resid_index = n-1
                for j in range(0, len(index)):
                    plt.plot(ts_pred[resid_index], xs_pred_end[resid_index,index[j]], color[j]+"x", linewidth=2.0, mew=1.5, ms=8)

    fig = plt.figure()
    plt.figtext(0.5,0.95, "Residuals before optimisation", ha="center", va="top", fontsize=14, color="k")
    plt.figtext(0.5,0.5, "Residuals after optimisation", ha="center", va="top", fontsize=14, color="k")
    fig.subplots_adjust(hspace=.5)
    colors = ["kbr", "kbr", "kbr", "kbr"]
    titles = ["position", "velocity", "quaternion", "rates"]
    indices = [[0,1, 2], [3, 4, 5], [ 7, 8, 9], [10, 11, 12]]
    labels = [["x", "y", "z"], ["x", "y", "z"], [ "q1", "q2", "q3"], ["x", "y", "z"]]
    ylabels = ["m", "m/s", "", "rad/s"]
    for i in range(0, 4):
        plt.subplot(2, 4, i+1)
        index = indices[i]
        label = labels[i]
        color = colors[i]
        for j in range(0, len(index)):
            matplotlib.pyplot.stem(ts_pred, resid_start[:, index[j]], linefmt=color[j]+'-', markerfmt=color[j]+'o', basefmt='r-', label = label[j])

        plt.legend()
        plt.title(titles[i])
        plt.xlabel('Seconds')
        plt.ylabel(ylabels[i])

        plt.subplot(2, 4, i+5)
        index = indices[i]
        label = labels[i]
        color = colors[i]

        if i == 0:
            divisor = 45
        elif i == 1:
            divisor = 15
        elif i == 2:
            divisor = 20
        elif i ==3:
            divisor = 5
        for j in range(0, len(index)):
            plt.plot(obs_t, xs_est[:, index[j]]/divisor, color[j], label=label[j], linewidth=2.0, alpha = 0.3)
            matplotlib.pyplot.stem(ts_pred, resid_end[:, index[j]], linefmt=color[j]+'-', markerfmt=color[j]+'o', basefmt='r-', label = label[j])

            # Check if we ahve model residuals
            if hasattr(estimator, 'model_resid_not_used'):

                # The bad residual indexes are -1 of the array
                bad_resid_indexes = estimator.model_resid_not_used - 1

                # Plot crosses over the bad residuals
                plt.plot(ts_pred[bad_resid_indexes], resid_end[bad_resid_indexes, index[j]], color[j]+'x', markersize = '9', linewidth=2.0, mew=1.5,)

        plt.legend()
        plt.title(titles[i])
        plt.xlabel('Seconds')
        plt.ylabel(ylabels[i])





def plotData(data):

    plt.figure()
    plt.subplot(231)
    plt.plot(data.posvel_t, data.pos[:,0], 'k', label="x", linewidth=2.0)
    plt.plot(data.posvel_t, data.pos[:,1], 'b', label="y", linewidth=2.0)
    plt.plot(data.posvel_t, data.pos[:,2], 'r', label="z", linewidth=2.0)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('m')
    plt.title('Position')
    plt.xlim((data.posvel_t[0], data.posvel_t[-1]))
    data
    plt.subplot(232)
    plt.plot(data.posvel_t, data.vel[:,0], 'k', label="x", linewidth=2.0)
    plt.plot(data.posvel_t, data.vel[:,1], 'b', label="y", linewidth=2.0)
    plt.plot(data.posvel_t, data.vel[:,2], 'r', label="z", linewidth=2.0)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('m/s')
    plt.title('Velocity')
    plt.xlim((data.posvel_t[0], data.posvel_t[-1]))

    plt.subplot(233)
    plt.plot(data.posvel_t, data.quat[:,0], 'g', label="q0", linewidth=2.0)
    plt.plot(data.posvel_t, data.quat[:,1], 'k', label="q1", linewidth=2.0)
    plt.plot(data.posvel_t, data.quat[:,2], 'b', label="q2", linewidth=2.0)
    plt.plot(data.posvel_t, data.quat[:,3], 'r', label="q3", linewidth=2.0)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.title('Quaternion')
    plt.xlim((data.posvel_t[0], data.posvel_t[-1]))

    plt.subplot(234)
    plt.plot(data.posvel_t, data.omega[:,0], 'k', label="x", linewidth=2.0)
    plt.plot(data.posvel_t, data.omega[:,1], 'b', label="y", linewidth=2.0)
    plt.plot(data.posvel_t, data.omega[:,2], 'r', label="z", linewidth=2.0)
    plt.plot(data.att_raw_t, data.omega_raw[:,0], 'kx', label="", linewidth=2.0)
    plt.plot(data.att_raw_t, data.omega_raw[:,1], 'bx', label="", linewidth=2.0)
    plt.plot(data.att_raw_t, data.omega_raw[:,2], 'rx', label="", linewidth=2.0)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.title('Angular Rates')
    plt.xlim((data.posvel_t[0], data.posvel_t[-1]))

    plt.subplot(235)
    plt.plot(data.rpm_t, data.rpm[:,0], 'k', label="1", linewidth=2.0)
    plt.plot(data.rpm_t, data.rpm[:,1], 'b', label="2", linewidth=2.0)
    plt.plot(data.rpm_t, data.rpm[:,2], 'r', label="3", linewidth=2.0)
    plt.plot(data.rpm_t, data.rpm[:,3], 'g', label="4", linewidth=2.0)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('RPM')
    plt.title('Propeller RPMs')
    plt.plot(data.rpm_t, data.rpm_raw[:,0], 'kx', label="", linewidth=2.0)
    plt.plot(data.rpm_t, data.rpm_raw[:,1], 'bx', label="", linewidth=2.0)
    plt.plot(data.rpm_t, data.rpm_raw[:,2], 'rx', label="", linewidth=2.0)
    plt.plot(data.rpm_t, data.rpm_raw[:,3], 'gx', label="", linewidth=2.0)
    plt.xlim((data.posvel_t[0], data.posvel_t[-1]))
    plt.ylim((0, 25000))

def plotStdDev(estimator, fileIndex, I_SW):
    theta_history = estimator.theta_history
     # If the standard deviation was saved plot it
    if hasattr(estimator, 'stddev'):
        rows, cols = estimator.Theta.shape
        rowNames = list(estimator.Theta.index)
        stddevIndex = 0
        for row in range(0, rows):
            if estimator.Theta['estimate'][rowNames[row]]:
                fig = plt.figure(rowNames[row])
                ax = plt.gca()
                if rowNames[row] == "Ixx":
                    ax.errorbar(fileIndex, theta_history[-1, row], estimator.stddev[stddevIndex], marker='s', mfc='black', mec='black', ms=3, mew=5, elinewidth=2.0, ecolor='black')
                    plt.plot([-1, fileIndex+1], [I_SW[0], I_SW[0]], 'k--', linewidth = 2.0)
                    plt.ylabel("kg.m^2")

                if rowNames[row] == "Iyy":
                    ax.errorbar(fileIndex, theta_history[-1, row], estimator.stddev[stddevIndex], marker='s', mfc='blue', mec='blue', ms=3, mew=5, elinewidth=2.0, ecolor='blue')
                    plt.plot([-1, fileIndex+1], [I_SW[1], I_SW[1]], 'b--', linewidth = 2.0)
                    plt.ylabel("kg.m^2")

                if rowNames[row] == "Izz":
                    ax.errorbar(fileIndex, theta_history[-1, row], estimator.stddev[stddevIndex], marker='s', mfc='red', mec='red', ms=3, mew=5, elinewidth=2.0, ecolor='red')
                    plt.plot([-1, fileIndex+1], [I_SW[2], I_SW[2]], 'r--', linewidth = 2.0)
                    plt.ylabel("kg.m^2")

                if rowNames[row] == "deltaDx":
                    ax.errorbar(fileIndex, theta_history[-1, row], estimator.stddev[stddevIndex], marker='s', mfc='black', mec='black', ms=3, mew=5, elinewidth=2.0, ecolor='black')
                    plt.ylabel("m")

                if rowNames[row] == "deltaDy":
                    ax.errorbar(fileIndex, theta_history[-1, row], estimator.stddev[stddevIndex], marker='s', mfc='blue', mec='blue', ms=3, mew=5, elinewidth=2.0, ecolor='blue')
                    plt.ylabel("m")

                if rowNames[row] == "Cq":
                    ax.errorbar(fileIndex, theta_history[-1, row], estimator.stddev[stddevIndex], marker='s', mfc='k', mec='k', ms=3, mew=5, elinewidth=2.0, ecolor='k')
                    plt.plot([-1, fileIndex+1], [2.94e-10, 2.94e-10], 'k--', linewidth = 2.0)

                if rowNames[row] == "tangoDx":
                    ax.errorbar(fileIndex, theta_history[-1, row], estimator.stddev[stddevIndex], marker='s', mfc='k', mec='k', ms=3, mew=5, elinewidth=2.0, ecolor='k')
                    plt.plot([-1, fileIndex+1], [0.089, 0.089], 'k--', linewidth = 2.0)
                    plt.ylabel("m")

                if rowNames[row] == "tangoDy":
                    ax.errorbar(fileIndex, theta_history[-1, row], estimator.stddev[stddevIndex], marker='s', mfc='b', mec='b', ms=3, mew=5, elinewidth=2.0, ecolor='b')
                    plt.plot([-1, fileIndex+1], [0.004, 0.004], 'b--', linewidth = 2.0)
                    plt.ylabel("m")

                if rowNames[row] == "tangoDz":
                    ax.errorbar(fileIndex, theta_history[-1, row], estimator.stddev[stddevIndex], marker='s', mfc='r', mec='r', ms=3, mew=5, elinewidth=2.0, ecolor='r')
                    plt.ylabel("m")

                if rowNames[row] == "R1":
                    ax.errorbar(fileIndex, theta_history[-1, row], estimator.stddev[stddevIndex], marker='s', mfc='k', mec='k', ms=3, mew=5, elinewidth=2.0, ecolor='k')

                if rowNames[row] == "R2":
                    ax.errorbar(fileIndex, theta_history[-1, row], estimator.stddev[stddevIndex], marker='s', mfc='b', mec='b', ms=3, mew=5, elinewidth=2.0, ecolor='b')

                if rowNames[row] == "R3":
                    ax.errorbar(fileIndex, theta_history[-1, row], estimator.stddev[stddevIndex], marker='s', mfc='r', mec='r', ms=3, mew=5, elinewidth=2.0, ecolor='r')

                if rowNames[row] == "k1":
                    ax.errorbar(fileIndex, theta_history[-1, row], estimator.stddev[stddevIndex], marker='s', mfc='k', mec='k', ms=3, mew=5, elinewidth=2.0, ecolor='k')

                if rowNames[row] == "k2":
                    ax.errorbar(fileIndex, theta_history[-1, row], estimator.stddev[stddevIndex], marker='s', mfc='k', mec='k', ms=3, mew=5, elinewidth=2.0, ecolor='k')

                if rowNames[row] == "Cdh":
                    ax.errorbar(fileIndex, theta_history[-1, row], estimator.stddev[stddevIndex], marker='s', mfc='k', mec='k', ms=3, mew=5, elinewidth=2.0, ecolor='k')

                if rowNames[row] == "Cdz":
                    ax.errorbar(fileIndex, theta_history[-1, row], estimator.stddev[stddevIndex], marker='s', mfc='r', mec='r', ms=3, mew=5, elinewidth=2.0, ecolor='r')

                if rowNames[row] == "Cdxy":
                    ax.errorbar(fileIndex, theta_history[-1, row], estimator.stddev[stddevIndex], marker='s', mfc='k', mec='k', ms=3, mew=5, elinewidth=2.0, ecolor='k')

                plt.xlim([-1, fileIndex + 1])
                stddevIndex = stddevIndex + 1
                plt.title(rowNames[row])
                plt.xlabel("Dataset")
