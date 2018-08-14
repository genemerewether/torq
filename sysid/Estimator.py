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
from pyquaternion import Quaternion
from Dynamics import getThrust
from Dynamics import quadDynamics
import matplotlib.pyplot as plt
import matplotlib
import math
import copy
from Plot import plotResults
import time

class Estimator:

    def __init__(self, Theta, tol, maxIter, step, Q, covSens, nSeedSamples, max_cstep, nBest, propToRemove, method):
        self.Theta = Theta
        self.tol = tol
        self.step = step
        self.maxIter = maxIter
        self.Q = Q
        self.covSens = covSens
        self.nSeedSamples = nSeedSamples
        self.max_cstep = max_cstep
        self.nBest = nBest
        self.propToRemove = propToRemove
        self.method = method

        # Store the history of theta
        self.theta_history = []
        theta_vals = Theta['value']
        self.theta_history.append([np.array(theta_vals)])
        self.cost_history = []

    def run(self, data, breakEarly, rpmStep):

        # Get the covariance weighting matrix
        W = getW(self.Q, self.Theta, self.covSens, data.posvel_t, data)
        W_diag = np.diag(W)
        length = len(W_diag)
        W_INV_STATIC = np.diag(np.ones(length)/W_diag)

        # Concatenate the required data into one array
        xs_meas =  np.concatenate((data.pos, data.vel, data.quat, data.omega), axis=1)
        self.xs_est = copy.deepcopy(xs_meas)
        nObs = data.posvel_t.size

        # Save the original theta values
        self.theta_orig = copy.deepcopy(self.Theta)

        # If its just WLS, optimise and exit
        if self.method == 'WLS':
            cost = self.optimise(data, W_INV_STATIC, breakEarly, rpmStep)
            self.saveQuit(nObs)
            return

        # Set the proportion of observations to remove for fast lts
        prop_to_remove = self.propToRemove

        # Store minimum cost from two seed samples
        nBest = self.nBest
        cost_min = np.ones(nBest)*float('inf')
        optimal_bad_resid = np.zeros([nBest, int(round(prop_to_remove*nObs))])

        # The maximum number of runs
        nCsteps = 2
        maxRuns = self.nSeedSamples*(nCsteps+1) + nBest*self.max_cstep
        run = 1
        lowestCostObserved = float('inf')

        # loop through a number of seed samples
        for seedSample in range(0, self.nSeedSamples):

#            # Randomly sample some of the observations TO BE REMOVED
#            bad_resid = np.random.choice(range(1, nObs), int(round(prop_to_remove*nObs)), replace=False)
#            W_inv = updateW(W_INV_STATIC, bad_resid)

            # The code here initialises an h subset by analogously doing the p subset thing

            # Remove all but 5 observations to create the analogy of the p-subset
            bad_resid = np.random.choice(range(1, nObs), nObs - 7, replace=False)

            # Make W_inv ignore all but 5 observations
            W_inv = updateW(W_INV_STATIC, bad_resid)

            # Create a fit to these 5 observations
            start_time = time.time()
            self.theta_history = []
            theta_vals = self.Theta['value']
            self.theta_history.append([np.array(theta_vals)])
            self.cost_history = []
            cost = self.optimise(data, W_inv, breakEarly, rpmStep)

            print("Run %g out of maximum %g completed in %s seconds (Initialisation run)" % (run, maxRuns, time.time() - start_time))
            run = run+1

            prevCost = float('inf')

            # Iterate for 2 c steps
            for cstep in range(0, nCsteps):

                # Get the h subset based on this optimisation
                bad_resid = self.get_new_resid_to_ignore(nObs, prop_to_remove, W_INV_STATIC)
                print bad_resid

                # Update the observations to ignore
                W_inv = updateW(W_INV_STATIC, bad_resid)

                self.theta_history = []
                theta_vals = self.Theta['value']
                self.theta_history.append([np.array(theta_vals)])
                self.cost_history = []

                # Run optimisation
                start_time = time.time()
                cost = self.optimise(data, W_inv, breakEarly, rpmStep)

                print("Run %g out of maximum %g completed in %s seconds" % (run, maxRuns, time.time() - start_time))
                print("Seed sample: %g, C-step %g, Cost: %g" % (seedSample, cstep, cost))
                if cost > 1.001*prevCost:
                    print("******Warning******: cost increased compared to previous cstep")
                if cost < lowestCostObserved:
                    lowestCostObserved = cost
                run = run+1
                prevCost = cost



            #If this is one of the the best run so far, save the cost and the h used
            if cost < np.max(cost_min):
                print("Seed sample: %g run added to best runs" % (seedSample))
                idx = np.argmax(cost_min)
                cost_min[idx] = cost
                optimal_bad_resid[idx, :] = bad_resid
                print cost_min

        cost_min_overall = float('inf')
        optimal_bad_resid = optimal_bad_resid.astype(int)

        # Loop through the best runs
        for bestRun in range(0, nBest):

            # Use the optimal residuals to weight W
            W_inv = updateW(W_INV_STATIC, optimal_bad_resid[bestRun, :])
            prevCost = float('inf')

            for cstep in range(0,self.max_cstep):
                self.theta_history = []
                theta_vals = self.Theta['value']
                self.theta_history.append([np.array(theta_vals)])
                self.cost_history = []

                # Run optimisation
                start_time = time.time()
                cost = self.optimise(data, W_inv, breakEarly, rpmStep)
                print("Run %g out of maximum %g completed in %s seconds" % (run, maxRuns, time.time() - start_time))
                print("Best run: %g, C-step %g, Cost: %g" % (bestRun, cstep, cost))

                if cost > 1.001*prevCost:
                    print("*******Warning******: cost increased over previous cstep")
                if cost < lowestCostObserved:
                    lowestCostObserved = cost
                run = run+1
                prevCost = cost
                # If this is the best cost so far store the cost and the bad resid used
                if cost < cost_min_overall:
                    cost_min_overall = cost
                    optimal_resid_not_used = bad_resid
                    overAllBestRun = bestRun
                    print("Best run: %g, cstep %g current best fit, cost: %g" % (bestRun, cstep, cost))

                # Save the old bad set
                prev_bad_resid = bad_resid

                # Get the h for the worst residuals
                bad_resid = self.get_new_resid_to_ignore(nObs, prop_to_remove, W_INV_STATIC)


                # If the list of bad residuals does not change we can exit
                if set(prev_bad_resid)  == set(bad_resid):
                    print("Breaking early, h has converged")
                    break

                # Update W_inv to remove weighting on the worst residuals
                W_inv = updateW(W_INV_STATIC, bad_resid)

        # Reset everything for a final run for nice plots
        print("Overall best run number %g used" % (overAllBestRun))
        W_inv = updateW(W_INV_STATIC, optimal_resid_not_used)
        self.model_resid_not_used = optimal_resid_not_used
        self.Theta = copy.deepcopy(self.theta_orig)
        self.xs_est = copy.deepcopy(xs_meas)

        # Store the history of theta
        self.theta_history = []
        theta_vals = self.Theta['value']
        self.theta_history.append([np.array(theta_vals)])
        self.cost_history = []

        # Don't break early for this run
        breakEarly = False
        cost = self.optimise(data, W_inv, breakEarly, rpmStep)
        print("Final run complete, model residuals ignored:")
        print self.model_resid_not_used
        print("Final cost: %g. For reference the lowest cost on any run was %g" % (cost, lowestCostObserved))
        # Save final stuff
        self.saveQuit(nObs)

    # If we are done do a few tidy up calculations needed to save the results
    def saveQuit(self, nObs):
        nTheta = sum(self.Theta.loc[:,"estimate"])
        self.nTheta = nTheta

        # Make the correction for degrees of freedom as per hayes paper
        nSample = nObs*13 + (nObs - 1)*13
        nDOF = nObs*13 + self.nTheta
        correction = nSample/(nSample - nDOF - 1)*1.8


        # Evaulate the final uncertainty with "sandwich estimator"
        bread1 = np.dot(np.linalg.inv(np.dot(np.transpose(self.jacob_full), self.jacob_full)), np.transpose(self.jacob_full))
        bread2 = np.dot(self.jacob_full, np.linalg.inv(np.dot(np.transpose(self.jacob_full), self.jacob_full)))
        middle = np.diag(np.diag(np.dot(self.resid_full, np.transpose(self.resid_full))))
        var = correction*np.diag(np.dot(np.dot(bread1, middle), bread2))
        stddev = np.sqrt(np.squeeze(var[0:nTheta]))
        self.stddev = stddev
        print stddev
        # Tehta history needs to be numpy array for plotting
        self.theta_history = np.squeeze(np.array(self.theta_history))

    def getCost(self, W_inv):
        cost = sum(np.dot(np.dot(np.transpose(self.resid_full),W_inv),self.resid_full))
        return cost

    def get_new_resid_to_ignore(self, nObs, prop_to_remove, W_INV_STATIC):

        weighted_resid_squares = np.squeeze(self.resid_full)*np.squeeze(np.squeeze(np.diag(W_INV_STATIC))*np.squeeze(self.resid_full))
        weighted_resid_squares = weighted_resid_squares.reshape([nObs*2-1, 13])

        # get the columns for the model residuals
        weighted_model_resid_squares = np.sum(weighted_resid_squares[1::2, :],axis=1)

        # number to remove
        nRemove = int(round(nObs*prop_to_remove))
        temp = np.argpartition(-weighted_model_resid_squares, nRemove)
        largest_indices = temp[:nRemove]

        # h represents obsNo which are one higher than the model residual index (as the obsNo 0 does not have a model residual)
        h = largest_indices + 1
        return h

    def optimise(self, data, W_inv, breakEarly, rpmStep):
        # Prepare to run optimisation loop
        delta_cost = -1*float('inf')
        nObs = data.posvel_t.size
        nTheta = sum(self.Theta.loc[:,"estimate"])
        nIter = 1

        # Concatenate the required data into one array
        xs_meas =  np.concatenate((data.pos, data.vel, data.quat, data.omega), axis=1)

        # Preallocate matrix for the initial prediction
        xs_pred_start = np.zeros([len(data.obs_t)-1, 13])
        xs_pred_end = np.zeros([len(data.obs_t)-1, 13])
        resid_end = np.zeros([len(data.obs_t)-1, 13])
        resid_start = np.zeros([len(data.obs_t)-1, 13])
        ts_pred = data.obs_t[1:]
        nStates = 13

        # Run
        while (delta_cost < 0.01):

            # Preallocate the size of the jacobian and residual vector
            self.jacob_full = np.zeros([(nObs-1)*nStates*2+nStates, nTheta + len(data.obs_t)*nStates])
            self.resid_full = np.zeros([(nObs-1)*nStates*2+nStates, 1])


            # Loop through the observations
            for obsNo in range(0, nObs):

                # If this is the first observation there is only measurement error
                if obsNo == 0:
                    self.updateMatrix(xs_meas[obsNo, :], self.xs_est[obsNo, :], 0, obsNo, nTheta, 0, 0, self.Theta)
                    continue

                # Store the time and observation at the start of this step
                t_start = data.obs_t[obsNo-1]
                t_end = data.obs_t[obsNo]

                # Need to deepcopy the initial measurement
                x_pred = copy.deepcopy(self.xs_est[obsNo-1, :].reshape(13, 1))

                # Flag that this is the first iteration
                flag_first_iter = True

                # Loop through the rpms until we reach the next observation
                obs_ended = False
                while not obs_ended:

                    # If it is the first iteration we interpolate the RPM to make sure it lines up in time
                    if flag_first_iter:
                        rpm1 = np.interp(t_start, data.rpm_t, data.rpm[:, 0])
                        rpm2 = np.interp(t_start, data.rpm_t, data.rpm[:, 1])
                        rpm3 = np.interp(t_start, data.rpm_t, data.rpm[:, 2])
                        rpm4 = np.interp(t_start, data.rpm_t, data.rpm[:, 3])
                        rpm_now = np.vstack((rpm1, rpm2, rpm3, rpm4))

                        # Find the index for the next rpm time which is greater than the current time
                        next_rpm_t_ind = np.argmax(data.rpm_t > t_start)
                        dt = data.rpm_t[next_rpm_t_ind] - t_start
                        rpm_next = data.rpm[next_rpm_t_ind, :].reshape(4, 1)
                    else:
                        # Otherwise if we are not at the first step we just pull out the rpm
                        rpm_now = data.rpm[rpm_t_ind, :].reshape(4, 1)
                        next_rpm_t_ind = rpm_t_ind + rpmStep


                        # The amount of time to integrate over depends on whether we get to the end of the observation step
                        if data.rpm_t[next_rpm_t_ind]<=t_end:

                            # If we have not reachde the end of the timestep the change in time is
                            # simply the difference between rpm times
                            dt = data.rpm_t[next_rpm_t_ind] - data.rpm_t[rpm_t_ind]
                            rpm_next = data.rpm[next_rpm_t_ind, :].reshape(4, 1)
                        else:

                            # Otherwise we take the difference from t_end
                            dt = t_end - data.rpm_t[rpm_t_ind]

                            # After this we should go to the next observation
                            obs_ended = True

                            # Interpolate for the rpm at the end of observation
                            proportion = dt/(data.rpm_t[next_rpm_t_ind] - data.rpm_t[rpm_t_ind])
                            rpm_next = rpm_now + (data.rpm[next_rpm_t_ind, :].reshape(4, 1) - rpm_now)*proportion

                    # The current rpm t ind gets iterated through
                    rpm_t_ind = next_rpm_t_ind

                    # If the first iteration then we calculate only dtheta
                    if flag_first_iter:
                        dX_dtheta, thrusts = diffTheta(x_pred, rpm_now, self.Theta, dt)
                        F_k = diffX(x_pred, dt, thrusts, self.Theta)
                        flag_first_iter = False
                    else:
                        dnextX_dtheta, thrusts = diffTheta(x_pred, rpm_now, self.Theta, dt)
                        dnextX_dprevX = diffX(x_pred, dt, thrusts, self.Theta)
                        F_k = np.dot(dnextX_dprevX,F_k)
                        dX_dtheta = dnextX_dtheta + np.dot(dnextX_dprevX,dX_dtheta)

                    # Update the state estimate with Runge Kutta integration
                    k1, flagErr1 = quadDynamics(x_pred, self.Theta, rpm_now)
                    k2, flagErr2 = quadDynamics(x_pred+0.5*dt*k1, self.Theta, rpm_now + (rpm_next - rpm_now)*0.5);
                    k3, flagErr3 = quadDynamics(x_pred+0.5*dt*k2, self.Theta, rpm_now + (rpm_next - rpm_now)*0.5);
                    k4, flagErr4 = quadDynamics(x_pred+dt*k3, self.Theta, rpm_next);
                    x_pred = x_pred + dt/6*(k1+2*k2+2*k3+k4);

                    # If an error is encountered, exit this function
                    if flagErr1 == True or flagErr2 == True or flagErr3 == True or flagErr4 == True:
                        print("Thrust error encountered, exiting")
                        self.xs_est = copy.deepcopy(xs_meas)
                        self.Theta = copy.deepcopy(self.theta_orig)
                        return float("inf")

                # At this point we have integrated through how changes in theta will effect changes in the state.
                # For all of the vectors this corresponds straight through to changes in residuals.
                # BuIRLSItert the residual for the quaternion is another quaternion, so we need to modify to account for this.
                q_est = self.xs_est[obsNo, 6:10]
                dr_dTheta = correctTheta(dX_dtheta, q_est)
                model_resid = self.updateMatrix(xs_meas[obsNo, :], self.xs_est[obsNo, :], x_pred, obsNo, nTheta, dr_dTheta, F_k, self.Theta)

                # Add the x predictions to be stored
                if nIter == 1:
                    xs_pred_start[obsNo-1, :] = np.squeeze(x_pred)
                    resid_start[obsNo-1, :] = np.squeeze(model_resid)
                else:
                    xs_pred_end[obsNo-1, :] = np.squeeze(x_pred)
                    resid_end[obsNo-1, :] = np.squeeze(model_resid)

            # Perform the update of theta
            lhs = np.dot(np.dot(np.transpose(self.jacob_full), W_inv), self.jacob_full)
            rhs = np.dot(np.dot(np.transpose(self.jacob_full), W_inv), self.resid_full)

            dy = np.linalg.solve(lhs, rhs)
            dTheta = dy[0:nTheta]
            dX_est = dy[nTheta:]

            # Update the current estimate and cost
            self.updateEstimate(dTheta, dX_est, nStates, len(data.posvel_t))
            theta_vals = self.Theta['value']
            self.theta_history.append([np.array(theta_vals)])
            cost = sum(np.dot(np.dot(np.transpose(self.resid_full),W_inv),self.resid_full))
            self.cost_history.append([cost])

            if nIter>self.maxIter:
                break

            if nIter > 1:
                delta_cost = np.array(self.cost_history[nIter-1]) - np.array(self.cost_history[nIter-2])
                delta_cost = delta_cost/self.cost_history[nIter-2]
                print("Iteration {}, change in cost: {}".format(nIter, delta_cost))

                if abs(delta_cost) < self.tol and breakEarly:
                    print("Converged within tolerance")
                    break
            nIter = nIter + 1


        # Save the stuff that we want to save
        self.xs_meas = xs_meas
        self.obs_t = data.obs_t
        self.xs_pred_start = xs_pred_start
        self.xs_pred_end = xs_pred_end
        self.ts_pred = ts_pred
        self.resid_end = resid_end
        self.resid_start = resid_start
        return cost

    def updateMatrix(self, x_meas, x_est, x_pred, obsNo, nTheta, dr_dTheta, F_k, Theta):

        nStates = 13

        q0 = x_meas[6]
        q1 = x_meas[7]
        q2 = x_meas[8]
        q3 = x_meas[9]

        # The measurement error does not change wrt theta for most of the parameters
        #self.jacob_full[((obsNo)*26):(obsNo)*26+13, 0:nTheta] = np.zeros([13, nTheta])
        dr_meas_dTheta = np.zeros([13, nTheta])
        omegax = x_est[10]
        omegay = x_est[11]
        omegaz = x_est[12]

        # the exception is the tango location if it is included. these are put in the last columns of jacobian
        index = nTheta - 1
        if self.Theta['estimate']['tangoDz']:
            quat = Quaternion(x_est[6:10])

            # Rotate a vector
            dr_meas_dTheta[0:3, index] = -1*quat.rotate(np.array([0, 0, 1]))

            # For the velocity
            dvmeas_resid_dtangoZ_BF = -1*np.array([omegay, -omegax, 0])
            dvmeas_resid_dtangoZ_IF = quat.rotate(dvmeas_resid_dtangoZ_BF)
            dr_meas_dTheta[3:6, index] = dvmeas_resid_dtangoZ_IF

            index = index - 1

        if self.Theta['estimate']['tangoDy']:
            quat = Quaternion(x_est[6:10])

            # Rotate a vector
            dr_meas_dTheta[0:3, index] = -1*quat.rotate(np.array([0, 1, 0]))

            # For the velocity
            dvmeas_resid_dtangoY_BF = -1*np.array([-omegaz, 0, omegax])
            dvmeas_resid_dtangoY_IF = quat.rotate(dvmeas_resid_dtangoY_BF)
            dr_meas_dTheta[3:6, index] = dvmeas_resid_dtangoY_IF

            index = index - 1


        if self.Theta['estimate']['tangoDx']:
            e0 = x_est[6]
            e1 = x_est[7]
            e2 = x_est[8]
            e3 = x_est[9]

            # Rotate a vector
            dr_meas_dTheta[0:3, index] = -1*quat.rotate(np.array([1, 0, 0]))

            # For the velocity
            dvmeas_resid_dtangoX_BF = -1*np.array([0, omegaz, -omegay])
            dvmeas_resid_dtangoX_IF = quat.rotate(dvmeas_resid_dtangoX_BF)
            dr_meas_dTheta[3:6, index] = dvmeas_resid_dtangoX_IF

            index = index - 1


        e0 = x_est[6]
        e1 = x_est[7]
        e2 = x_est[8]
        e3 = x_est[9]
        m0 = x_meas[6]
        m1 = x_meas[7]
        m2 = x_meas[8]
        m3 = x_meas[9]
        theta = Theta['value']
        R0 = theta['R0']
        R1 = theta['R1']
        R2 = theta['R2']
        R3 = theta['R3']

        if self.Theta['estimate']['R3']:
            dr_meas_dTheta[6:10, index] = -1*np.array([-e3*m0-e2*m1+e1*m2+e0*m3, -e3*m1+e2*m0+e1*m3-e0*m2, -e3*m2+e2*m3-e1*m0+e0*m1, -e3*m3-e2*m2-e1*m1-e0*m0])
            index = index - 1

        if self.Theta['estimate']['R2']:
            dr_meas_dTheta[6:10, index] = -1*np.array([-m0*e2+m1*e3+m2*e0-m3*e1, -e2*m1-e3*m0+e0*m3+e1*m2, -e2*m2-e3*m3-e0*m0-e1*m1, -e2*m3+e3*m2-e0*m1+e1*m0])
            index = index - 1

        if self.Theta['estimate']['R1']:
            dr_meas_dTheta[6:10, index] = -1*np.array([-m0*e1+e0*m1-m2*e3+m3*e2, -e1*m1-e0*m0-e3*m3-e2*m2, -e1*m2-e0*m3+e3*m0+e2*m1, -e1*m3+e0*m2+e3*m1-e2*m0])
            index = index - 1

        if self.Theta['estimate']['R0']:
            dr_meas_dTheta[6:10, index] = -1*np.array([m0*e0+m1*e1+m2*e2+m3*e3, e0*m1-e1*m0+e2*m3-e3*m2, e0*m2-e1*m3-e2*m0+e3*m1, e0*m3+e1*m2-e2*m1-e3*m0])

        # put the stuff in the matrix
        self.jacob_full[((obsNo)*nStates*2):(obsNo)*nStates*2+nStates, 0:nTheta] = dr_meas_dTheta[0:nStates, :]

        # The measurement error decreases with increasing state. This is the derivative of measurement error wrt estimate
        temp = np.diag(np.ones(13))*-1
        temp[6, 6:10] = np.array([R0*m0+R1*m1+R2*m2+R3*m3, -R1*m0+R0*m1+R3*m2-R2*m3, -R2*m0-R3*m1+R0*m2+R1*m3, -R3*m0+R2*m1-R1*m2+R0*m3])
        temp[7, 6:10] = np.array([R0*m1-R1*m0+R2*m3-R3*m2, -R1*m1-R0*m0+R3*m3+R2*m2, -R2*m1+R3*m0+R0*m3-R1*m2, -R3*m1-R2*m0-R1*m3-R0*m2])
        temp[8, 6:10] = np.array([R0*m2-R1*m3-R2*m0+R3*m1, -R1*m2-R0*m3-R3*m0-R2*m1, -R2*m2+R3*m3-R0*m0+R1*m1, -R3*m2-R2*m3+R1*m0+R0*m1])
        temp[9, 6:10] = np.array([R0*m3+R1*m2-R2*m1-R3*m0, -R1*m3+R0*m2-R3*m1+R2*m0, -R2*m3-R3*m2-R0*m1-R1*m0, -R3*m3+R2*m2+R1*m1-R0*m0])
        self.jacob_full[((obsNo)*nStates*2):(obsNo)*nStates*2+nStates, (nTheta+nStates*(obsNo)):nTheta+nStates*(obsNo+1)] = temp[0:nStates, 0:nStates]

        # First thing is to calculated the expected measurement
        tangoDx = theta['tangoDx']
        tangoDy = theta['tangoDy']
        tangoDz = theta['tangoDz']
        cgToTangoBF = np.array([tangoDx, tangoDy, tangoDz])
        quat = Quaternion(x_est[6:10])

        # Rotate the vector from cgToTango into the global frame
        cgToTangoIF = quat.rotate(cgToTangoBF)

        # The measurement is of tangos position, so the expected measurement is where tango is expecetd to be (cg_est + cgToTangoIF)
        expectedPosMeas = x_est[0:3] + cgToTangoIF

        # Retrieve the angular velocity
        omega_est = x_est[10:13]

        # Get the velocity of the tango frame wrt the cg
        vRel_BF = np.cross(omega_est, cgToTangoBF)

        # Rotate the relative velocity into the global frame
        relV_IF = quat.rotate(vRel_BF)

        # The expected velocity measurement is cg plus relative velocity of tango
        expectedVelMeas = x_est[3:6] + relV_IF

        # Start putting stuff in the residual
        meas_resid = np.zeros([13,1])
        meas_resid[0:3] = (x_meas[0:3] - expectedPosMeas).reshape(3,1)
        meas_resid[3:6] = (x_meas[3:6] - expectedVelMeas).reshape(3,1)
        meas_resid[10:13] = (x_meas[10:13] - x_est[10:13]).reshape(3,1)

        # Log of the error quat
        qR = Quaternion([R0, R1, R2, R3])
        q_meas = Quaternion(x_meas[6:10])
        q_est = Quaternion(x_est[6:10])
        qR_qest = qR*q_est
        meas_error_quat = q_meas*qR_qest.inverse
        meas_error_quat = meas_error_quat.elements
        meas_error_quat[0] = 0
        meas_resid[6:10] = meas_error_quat.reshape(4,1)

        # Put in the larger vector
        self.resid_full[obsNo*nStates*2:(obsNo*nStates*2+nStates)] = meas_resid[0:nStates].reshape([nStates,1])

        # If the observation is greater than zero we also do the model error
        if obsNo > 0:
            self.jacob_full[((obsNo-1)*nStates*2 + nStates):(obsNo)*nStates*2, 0:nTheta] = dr_dTheta[0:nStates, :]

            # Increasing the prediction decreases the residual so negate
            temp_deriv = -1*F_k

            qe0 = x_est[6]
            qe1 = x_est[7]
            qe2 = x_est[8]
            qe3 = x_est[9]

            # The quaternion part is
            array = np.array([[qe0,qe1,qe2,qe3],[qe1, -qe0, qe3, -qe2],[qe2, -qe3, -qe0, qe1],[qe3,qe2,-qe1,-qe0]])
            temp_deriv[6:10, 6:10] = np.dot(array, F_k[6:10, 6:10])
            temp_deriv = (temp_deriv)
            self.jacob_full[((obsNo-1)*nStates*2 + nStates):(obsNo)*nStates*2, (nTheta+nStates*(obsNo-1)):(nTheta+nStates*(obsNo))] = temp_deriv[0:nStates, 0:nStates]

            # Increasing the estimate at this step increases the residual
            temp2 = np.diag(np.ones(13))
            e0 = x_pred[6]
            e1 = x_pred[7]
            e2 = x_pred[8]
            e3 = x_pred[9]
            temp2[6:10, 6:10] = np.squeeze(np.array([[e0,e1,e2,e3],[-e1, e0, -e3, e2],[-e2, e3, e0, -e1],[-e3, -e2, e1, e0]]))
            self.jacob_full[((obsNo-1)*nStates*2 + nStates):(obsNo)*nStates*2, (nTheta+nStates*(obsNo)):(nTheta+nStates*(obsNo+1))] = temp2[0:nStates, 0:nStates]

            # Also add the model residual
            model_resid = x_est - x_pred.reshape([1, 13])
            model_resid = np.squeeze(model_resid)
            q_est = Quaternion(x_est[6:10])
            q_pred = Quaternion(x_pred[6:10])
            try:
                q_pred_inv = q_pred.inverse
            except ZeroDivisionError:
                q_pred_inv = Quaternion([1,0,0,0])
                print('Warning: Quaternion could not be inverted')
            model_error_quat = q_est*q_pred_inv
            model_error_quat = model_error_quat.elements
            model_error_quat[0] = 0
            model_resid[6:10] = model_error_quat
            ##### If estimating thrust coefs have to switch off these guys
            if Theta['estimate']['k1'] or Theta['estimate']['k2'] :
                model_resid[6:] = model_resid[6:]*0

            self.resid_full[(obsNo*nStates*2-nStates):(obsNo*nStates*2)] = model_resid[0:nStates].reshape([nStates,1])
            return model_resid



    def updateEstimate(self, dTheta, dX_est, nStates, nObs):

        # Update state estiamte
        dX_est = dX_est.reshape([nObs, nStates])

        # Add update to postion and velocity
        self.xs_est[:, 0:6] = self.xs_est[:, 0:6] - self.step*dX_est[:, 0:6]

        if nStates == 13:
            self.xs_est[:, 10:13] = self.xs_est[:, 10:13] - self.step*dX_est[:, 10:13]

        if nStates == 13:
            # Update the quaternions by rotating them appropriately
            rows, cols = self.xs_est.shape
            for i in range(0, rows):
                quatUpdate = dX_est[i, 6:10]
                newQuat = self.xs_est[i, 6:10] - quatUpdate*self.step
                newQuat = newQuat/np.linalg.norm(newQuat)
                self.xs_est[i, 6:10] = newQuat

        # Index in dTheta to pull out of
        index = 0

        if self.Theta['estimate']['m']:
             self.Theta.set_value('m', 'value', self.Theta['value']['m'] + self.step*dTheta[index])
             index = index + 1

        if self.Theta['estimate']['Ixx']:
             self.Theta.set_value('Ixx', 'value', self.Theta['value']['Ixx'] + self.step*dTheta[index])
             index = index + 1

        if self.Theta['estimate']['Iyy']:
            self.Theta.set_value('Iyy', 'value', self.Theta['value']['Iyy'] + self.step*dTheta[index])
            index = index + 1

        if self.Theta['estimate']['Izz']:
            self.Theta.set_value('Izz', 'value', self.Theta['value']['Izz'] + self.step*dTheta[index])
            index = index + 1

        if self.Theta['estimate']['deltaDx']:
            self.Theta.set_value('deltaDx', 'value', self.Theta['value']['deltaDx'] + self.step*dTheta[index])
            index = index + 1

        if self.Theta['estimate']['deltaDy']:
             self.Theta.set_value('deltaDy', 'value', self.Theta['value']['deltaDy'] + self.step*dTheta[index])
             index = index + 1
#
        if self.Theta['estimate']['Cq']:
             self.Theta.set_value('Cq', 'value', self.Theta['value']['Cq'] + self.step*dTheta[index])
             index = index + 1

        if self.Theta['estimate']['k1']:
            self.Theta.set_value('k1', 'value', self.Theta['value']['k1'] + self.step*dTheta[index])
            index = index + 1

        if self.Theta['estimate']['k2']:
            self.Theta.set_value('k2', 'value', self.Theta['value']['k2'] + self.step*dTheta[index])
            index = index + 1

        if self.Theta['estimate']['k4']:
            self.Theta.set_value('k4', 'value', self.Theta['value']['k4'] + self.step*dTheta[index])
            index = index + 1

        if self.Theta['estimate']['Cdh']:
            self.Theta.set_value('Cdh', 'value', self.Theta['value']['Cdh'] + self.step*dTheta[index])
            index = index + 1

        if self.Theta['estimate']['Cdxy']:
            self.Theta.set_value('Cdxy', 'value', self.Theta['value']['Cdxy'] + self.step*dTheta[index])
            index = index + 1

        if self.Theta['estimate']['Cdy']:
            self.Theta.set_value('Cdy', 'value', self.Theta['value']['Cdy'] + self.step*dTheta[index])
            index = index + 1

        if self.Theta['estimate']['Cdz']:
             self.Theta.set_value('Cdz', 'value', self.Theta['value']['Cdz'] + self.step*dTheta[index])
             index = index + 1

        if self.Theta['estimate']['R0']:
             self.Theta.set_value('R0', 'value', self.Theta['value']['R0'] + self.step*dTheta[index])
             index = index + 1

        if self.Theta['estimate']['R1']:
             self.Theta.set_value('R1', 'value', self.Theta['value']['R1'] + self.step*dTheta[index])
             index = index + 1

        if self.Theta['estimate']['R2']:
             self.Theta.set_value('R2', 'value', self.Theta['value']['R2'] + self.step*dTheta[index])
             index = index + 1

        if self.Theta['estimate']['R3']:
             self.Theta.set_value('R3', 'value', self.Theta['value']['R3'] + self.step*dTheta[index])
             index = index + 1

        if self.Theta['estimate']['tangoDx']:
             self.Theta.set_value('tangoDx', 'value', self.Theta['value']['tangoDx'] - self.step*dTheta[index])
             index = index + 1

        if self.Theta['estimate']['tangoDy']:
             self.Theta.set_value('tangoDy', 'value', self.Theta['value']['tangoDy'] - self.step*dTheta[index])
             index = index + 1

        if self.Theta['estimate']['tangoDz']:
             self.Theta.set_value('tangoDz', 'value', self.Theta['value']['tangoDz'] - self.step*dTheta[index])
             index = index + 1

        # Need to renormalise the sensor rotation.
        q = np.array([self.Theta['value']['R0'], self.Theta['value']['R1'], self.Theta['value']['R2'], self.Theta['value']['R3']])
        q = q/np.linalg.norm(q)
        self.Theta.set_value('R0', 'value', q[0])
        self.Theta.set_value('R1', 'value', q[1])
        self.Theta.set_value('R2', 'value', q[2])
        self.Theta.set_value('R3', 'value', q[3])

def updateW(W_INV_STATIC, h):
    W_inv = copy.deepcopy(W_INV_STATIC)
    for this_h in h:
        start = this_h*26-13
        end = start+13
        W_inv[start:end, start:end] = np.zeros([13,13])
    return W_inv

def getW(Q, Theta, covSens, ts, data):

    # Allocate
    nObs = ts.size
    W = np.zeros([(nObs-1)*13*2+13,(nObs-1)*13*2+13])
    xs_obs =  np.concatenate((data.pos, data.vel, data.quat, data.omega), axis=1)

    W[0:13, 0:13] = covSens
    rpmInterval = 3


    # Loop through the observations
    for obsNo in range(0, nObs-1):

        # Store the time and observation at the start of this step
        t_start = data.obs_t[obsNo]
        t_end = data.obs_t[obsNo+1]
        t_this_obs = t_end - t_start
        x = xs_obs[obsNo, :].reshape(13, 1)


        Q_temp = copy.deepcopy(Q)
        x1 = xs_obs[obsNo, 10:]
        x2 = xs_obs[obsNo+1, 10:]
        ang_accel = (x2 - x1)/t_this_obs
        alphax = ang_accel[0]
        alphay = ang_accel[1]
        alphaz = ang_accel[2]


        # Flag that this is the first iteration
        flag_first_iter = True

        # Loop through the rpms until we reach the next observation
        obs_ended = False
        while not obs_ended:

            # If it is the first iteration we interpolate the RPM to make sure it lines up in time
            if flag_first_iter:
                rpm1 = np.interp(t_start, data.rpm_t, data.rpm[:, 0])
                rpm2 = np.interp(t_start, data.rpm_t, data.rpm[:, 1])
                rpm3 = np.interp(t_start, data.rpm_t, data.rpm[:, 2])
                rpm4 = np.interp(t_start, data.rpm_t, data.rpm[:, 3])
                rpm_now = np.vstack((rpm1, rpm2, rpm3, rpm4))

                # Find the index for the next rpm time which is greater than the current time
                next_rpm_t_ind = np.argmax(data.rpm_t > t_start)
                dt = data.rpm_t[next_rpm_t_ind] - t_start


            else:
                # Otherwise if we are not at the first step we just pull out the rpm
                rpm_now = data.rpm[rpm_t_ind, :].reshape(4, 1)
                next_rpm_t_ind = rpm_t_ind + rpmInterval

                # The amount of time to integrate over depends on whether we get to the end of the observation step
                if data.rpm_t[next_rpm_t_ind]<=t_end:

                    # If we have not reachde the end of the timestep the change in time is
                    # simply the difference between rpm times
                    dt = data.rpm_t[next_rpm_t_ind] - data.rpm_t[rpm_t_ind]
                else:

                    # Otherwise we take the difference from t_end
                    dt = t_end - data.rpm_t[rpm_t_ind]

                    # After this we should go to the next observation
                    obs_ended = True

            # The current rpm t ind gets iterated through
            rpm_t_ind = next_rpm_t_ind



            # If the first iteration then we calculate only dtheta
            if flag_first_iter:
                P = np.zeros([13, 13])
                flag_first_iter = False
            else:
                F_k = diffX(x, dt, np.array([0,0,0,0]), Theta)
                G = diffW(Theta,dt)
                P= np.dot(np.dot(F_k, P), np.transpose(F_k)) + np.dot(np.dot(G, Q_temp), np.transpose(G))

        # Return the matrix
        start = (obsNo)*26+13
        mid = (obsNo+1)*26
        end = (obsNo+1)*26 + 13
        P[0:3, 0:3] = P[0:3, 0:3]/t_this_obs*10
        P[3:6, 3:6] = P[3:6, 3:6]
        P[6:10, 6:10] = P[6:10, 6:10]/t_this_obs*100

        W[start:mid, start:mid] = P
        W[mid:end, mid:end] = covSens
    return W

def diffW(Theta, dt):

    # Jacobian is 13x6. Rows are each state, columns are each noise param.
    Jacobian = np.zeros([13, 6])
    theta = Theta['value']
    m   = theta['m']
    Ixx = theta['Ixx']
    Iyy = theta['Iyy']
    Izz = theta['Izz']

    # I have done this to put equal weightings on all the axes
    Ixx = (Ixx + Iyy + Izz)/3
    Iyy = Ixx
    Izz = Ixx

    Jacobian[3,0] = dt/m
    Jacobian[4,1] = dt/m
    Jacobian[5,2] = dt/m

    Jacobian[10, 3] = dt/Ixx
    Jacobian[11, 4] = dt/Iyy
    Jacobian[12, 5] = dt/Izz
    return Jacobian

def diffW_onestep(Theta, dt):

    # Jacobian is 13x6. Rows are each state, columns are each noise param.
    Jacobian = np.zeros([13, 6])
    theta = copy.deepcopy(Theta['value'])
    m   = theta['m']
    Ixx = theta['Ixx']
    Iyy = theta['Iyy']
    Izz = theta['Izz']

#    This jacobian represents the change in the states with respect to each of
#    the noise parameters. The rough approximation is that first of all the dynamics
#    over the timestep can just approximated as second order as:
#    dx2 = dx1 + dx1_dot*dt + 1/2*dx1_dot_dot*dt^2
#    Then we are taking dx2/dw where w is the noise.
#    In the case of the position, the second derivative is the accel. the change in accel
#    wrt a force is simply 1/m (dx1_dot_dot). The angular stuff is analogous
    Jacobian[0,0] = dt**2/(2*m)
    Jacobian[1,1] = dt**2/(2*m)
    Jacobian[2,2] = dt**2/(2*m)
    Jacobian[3,0] = dt/m
    Jacobian[4,1] = dt/m
    Jacobian[5,2] = dt/m

#    For the angular stuff, ignoring the couple terms, then omega_i_dot = torque/I_i
#    So then omega_i_dot/dw = 1/I_i. This is fair.
#    For the quaternion, we need to know how the moment noise propgates to a quaternion
#    second derivative. However, this would be arduous to calculate. So Ive assumed that
#    it is the same as how it would propagate to an angle  in radians second derivative,
#    as quaternions are similar order of magnitude as radians. Not strictly correct,
#    but should be valid for order of magnitude. This is the only hackey math in all of this code
    Jacobian[6:10, 3] = np.array([dt**2/(2*Ixx), dt**2/(2*Ixx), 0, 0])
    Jacobian[6:10, 4] = np.array([dt**2/(2*Iyy), 0, dt**2/(2*Iyy), 0])
    Jacobian[6:10, 5] = np.array([dt**2/(2*Izz), 0, 0, dt**2/(2*Izz)])
    Jacobian[10, 3] = dt/Ixx
    Jacobian[11, 4] = dt/Iyy
    Jacobian[12, 5] = dt/Izz
    return Jacobian

def diffTheta(x, rpm_now, Theta, dt):

    quat = Quaternion(x[6:10])
    quat = quat.normalised
    try:
        quat_inv = quat.inverse
    except ZeroDivisionError:
        quat_inv = Quaternion([1,0,0,0])
        quat = Quaternion([1,0,0,0])
        print('Warning: Quaternion could not be inverted')
    vel = x[3:6]

    theta = Theta['value']
    m   = theta['m']
    Ixx = theta['Ixx']
    Iyy = theta['Iyy']
    Izz = theta['Izz']
    Cq = theta['Cq']
    k1 = theta['k1']
    k2 = theta['k2']
    k4 = theta['k4']
    Cdh = theta['Cdh']
    Cdxy = theta['Cdxy']
    Cdy = theta['Cdy']
    Cdz = theta['Cdz']
    deltaDx = theta['deltaDx']
    deltaDy = theta['deltaDy']
    Dx = theta['Dx']
    Dy = theta['Dy']

    thrusts, flagErr = getThrust(x, Theta, rpm_now)

    # Start with an empty array
    jacobian = []

    # for more explanations of this part of the code see the matlab
    if Theta['estimate']['m']:

        # Thrust if IF
        thrust_BF = np.array([0, 0, -sum(thrusts)]).reshape(3, 1)
        thrust_IF = quat.rotate(thrust_BF)

        # Drag in IF
        vel_BF = quat_inv.rotate(vel)
        airframeDrag_BF = -np.array([Cdxy, Cdxy, Cdz])*vel_BF*np.absolute(vel_BF)
        airframeDrag_IF = quat.rotate(airframeDrag_BF)

        # dvel/dt = (thrust + drag + mg)/m, so d/dm(dvel/dt) = -(thrust + drag)/m^2
        temp = -(thrust_IF + airframeDrag_IF)/m**2;

        # return the derivative
        jacob_mass = np.zeros([13, 1])
        jacob_mass[3:6] = temp.reshape(3, 1)*dt
        jacobian.append(jacob_mass)


    omegax = x[10]
    omegay = x[11]
    omegaz = x[12]
    Tx =-thrusts[0]*(Dy-deltaDy)+thrusts[1]*(Dy+deltaDy)+thrusts[2]*(Dy+deltaDy)-thrusts[3]*(Dy-deltaDy)
    Ty = thrusts[0]*(Dx-deltaDx)-thrusts[1]*(Dx+deltaDx)+thrusts[2]*(Dx-deltaDx)-thrusts[3]*(Dx+deltaDx)
    Tz = Cq*rpm_now[0]**2+Cq*rpm_now[1]**2-Cq*rpm_now[2]**2-Cq*rpm_now[3]**2

    if Theta['estimate']['Ixx']:
        dalphax = -Tx/Ixx**2 + 1/Ixx**2*(omegay*omegaz*Izz - omegaz*omegay*Iyy)
        dalphay = -omegax*omegaz/Iyy
        dalphaz = omegay*omegax/Izz
        jacob_Ixx = np.zeros([13, 1])
        jacob_Ixx[10:13] = np.array([dalphax, dalphay, dalphaz]).reshape(3,1)*dt
        jacobian.append(jacob_Ixx)

    if Theta['estimate']['Iyy']:
        dalphax = omegaz*omegay/Ixx;
        dalphay = -Ty/Iyy**2 + 1/Iyy**2*(omegaz*omegax*Ixx - omegax*omegaz*Izz);
        dalphaz = -omegay*omegax/Izz;
        jacob_Iyy = np.zeros([13, 1])
        jacob_Iyy[10:13] = np.array([dalphax, dalphay, dalphaz]).reshape(3,1)*dt
        jacobian.append(jacob_Iyy)

    if Theta['estimate']['Izz']:
        dalphax = -omegay*omegaz/Ixx;
        dalphay = omegax*omegaz/Iyy;
        dalphaz = -Tz/Izz**2 + 1/Izz**2*(omegax*omegay*Iyy - omegay*omegax*Ixx);
        jacob_Izz = np.zeros([13, 1])
        jacob_Izz[10:13] = np.array([dalphax, dalphay, dalphaz]).reshape(3,1)*dt
        jacobian.append(jacob_Izz)

    if Theta['estimate']['deltaDx']:
        derivative = 1/Iyy *-sum(thrusts)
        jacob_deltaDx = np.zeros([13, 1])
        jacob_deltaDx[11] = derivative*dt
        jacobian.append(jacob_deltaDx)

    if Theta['estimate']['deltaDy']:
        derivative = 1/Ixx*sum(thrusts)
        jacob_deltaDy = np.zeros([13, 1])
        jacob_deltaDy[10] = derivative*dt
        jacobian.append(jacob_deltaDy)

    if Theta['estimate']['Cq']:
        derivative = (rpm_now[0]**2+rpm_now[1]**2-rpm_now[2]**2-rpm_now[3]**2)/Izz
        jacob_Cq = np.zeros([13, 1])
        jacob_Cq[12] = derivative*dt
        jacobian.append(jacob_Cq)

    if Theta['estimate']['k1']:
        k1_plus = k1*1.0001
        k1_minus = k1*0.9999
        Theta_temp = copy.deepcopy(Theta)
        Theta_temp.set_value('k1', 'value', k1_plus)
        xDot_plus, flagErr = quadDynamics(x, Theta_temp, rpm_now)
        Theta_temp.set_value('k1', 'value', k1_minus)
        xDot_minus, flagErr = quadDynamics(x, Theta_temp, rpm_now)
        jacob_k1 = (xDot_plus-xDot_minus)/(k1_plus-k1_minus)*dt
        jacobian.append(jacob_k1)

    if Theta['estimate']['k2']:

        k2_plus = k2*1.0001
        k2_minus = k2*0.9999
        Theta_temp = copy.deepcopy(Theta)
        Theta_temp.set_value('k2', 'value', k2_plus)
        xDot_plus, flagErr = quadDynamics(x, Theta_temp, rpm_now)
        Theta_temp.set_value('k2', 'value', k2_minus)
        xDot_minus, flagErr = quadDynamics(x, Theta_temp, rpm_now)
        jacob_k2 = (xDot_plus-xDot_minus)/(k2_plus-k2_minus)*dt
        jacobian.append(jacob_k2)

    if Theta['estimate']['k4']:
        k4_plus = k4*1.0001
        k4_minus = k4*0.9999
        Theta_temp = copy.deepcopy(Theta)
        Theta_temp.set_value('k4', 'value', k4_plus)
        xDot_plus, flagErr = quadDynamics(x, Theta_temp, rpm_now)
        Theta_temp.set_value('k4', 'value', k4_minus)
        xDot_minus, flagErr = quadDynamics(x, Theta_temp, rpm_now)
        jacob_k4 = (xDot_plus-xDot_minus)/(k4_plus-k4_minus)*dt*10
        jacobian.append(jacob_k4)

    if Theta['estimate']['Cdh']:
        vel_BF = quat_inv.rotate(vel)

        # Derivative wrt Cdh in the body frame
        thrust = abs(sum(thrusts))
        dragDerivative_BF = -thrust*vel_BF[0:2]/m
        dragDerivative_BF = np.append(dragDerivative_BF, 0)

        # Need this in the body frame for the jacobian
        dragDerivative_IF = quat.rotate(dragDerivative_BF)
        jacob_Cdh = np.zeros([13, 1])
        jacob_Cdh[3:6] = dragDerivative_IF.reshape(3,1)*dt
        jacobian.append(jacob_Cdh)


    if Theta['estimate']['Cdxy']:
        vel_BF = quat_inv.rotate(vel)

        # Derivative wrt Cdx in the body frame
        dragDerivative_BF = np.array([-abs(vel_BF[0])*vel_BF[0]/m, -abs(vel_BF[1])*vel_BF[1]/m, 0])

        # Need this in the body frame for the jacobian
        dragDerivative_IF = quat.rotate(dragDerivative_BF)
        jacob_Cdxy = np.zeros([13, 1])
        jacob_Cdxy[3:6] = dragDerivative_IF.reshape(3,1)*dt
        jacobian.append(jacob_Cdxy)

    if Theta['estimate']['Cdy']:
        vel_BF = quat_inv.rotate(vel)

        # Derivative wrt Cdx in the body frame
        dragDerivative_BF = np.array([0, -abs(vel_BF[1])*vel_BF[1]/m, 0])

        # Need this in the body frame for the jacobian
        dragDerivative_IF = quat.rotate(dragDerivative_BF)
        jacob_Cdy = np.zeros([13, 1])
        jacob_Cdy[3:6] = dragDerivative_IF.reshape(3,1)*dt
        jacobian.append(jacob_Cdy)

    if Theta['estimate']['Cdz']:
        vel_BF = quat_inv.rotate(vel)

        # Derivative wrt Cdx in the body frame
        dragDerivative_BF = np.array([0, 0, -abs(vel_BF[2])*vel_BF[2]/m])

        # Need this in the body frame for the jacobian
        dragDerivative_IF = quat.rotate(dragDerivative_BF)
        jacob_Cdz = np.zeros([13, 1])
        jacob_Cdz[3:6] = dragDerivative_IF.reshape(3,1)*dt
        jacobian.append(jacob_Cdz)

    if Theta['estimate']['R0']:

        # The model residual does not change with respect to tango, because that is part of the MEASUREMENT Model.
        jacob_R0 = np.zeros([13, 1])
        jacobian.append(jacob_R0)

    if Theta['estimate']['R1']:

        # The model residual does not change with respect to tango, because that is part of the MEASUREMENT Model.
        jacob_R1 = np.zeros([13, 1])
        jacobian.append(jacob_R1)

    if Theta['estimate']['R2']:

        # The model residual does not change with respect to tango, because that is part of the MEASUREMENT Model.
        jacob_R2 = np.zeros([13, 1])
        jacobian.append(jacob_R2)

    if Theta['estimate']['R3']:

        # The model residual does not change with respect to tango, because that is part of the MEASUREMENT Model.
        jacob_R3 = np.zeros([13, 1])
        jacobian.append(jacob_R3)

    if Theta['estimate']['tangoDx']:

        # The model residual does not change with respect to tango, because that is part of the MEASUREMENT Model.
        jacob_tangoDx = np.zeros([13, 1])
        jacobian.append(jacob_tangoDx)

    if Theta['estimate']['tangoDy']:

        # The model residual does not change with respect to tango, because that is part of the MEASUREMENT Model.
        jacob_tangoDy = np.zeros([13, 1])
        jacobian.append(jacob_tangoDy)

    if Theta['estimate']['tangoDz']:

        # The model residual does not change with respect to tango, because that is part of the MEASUREMENT Model.
        jacob_tangoDy = np.zeros([13, 1])
        jacobian.append(jacob_tangoDy)

    jacobian = np.array(jacobian)
    jacobian = np.transpose(np.squeeze(jacobian))
    return jacobian, thrusts


def diffX(x, dt, thrusts, Theta):
    temp = np.zeros([13, 13])
    q0 = x[6]
    q1 = x[7]
    q2 = x[8]
    q3 = x[9]
    wx = x[10]
    wy = x[11]
    wz = x[12]


    # Velocity changes position dot
    temp[0,3] = 1
    temp[1,4] = 1
    temp[2,5] = 1

    # Acceleration is changed by the quaternion
    mass = Theta['value']['m']
    temp[3, 6:10] = np.squeeze(-1*sum(thrusts)/mass*np.array([2*q2, 2*q3, 2*q0, 2*q1]))
    temp[4, 6:10] = np.squeeze(-1*sum(thrusts)/mass*np.array([-2*q1, -2*q0, 2*q3, 2*q2]))
    temp[5, 6:10] = np.squeeze(-1*sum(thrusts)/mass*np.array([2*q0, -2*q1, -2*q2, 2*q3]))

    # qdot effected both by q and omega
    temp[6, 10:13] = np.array([-0.5*q1, -0.5*q2, -0.5*q3]).reshape(3,)
    temp[7, 10:13] = np.array([0.5*q0, -0.5*q3, 0.5*q2]).reshape(3,)
    temp[8, 10:13] = np.array([0.5*q3, 0.5*q0, -0.5*q1]).reshape(3,)
    temp[9, 10:13] = np.array([-0.5*q2, 0.5*q1, 0.5*q0]).reshape(3,)
    temp[6, 6:10] = np.array([0, -0.5*wx, -0.5*wy, -0.5*wz]).reshape(4,)
    temp[7, 6:10] = np.array([0.5*wx, 0, 0.5*wz, -0.5*wy]).reshape(4,)
    temp[8, 6:10] = np.array([0.5*wy, -0.5*wz, 0, 0.5*wx]).reshape(4,)
    temp[9, 6:10] = np.array([0.5*wz, 0.5*wy, -0.5*wx, 0]).reshape(4,)

    # Return jacobian
    result = np.identity(13) + temp*dt;
    return result

def correctTheta(dX_dtheta, q_est):
    dqp_dTheta = dX_dtheta[6:10, :]

    dqe_dqp = np.array([[0, 0, 0, 0],
                        [-q_est[1], q_est[0], -q_est[3], q_est[2]],
                        [-q_est[2], q_est[3], q_est[0], -q_est[1]],
                        [-q_est[3], -q_est[2], q_est[1], q_est[0]]])

    # Then update F so it is now dqe/dTheta
    dqe_dTheta = np.dot(dqe_dqp, dqp_dTheta)

    # Now update the Jacobian to the change in the residual wrt to theta
    dr_dTheta = dX_dtheta
    dr_dTheta[6:10, :] = dqe_dTheta
    return dr_dTheta

class DataStore:

    def __init__(self, data):
        self.posvel_t = data.posvel_t
        self.pos = data.pos
        self.vel = data.vel
        self.quat = data.quat
        self.omega = data.omega
        self.att_raw_t = data.att_raw_t
        self.omega_raw = data.omega_raw
        self.directory = data.directory
        self.rpm_t = data.rpm_t
        self.rpm = data.rpm
        self.rpm_raw = data.rpm_raw

class EstimatorStore:
    def __init__(self, estimator):
        self.xs_meas = copy.deepcopy(estimator.xs_meas)
        self.obs_t = copy.deepcopy(estimator.obs_t)
        self.xs_pred_start = copy.deepcopy(estimator.xs_pred_start)
        self.xs_pred_end = copy.deepcopy(estimator.xs_pred_end)
        self.ts_pred = copy.deepcopy(estimator.ts_pred)
        self.resid_end = copy.deepcopy(estimator.resid_end)
        self.resid_start = copy.deepcopy(estimator.resid_start)
        self.theta_history =  copy.deepcopy(estimator.theta_history)
        self.stddev = copy.deepcopy(estimator.stddev)
        self.xs_est = copy.deepcopy(estimator.xs_est)
        self.cost_history = copy.deepcopy(estimator.cost_history)
        self.Theta = copy.deepcopy(estimator.Theta)
        self.method = copy.deepcopy(estimator.method)
        if hasattr(estimator, 'model_resid_not_used'):
            self.model_resid_not_used = copy.deepcopy(estimator.model_resid_not_used)
