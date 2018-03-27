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
__email__ = "benjamin.morrell@sydney.edu.au"

import numpy as np


class data_track(object):


    def __init__(self,keys):
        """
        Utility class for tracking data in optimisation steps
        """
        self.c_leg_poly = dict()
        self.cost_grad = dict()
        for key in keys:
            self.c_leg_poly[key] = np.array([])
            self.cost_grad[key] = np.array([])

        self.cost = np.array([])
        self.step_coeff = np.array([]) # backtracking step coefficient
        self.inner_iter = np.array([])
        self.optimise_time = 0.0
        self.outer_opt_time = 0.0
        self.iterations = 0

    def update_data_track(self,c_leg_poly,cost,cost_grad):
        """
        Update the data to track through the iterations

        Args:
            Data to be stored:
                c_leg_poly: optimisation coefficients
                cost: the cost
                cost_grad: cost gradient

        Modifies:
            self.
            c_leg_poly: dictionary of the optimisation coefficients (For each dimension)
                        Stack up columns for each iteration
            cost_grad: vectors of the gradient of the cost function, stacked for each dimension
            cost: vector with the cost for each iteration

        """

        # Loop for each dimension
        for key in c_leg_poly.keys():
            # Total number of coefficients for a dimension (capturing all segments)
            n_coeff_dim = c_leg_poly[key].shape[0]

            # Different case if the first data point to track
            if self.c_leg_poly[key].size == 0:
                # Just initialise with the current input
                self.c_leg_poly[key] = c_leg_poly[key].reshape([n_coeff_dim,1])
                self.cost_grad[key] = cost_grad[key].reshape([n_coeff_dim,1])
            else:
                # Stack the values
                self.c_leg_poly[key] = np.concatenate([self.c_leg_poly[key],c_leg_poly[key].reshape([n_coeff_dim,1])],1)
                self.cost_grad[key] = np.concatenate([self.cost_grad[key],cost_grad[key].reshape([n_coeff_dim,1])],1)

        # Append the cost onto the vector
        self.cost = np.append(self.cost,cost)


    def update_inner_data(self,inner_iter,step_coeff):
        """
        Update the data from the inner loop to track through the iterations

        Args:
            Data to be stored:
                inner_iter: number of iterations in the inner loop (linesearch)
                step_coeff: final step taken in line search

        Modifies:
            Stored as vectors for each iteration:
            self.
            inner_iter: number of iterations in the inner loop (linesearch)
            step_coeff: final step taken in line search

        """
        self.step_coeff = np.append(self.step_coeff,step_coeff)
        self.inner_iter = np.append(self.inner_iter,inner_iter)
