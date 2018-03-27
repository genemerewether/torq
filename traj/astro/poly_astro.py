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


class poly_astro(object):

    def __init__(self,pol_leg,tsteps):
        """
        poly_astro:
        a utility class for handling the polynomial components and pre-computed
        parts for use in the ASTRO trajectory optimiser

        Args:
            pol_leg: Legendre polynomial matrix (3D np.array)
                        dim1 - number of coefficients
                        dim2 - number of time steps
                        dim3 - number of derivatives
            tsteps: number of discretisation step_size

        """
        # initialise
        self.state_unscaled = dict()
        self.state_scaled = dict()
        self.int_der_sq = dict()
        # Load in
        self.pol_leg = pol_leg
        self.tsteps = tsteps

        # Generate the unscaled state matrices
        self.create_unscaled_matrices()

        # Pre-compute the path cost component
        self.compute_integral_path_cost_values()

    def create_unscaled_matrices(self):
        """
        create_unscaled_matrices:
        Compute matrix with the polynomail values for each order, for each
        timestep. Also computed for each derivative.
        Used to quickly get the state by multiplying the matrix by the
        coefficients that are solved for.

        Uses:
            self.
            state_unscaled: dictionary for each dimension, storing
            pol_leg: Legendre polynomial matrix (3D np.array)
                        dim1 - number of coefficients
                        dim2 - number of time steps
                        dim3 - number of derivatives
            tsteps: number of discretisation step_size

        Modifies:
            state_unscaled: For each dimension, creates a 3D array with:
                            dim1 - time steps
                            dim2 - coefficients
                            dim3 - derivative number

        """
        # Rename the variables for convenience
        state_unscaled = self.state_unscaled
        pol_leg = self.pol_leg
        tsteps = self.tsteps

        # For each dimension
        for key in pol_leg.keys():
            # Number of coefficients
            N = pol_leg[key].shape[0]
            # Number of derivatives
            n_der = pol_leg[key].shape[2]

            # Initialise 3D array
            state_unscaled[key] = np.zeros([tsteps.size,N,n_der])

            # For each derivatives
            for der in range(0,n_der):
                # For each coefficient
                for pid in range(0,N):
                    # Evaluate the value of the polynomial at every timestep and store
                    state_unscaled[key][:,pid,der] = np.polyval(pol_leg[key][pid,:,der],tsteps)

        # Store in the class
        self.state_unscaled = state_unscaled

    def compute_integral_path_cost_values(self):
        """
        compute_integral_path_cost_values:
        Pre-compute the vector that, when multiplied by the coefficients that
        are being solved for, gives the path cost.

        Cost is the integral of the derivative of interest squared.
        Orthogonality of Legendre polynomails makes this very simple.

        Uses:
            self.
            pol_leg: Legendre polynomial matrix (3D np.array)
                        dim1 - number of basis polynomials
                        dim2 - number of coefficients
                        dim3 - number of derivatives

        Modifies:
            int_der_sq: Vector for each dimension representing the integral of the
                        derivative squared when multiplied by the coefficients.
                        Dictionary of np.arrays, or length equal to the number of coefficients
        """
        # initialise
        pol_leg = self.pol_leg
        int_der_sq= dict()

        # loop for each dimension
        for key in pol_leg.keys():
            # Number of coefficients
            N = pol_leg[key].shape[0]
            int_der_sq[key] = np.zeros(N) # initialise vector

            # For each coefficient
            for pid in range(0,N):
                # Polynomial representing item that cost is on
                # TODO(bmorell@jpl.nasa.gov) change how n_der is calculated and make this adjustable to having a lower derivative as the cost
                pol = np.poly1d(pol_leg[key][pid,:,-1])
                # integral of polynomial squared
                intpol = np.polyint(pol*pol)
                # evaluate at end points
                intsqeval = np.polyval(intpol,[1, -1])
                # Store absolute integral between -1 and 1
                int_der_sq[key][pid] = intsqeval[0] - intsqeval[1]

        self.int_der_sq = int_der_sq

    def scale_matrices(self,times):
        """
        scale_matrices:
        Scales the unscaled matrices with the given final time

        Args:
            times: The time duration for each segment

        Uses:
            self.
            state_unscaled: dictionary for each dimension, storing
            pol_leg: Legendre polynomial matrix (3D np.array)

        Modifies:
            state_scaled: For each dimension, creates a 3D, scaled with the final
                         time array with dimensions:
                            dim1 - time steps
                            dim2 - coefficients
                            dim3 - derivative number
                            dim4 - the segment number
                         Scaled with the final time

        """
        # Rename for convenience
        state_unscaled = self.state_unscaled
        pol_leg = self.pol_leg
        # initialise
        state_scaled = dict()
        n_seg = times.size

        # Loop for each dimension
        for key in state_unscaled.keys():
            # Number of coefficients
            N = pol_leg[key].shape[0]
            # Number of derivatives
            n_der = pol_leg[key].shape[2]
            # initialise
            state_scaled[key] = np.zeros(state_unscaled[key].shape+(n_seg,))

            # For each segment
            for k in range(0,n_seg):
                tf = times[k]
                # For each derivative
                for i in range(0,n_der):
                    # multiply by (tf/2)^XX where XX is the number of integrals from the highest derivative
                    # 0 in the 3rd dimension is positions
                    state_scaled[key][:,:,i,k] = state_unscaled[key][:,:,i]*((tf/2)**(n_der-i-1))

        # Store output in the class
        self.state_scaled = state_scaled
