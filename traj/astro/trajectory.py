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

import abc
import numpy as np


class trajectoryBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self,n_seg=1,n_samp=100,curv_func=False):

        self.n_samp         = n_samp        #     number of trajectory points to evaluate constraints at
        # self.tf            = tf         #      Trajectory time
        # self.costFuncType  = costFuncType         #        1 = Velocity, 2 = snap
        self.curv_func       = curv_func          # Boolean flag. Trues means use the cruvature in the cost functions
        self.n_seg          = n_seg

        # self.c_leg_poly = np.zeros([N*Nx*n_seg,1]);
        # self.states = np.zeros([Nx,n_samp]);

        ########################################
        # Optimisation options
        optim_opt = dict()
        optim_opt['print'] = True
        optim_opt['max_iterations'] = 2000
        optim_opt['step_size'] = 1.0
        # Exit tolerances
        optim_opt['exit_tol'] = 1e-14 # decrease in cost required
        optim_opt['first_order_tol'] = 1e-14 # first order decrease in cost required
        # line search options
        optim_opt['max_armijo'] = 500
        optim_opt['beta'] = 0.85
        optim_opt['sigma'] = 1e-5
        # tracking
        optim_opt['track_data'] = True

        self.optim_opt = optim_opt
        self.step_coeff = 1.0



    @abc.abstractmethod
    def initial_guess(self):
        """ Generate the initial least squares guess """
        return

    # @abc.abstractmethod
    # def load_BCs(self):
    #     """ load in the boundary conditions """
    #     self.BCs = BCs
    #     return

    @abc.abstractmethod
    def generate_legendre_poly(self):
        """ Generate Legendre Polynomails """
        return

    @abc.abstractmethod
    def get_trajectory(self):
        """ return the trajectory """
        return

    @abc.abstractmethod
    def compute_path_cost_grad(self):
        """ compute cost and gradient of the path """
        return

    @abc.abstractmethod
    def enforce_BCs(self):
        """ projection onto subspace enforcing BCs """
        return

    @abc.abstractmethod
    def plot(self):
        """ plot trajectory """
        return

    def legendre_matrix(self,Npl):
        """
        legendre_matrix: Generate a matrix of Legendre polynomial coefficients
        This function creates coefficients for the legendre polynomials with
        order 0 to N-1.  The resulting matrix is NxN

        Args:
            Npl:

        Returns:
            PLeg:Legendre polyomail coefficient matrix

        """
        # Initialize matrix
        PLeg = np.zeros([Npl,Npl])

        # FIrst Legendre Term
        PLeg[0,Npl-1] = 1

        # Loop through each term to define the Legendre Polynomial coefficients
        for i in range(1,Npl):
            PLeg[i,(Npl-i-1):] = self.legendre_poly(i)


        # TODO(bmorrell@jpl.nasa.gov) should this be stored or just be output?
        return PLeg

    def legendre_poly(self,N):
        """
        legendre_poly: creates a legendre polynomial
        Polynomial coefficients are a vector in standard MATLAB form with
        decreasing order from left to right. 1D sequence of polynomial coefficients, from highest to lowest degree

        Args:
            N:

        Returns:
            Pn:Legendre polyomail coefficient matrix

        """
        if N==0: # P_0 = 1
            Pn = 1
        elif N==1: # P_1 = x
            Pn = np.array([1.0, 0])
        else: # Generate according to Bonnet's Recursion
            Pn2 = self.legendre_poly(N-2) #P_(N-2)
            Pn1 = self.legendre_poly(N-1) #P_(N-1)

            # Formula is: P_N = (1/N) * [(2N-1)xP_(N-1) - (N-1)P_(N-2)]
            Pn = (1.0/N)*((2.0*N-1.0)*np.polymul([1.0, 0.0],Pn1) - (N-1.0)*np.hstack(([0.0, 0.0],Pn2)));

        return Pn

def main():
    traj = trajectoryBase()

    import pdb; pdb.set_trace()
    PLeg = traj.legendre_matrix(5)

if __name__ == '__main__':
    main()
