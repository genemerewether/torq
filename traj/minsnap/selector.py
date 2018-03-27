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

__author__ = "Gene Merewether"
__email__ = "mereweth@jpl.nasa.gov"


# Import statements
# =================

import numpy as np

import scipy.sparse
sp = scipy

from minsnap import utils
from scipy.linalg import block_diag


def block_selector(der_fixed,closed_loop=False):
    """Generate sparse selector matrix

    Richter, Bry, Roy; Polynomial Trajectory Planning for Quadrotor Flight
    Equation 31

    This matrix maps between the partitioned arrangement of free and fixed
    derivatives, Eqn. 26, and the ordering in the constraint matrix, A.

    Args:
        der_fixed: Boolean array of n_der (max potential number of free
            derivatives per segment) x n_seg + 1 (one column per waypoint).
            Fixing a derivative at a waypoint is performed by setting the
            corresponding entry to True.
        order: the highest order of the polynomials used in the optimisation

    Returns:
        M: sparse selector matrix (nonsquare)

    Raises:
    """

    # TODO(mereweth@jpl.nasa.gov) - the operations utils.dup_columns(der_fixed)
    # and np.logical_not(der_fixed) are carried out repeatedly to avoid having
    # extra named variables. If you are trying to improve performance, storing
    # the result the first time these functions are evaluated may make sense.

    # der_fixed should already be a numpy array, but make it one anyway
    der_fixed = np.array(der_fixed)
    num_der_fixed = np.sum(der_fixed)
    if closed_loop:
        num_der_fixed_end = np.sum(der_fixed[:,0])

    # examples from MATLAB use column major indexing; we want to use row-major
    # indexing internally in this function for Numpy ease
    der_fixed = der_fixed.T
    # import pdb; pdb.set_trace()

    # fixed derivatives mapping to [start end] list of segment derivatives
    row = np.flatnonzero(utils.dup_internal(der_fixed))
    # necessary because numpy is 0-based
    tmp_col = -1 * np.ones(np.shape(der_fixed), dtype=int)
    tmp_col[der_fixed] = np.arange(num_der_fixed, dtype=int)
    # tmp_col[der_fixed] = np.r_[:num_der_fixed, dtype=int]
    tmp_col = utils.dup_internal(tmp_col)
    if closed_loop:
        # Copy first column to the last to force continuity
        tmp_col[-1,:] = tmp_col[0,:]
    col = tmp_col[tmp_col >= 0]

    # free derivatives mapping to [start end] list of segment derivatives
    # these are the derivatives that are optimized
    row = np.append(row, np.flatnonzero(utils.dup_internal(
        np.logical_not(der_fixed))))
    # necessary because numpy is 0-based
    tmp_col = -1 * np.ones(np.shape(der_fixed), dtype=int)
    tmp_col[np.logical_not(der_fixed)] = np.arange(np.size(der_fixed)
                                                   - num_der_fixed,
                                                   dtype=int)
    # tmp_col[np.logical_not(der_fixed)] = np.r_[:(np.size(der_fixed)
    #                                             - num_der_fixed)]
    tmp_col = utils.dup_internal(tmp_col)
    if closed_loop:
        # Copy first column to the last to force continuity
        tmp_col[-1,:] = tmp_col[0,:]
        col = np.append(col, (num_der_fixed-num_der_fixed_end) + tmp_col[tmp_col >= 0])
    else:
        col = np.append(col, num_der_fixed + tmp_col[tmp_col >= 0])

    val = np.ones(np.size(row), dtype=int)
    # the row & col indices we generated are for M_trans, so just flip them
    if closed_loop:
        M = sp.sparse.coo_matrix((val, (col, row)),
                             shape=(np.size(der_fixed) - der_fixed.shape[1],
                                    np.size(utils.dup_internal(der_fixed))),
                             dtype=int)

    else:
        M = sp.sparse.coo_matrix((val, (col, row)),
                              shape=(np.size(der_fixed),
                                     np.size(utils.dup_internal(der_fixed))),
                              dtype=int)


    return M


def main():
    der_fixed = [[True, True, True], [True, False, True]]
    print("Selector matrix for derivatives {}\n".format(der_fixed))
    result = block_selector(der_fixed)
    print(result.todense())

    import pdb; pdb.set_trace()
    der_fixed = [[True, True, True], [False, False, False]]
    result = block_selector(der_fixed,closed_loop=True)
    print(result.todense())

    der_fixed = [[True, True], [True, True]]
    print("Selector matrix for derivatives {}\n".format(der_fixed))
    result = block_selector(der_fixed)
    print(result.todense())

    der_fixed = [[True, True, True], [True, False, True], [True,False, True]]
    print("Selector matrix for derivatives {}\n".format(der_fixed))
    result = block_selector(der_fixed)
    print(result.todense())

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
