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
from scipy.linalg import block_diag
from math import factorial

import scipy.sparse
sp = scipy

from minsnap import utils


def insert(A, new_index, new_times, order):
    """
    Insert new waypoints to start at the selected index

    Args:
        A: previous block diagonal coo scipy sparse matrix of constraints
        new_index: index that new waypoints should start at after insertion
        new_waypoints: numpy array
        new_times:
        new_der_fixed:

    Returns:
        A: Block diagonal coo scipy sparse matrix for joint optimization

    Raises:
    """
    if A.format != 'csc':
        A = A.tocsc(copy=True)

    # Compute new matrix part
    A_insert = block_constraint(new_times[1], order).tocsc(copy=True)

    if A_insert.format != 'csc' or A.format != 'csc':
        msg = 'Can only column-insert in place with both csc format, not {0} and {1}.'
        msg = msg.format(A_insert.format, A.format)
        raise ValueError(msg)

    n_coeff = utils.n_coeffs_free_ders(order)[0]
    n_new_seg = np.size(new_times)/2
    n_seg = A.shape[0]/n_coeff

    if new_index > n_seg:
        index = new_index - 1
    else:
        index = new_index

    # Construct new matrix
    A_new = sp.sparse.block_diag((A[:index*n_coeff,:index*n_coeff],A_insert,A[index*n_coeff:,index*n_coeff:]))

    A = A_new.tocsc(copy=True)

    # modify connecting segment
    if new_index != 0 and new_index != n_seg+1: # Nothing to update for the first segment or last segment
        # Update time for adjoining segment
        update(A, [new_index-1], new_times[0], order)

    return A


def delete(A, delete_index, new_time, order):
    """
    Delete waypoints with specified indices and set duration of joining segment

    Args:
        A: previous block diagonal coo scipy sparse matrix of constraints
        delete_index: indices of waypoints to delete
        new_time: duration of segment that joins the waypoints on each side
            of the deletion

    Returns:
        A: Block diagonal coo scipy sparse matrix for joint optimization

    Raises:
    """
    if A.format != 'lil':
        A_calc = A.tolil(copy=True)

    n_coeff = utils.n_coeffs_free_ders(order)[0]
    n_seg = A.shape[0]/n_coeff

    if delete_index == n_seg:
        # Change index if the last segment
        index = delete_index - 1
    else:
        index = delete_index

    # Compute indices of parts of matrix to keep
    # start_ind = np.arange(0,index*n_coeff,1)
    # end_ind = np.arange((index+1)*n_coeff,n_seg*n_coeff,1)
    # keep_ind = np.concatenate((start_ind,end_ind))

    # Remove from delete index
    A_new = sp.sparse.lil_matrix((n_coeff*(n_seg-1),n_coeff*(n_seg-1)))

    if index > 0:
        A_new[:index*n_coeff,:index*n_coeff] = A_calc[:index*n_coeff,:index*n_coeff]

    if index < n_seg-1:
        A_new[index*n_coeff:(n_seg-1)*n_coeff,index*n_coeff:(n_seg-1)*n_coeff] = A_calc[(index+1)*n_coeff:n_seg*n_coeff,(index+1)*n_coeff:n_seg*n_coeff]

    A = A_new.tocsc().copy()

    if delete_index != 0 and delete_index != n_seg: # Nothing to update for the first segment or last segment
        # Update time for adjoining segment
        update(A, [index-1], new_time, order)

    return A


def invert(A, order):
    n_coeff, n_der = utils.n_coeffs_free_ders(order)
    n_seg = np.shape(A)[0] / 2 / n_der

    # not square or can't figure out the format
    if (np.shape(A)[0] != np.shape(A)[1]  or
                                        n_seg != np.shape(A)[1] / n_coeff or
                                        round(n_seg) != n_seg):
        return np.asmatrix(sp.linalg.pinv(A.todense()))

    else:
        blocks = []
        # use Schur complement as described in:
        # Real-Time Visual-Inertial Mapping, Re-localization and Planning
        # Onboard MAVs in Unknown Environments

        # TODO(mereweth@jpl.nasa.gov) - don't loop here
        # TODO(mereweth@jpl.nasa.gov) - is there a better way to slice A?
        for seg in range(n_seg):
            A_seg = A[seg * n_coeff:(seg + 1) * n_coeff,
                      seg * n_coeff:(seg + 1) * n_coeff].todense()
            sigma = np.array(A_seg[:n_der, :n_der].diagonal()).squeeze()
            sigma_inv = np.matrix(np.diag(1 / sigma))
            gamma = np.matrix(A_seg[n_der:, :n_der])
            delta = A_seg[n_der:, n_der:]
            # TODO(mereweth@jpl.nasa.gov) - is this always invertible?
            delta_inv = np.matrix(np.linalg.inv(delta))

            A_seg_inv = np.r_[np.c_[sigma_inv, np.zeros((n_der, n_der))],
                              np.c_[- delta_inv * gamma * sigma_inv, delta_inv]]

            blocks.append(A_seg_inv)

        return sp.sparse.block_diag(blocks, 'csr')

def update(A, new_indices, new_times, order):
    #TODO(mereweth@jpl.nasa.gov) - make this an Object to store order?
    n_coeff, n_der = utils.n_coeffs_free_ders(order)
    n_seg = np.size(new_times)

    A_update = block_constraint(new_times, order).tocsc(copy=True)

    if A_update.format != 'csc' or A.format != 'csc':
        msg = 'Can only column-update in place with both csc format, not {0} and {1}.'
        msg = msg.format(A_update.format, A.format)
        raise ValueError(msg)

    # replace the appropriate sections of A with sections of A_update
    for i in range(0, len(new_indices)):
        # the slow way:
        A[new_indices[i] * 2 * n_der:(new_indices[i] + 1) * 2 * n_der,
          new_indices[i] * 2 * n_der:(new_indices[i] + 1) * 2 * n_der] = A_update[i * 2 * n_der:(i + 1) * 2 * n_der,
                                                                                  i * 2 * n_der:(i + 1) * 2 * n_der]

        # TODO(mereweth@jpl.nasa.gov) - fix this and check if it is faster
        # From scipy csc docs:
        #the row indices for column i are stored in indices[indptr[i]:indptr[i+1]]
        # and their corresponding values are stored in data[indptr[i]:indptr[i+1]]
        #A.data[A.indptr[new_indices[i] * n_coeff:(new_indices[i] + 1) * n_coeff]] = A_update.data[A_update.indptr[i * n_coeff:(i + 1) * n_coeff]]

def sparse_values(times, order):
    """Generate the sparse values for the constraint matrix

    Richter, Bry, Roy; Polynomial Trajectory Planning for Quadrotor Flight
    Equations 18 & 20; using joint formulation

    Args:
        times: time in which to complete each segment
        order: the order of the polynomial segments

    Returns:
        A tuple with a_0_val, a_t_val, where:
            a_0_val: 0th derivative values
            a_t_val: higher derivative values
    Raises:
    """
    # TODO(mereweth@jpl.nasa.gov) could reuse this by returning it
    n_coeff, n_der = utils.n_coeffs_free_ders(order)
    n_seg = np.size(times)

    n_minus_r, prod = utils.poly_der_coeffs(order)

    # raise segment times to n_minus_r power
    t_power = np.reshape(times, (n_seg, 1, 1)) ** n_minus_r


    a_0_val = np.tile(np.diag(prod), (n_seg))  # equation 18
    a_t = np.tile(prod, (n_seg, 1, 1)) * t_power  # equation 20

    # unlike MATLAB, have to convert to linear sequence, index, then convert back
    # which values are below the diagonal?
    a_t_sel = np.squeeze(np.tile(
        [np.tri(n_coeff, n_der, dtype=np.int)], (n_seg, 1, 1)).reshape((1, 1, -1)))

    # transpose a_t and a_t_sel?
    # use triu_indices instead?

    # get the indices for all values
    a_t_idx = np.r_[:n_coeff * n_der * n_seg]

    a_t = np.squeeze(a_t.reshape((1, 1, -1)))

    # get only the values below the diagonal
    a_t_val = a_t[a_t_idx[a_t_sel > 0]]

    return (a_0_val, a_t_val)


def sparse_indices(times, order):
    """Generate the sparse indices for the constraint matrix

    Richter, Bry, Roy; Polynomial Trajectory Planning for Quadrotor Flight
    Equations 18 & 20; using joint formulation

    Args:
        times: time in which to complete each segment
        order: the order of the polynomial segments

    Returns:
        A tuple with a_0_row, a_0_col, a_t_row, a_t_col, where:
            a_0_row: 0th derivative row indices
            a_0_col: 0th derivative column indices
            a_t_row: higher derivative row indices
            a_t_col: higher derivative column indices
    Raises:
    """
    # TODO(mereweth@jpl.nasa.gov) could reuse this by returning it
    n_coeff, n_der = utils.n_coeffs_free_ders(order)
    n_seg = np.size(times)

    i, j = np.nonzero(np.tri(n_coeff, n_der, dtype=np.int))

    a_0_row = np.r_[:n_der * n_seg] + np.repeat(np.r_[:n_seg] * n_der, n_der)
    if order % 2 == 0:
        # for even polynomial order, each block is taller by 1
        a_0_col = a_0_row + np.repeat(np.r_[:n_seg], n_der)

        # 2 * n_der is correct here; there are two submatrices of n_der rows each
        # per segment
        a_t_row = np.tile(j, (n_seg)) + \
            np.repeat(np.r_[:n_seg] * 2 * n_der, i.size) + n_der

        # n_coeff is correct here; there is one matrix of n_coeff columns per
        # segment
        a_t_col = np.tile(i, (n_seg)) + \
            np.repeat(np.r_[:n_seg] * n_coeff, i.size)
    else:
        a_0_col = a_0_row  # no copy is ok because a_0_row is not modified later

        # n_coeff is correct here; there is one matrix of n_coeff columns per
        # segment
        a_t_row = np.tile(j, (n_seg)) + \
            np.repeat(np.r_[:n_seg] * n_coeff, i.size) + n_der

        # 2 * n_der is correct here; there are two submatrices of n_der rows each
        # per segment
        a_t_col = np.tile(i, (n_seg)) + \
            np.repeat(np.r_[:n_seg] * 2 * n_der, i.size)

    return (a_0_row, a_0_col, a_t_row, a_t_col)

def block_constraint(times, order):
    """Generate the constraint matrix for beginning and end of each segment.

    Maps between polynomial coefficients and derivatives evaluated at
    beginning and end of segments.

    Bry, Richter, Bachrach and Roy; Agressive Flight of Fixed-Wing and Quadrotor
    Aircraft in Dense Indoor Environments
    Equations 116 - 122; using joint formulation
    Continuity A is from eqn 12
    Boundary Conditions A is from eqn 13

    Args:
        times: time in which to complete each segment
        order: the order of the polynomial segments

    Returns:
        A: Block diagonal coo scipy sparse matrix for joint optimization
    Raises:
    """

    n_coeff, n_der = utils.n_coeffs_free_ders(order)
    n_seg = np.size(times)

    a_0_val, a_t_val = sparse_values(times, order)

    (a_0_row, a_0_col, a_t_row, a_t_col) = sparse_indices(times, order)

    A = sp.sparse.coo_matrix((np.concatenate([a_0_val, a_t_val]),
                              (np.concatenate([a_0_row, a_t_row]),
                               np.concatenate([a_0_col, a_t_col]))),
                             shape=(2 * n_der * n_seg, n_coeff * n_seg))

    # Make the matrix sparse in format
    A = sp.sparse.coo_matrix(A)

    return A

def block_ineq_constraint(der_fixed,der_ineq):
    """Generate the a mapping matrix for inequality constraints

    Creates a matrix to pick out the free derivatives that have inequality constraints
    associated with them

    Args:
        der_fixed: the fixed derivatives boolean matrix. A column for each waypoint (nder by nseg+1)
        der_ineq: boolean matrix for the derivatives that have inequalities associated with them. A column for each waypoint (nder by nseg+1)

    Returns:
        E: Block diagonal coo scipy sparse matrix for joint optimization with inequality constraints
    Raises:
    """

    # Get vector of the free derivatives
    free_vec = ~np.reshape(der_fixed.T,[der_fixed.size])

    # From der_ineq into a vector
    ineq_vec = np.reshape(der_ineq.T,[der_ineq.size])

    # Extract ineq vector for the free terms
    ineq_vec = np.array(ineq_vec[free_vec])

    # Create E matrix block
    E = np.diag(ineq_vec)

    # Take out zero rows
    E = np.array(E[np.sum(E,1)!=0,:]).astype(np.int)

    # Bring the matrices together
    E = np.concatenate((E,-E),axis=0)

    # Put into matrix format
    E = sp.sparse.coo_matrix(E,dtype=int)

    return E


def main():
    times = 1.
    order = 5
    print("Joint constraint matrix for seg times {}; poly order {}\n".format(
        times, order))
    result = block_constraint(times, order)
    print(result.todense())

    times = [1.0,1.0,1.0,1.0,1.0]
    order = 9
    print("Joint constraint matrix for seg times {}; poly order {}\n".format(
        times, order))
    result = block_constraint(times, order)
    print(result.todense())

    import pdb; pdb.set_trace()
    times = [2]
    order = 3
    res = block_constraint(times, order)
    res1 = insert(res,1,[3,3],order)




    # res2 = delete(res1,1,2,order)

    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()
