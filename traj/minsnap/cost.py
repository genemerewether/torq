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


def insert(Q, new_index, new_times, costs, order):
    """
    Insert one new waypoint to start at the selected index

    Args:
        Q: previous block diagonal csc or coo scipy sparse matrix of costs
        new_index: index that new waypoints should start at after insertion
        new_waypoints: numpy array
        new_times: array of size 2 with the times for the two segments on either side of the inserted point

    Returns:
        Q: Block diagonal coo scipy sparse matrix for joint optimization

    Raises:
    """

    if Q.format != 'csc':
        Q = Q.tocsc(copy=True)

    # Compute new matrix part
    Q_insert = block_cost(new_times[1], costs, order).tocsc(copy=True)

    if Q_insert.format != 'csc' or Q.format != 'csc':
        msg = 'Can only column-insert in place with both csc format, not {0} and {1}.'
        msg = msg.format(Q_insert.format, Q.format)
        raise ValueError(msg)

    n_coeff = utils.n_coeffs_free_ders(order)[0]
    n_new_seg = np.size(new_times)/2
    n_seg = Q.shape[0]/n_coeff

    if new_index > n_seg:
        index = new_index - 1
    else:
        index = new_index

    # Construct new matrix
    Q_new = sp.sparse.block_diag((Q[:index*n_coeff,:index*n_coeff],Q_insert,Q[index*n_coeff:,index*n_coeff:]))

    Q = Q_new.tocsc(copy=True)

    # modify connecting segment
    if new_index != 0 and new_index != n_seg+1: # Nothing to update for the first segment or last segment
        # Update time for adjoining segment
        update(Q, [new_index-1], new_times[0], costs, order)

    return Q

def delete(Q, delete_index, new_time, costs, order):
    """
    Delete one waypoint with specified index and set duration of joining segment

    Args:
        Q: previous block diagonal csc or coo scipy sparse matrix of costs
        delete_indices: indices of waypoints to delete
        new_time: duration of segment that joins the waypoints on each side
            of the deletion

    Returns:
        Q: Block diagonal coo scipy sparse matrix for joint optimization

    Raises:
    """

    if Q.format != 'lil':
        Q_calc = Q.tolil(copy=True)

    n_coeff = utils.n_coeffs_free_ders(order)[0]
    n_seg = Q.shape[0]/n_coeff

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
    Q_new = sp.sparse.lil_matrix((n_coeff*(n_seg-1),n_coeff*(n_seg-1)))

    if index > 0:
        Q_new[:index*n_coeff,:index*n_coeff] = Q_calc[:index*n_coeff,:index*n_coeff]

    if index < n_seg-1:
        Q_new[index*n_coeff:(n_seg-1)*n_coeff,index*n_coeff:(n_seg-1)*n_coeff] = Q_calc[(index+1)*n_coeff:n_seg*n_coeff,(index+1)*n_coeff:n_seg*n_coeff]

    Q = Q_new.tocsc().copy()

    if delete_index != 0 and delete_index != n_seg: # Nothing to update for the first segment or last segment
        # Update time for adjoining segment
        update(Q, [index-1], new_time, costs, order)

    return Q


def update(Q, new_indices, new_times, costs, order):
    #TODO(mereweth@jpl.nasa.gov) - make this an Object to store costs & order?
    n_coeff = utils.n_coeffs_free_ders(order)[0]
    n_seg = np.size(new_times)

    Q_update = block_cost(new_times, costs, order).tocsc(copy=True)

    if Q_update.format != 'csc' or Q.format != 'csc':
        msg = 'Can only column-update in place with both csc format, not {0} and {1}.'
        msg = msg.format(Q_update.format, Q.format)
        raise ValueError(msg)

    # replace the appropriate sections of Q with sections of Q_update
    for i in range(0, len(new_indices)):
        # the slow way:
        Q[new_indices[i] * n_coeff:(new_indices[i] + 1) * n_coeff,
          new_indices[i] * n_coeff:(new_indices[i] + 1) * n_coeff] = Q_update[i * n_coeff:(i + 1) * n_coeff,
                                                                              i * n_coeff:(i + 1) * n_coeff]

        # TODO(mereweth@jpl.nasa.gov) - fix this and check if it is faster
        # From scipy csc docs:
        #the row indices for column i are stored in indices[indptr[i]:indptr[i+1]]
        # and their corresponding values are stored in data[indptr[i]:indptr[i+1]]
        #Q.data[Q.indptr[new_indices[i] * n_coeff:(new_indices[i] + 1) * n_coeff]] = Q_update.data[Q_update.indptr[i * n_coeff:(i + 1) * n_coeff]]


def block_cost_per_order(times, costs, order, return_all=False):
    """Generate the sparse indices and values for all cost orders

    Richter, Bry, Roy; Polynomial Trajectory Planning for Quadrotor Flight
    Equations 10-15; using joint formulation

    This is broken out as a separate function for testing purposes.

    Args:
        times: time in which to complete each segment
        cost: weight in the sum of Hessian matrices (Equation 15)
        order: the order of the polynomial segments
        return_all: Whether to return all cost terms even if only some are
            nonzero

    Returns:
        A tuple, (i, j, k, val) where i is the index of the cost order, j is
            the row, k is the column, and val is the appropriate Hessian value
    Raises:
    """

    n_coeff = utils.n_coeffs_free_ders(order)[0]
    n_seg = np.size(times)

    # debugging feature that returns the Hessian for each derivative order
    if return_all:
        der = np.ones(n_coeff)

    # derivatives from r = 0 to polyOrder where cost is not zero
    # numpy.nonzero returns a tuple; one element for each dimension of input
        # array
    der = np.nonzero(costs)[0]

    # changed dimension order here with respect to MATLAB scripts
    # due to numpy native print order
    der_outer_prods = np.zeros((np.size(der), n_coeff, n_coeff))
    t_power = np.zeros(np.shape(der_outer_prods))
    for i in range(np.size(der)):
        der_order = der[i]

        if der_order > 0:
            der_outer_prods[i, :, :] = utils.poly_hessian_coeffs(order,
                                                                 der_order)[1]
        else:
            # empty product because r=0-1 is less than m=0
            der_outer_prods[i, :, :] = np.ones((n_coeff, n_coeff))

        [row, col] = np.mgrid[:n_coeff, :n_coeff]
        # coefficients of terms according to their order; t is raised to this
        # power, and each term is divided by this value as well
        t_power[i, :, :] = row + col + np.ones(np.shape(row)) * \
            (-2 * der_order + 1)

    # See Equation 14 - strictly greater than 0
    gt_zero_idx = np.nonzero(t_power > 0)

    # this works for a single cost term and for multiple cost terms
    cost_per_order = 2 * np.expand_dims(der_outer_prods[gt_zero_idx], axis=1) \
        * times ** np.expand_dims(t_power[gt_zero_idx], axis=1) \
        / np.expand_dims(t_power[gt_zero_idx], axis=1)

    # first list in tuple determines index for 3rd dimension of sparse matrix,
    # which gives the derivative order that the cost was calculated for

    # index in cost_per_order corresponding to the length of the segment times
    # list tells which block in sparse matrix to insert into

    # need to duplicate gt_zero_idx and offset to the appropriate block

    tiled_idx = np.tile(gt_zero_idx, (1, np.size(times)))

    tiled_idx[1:, np.size(gt_zero_idx[0]):] \
        += np.tile(n_coeff * np.r_[1:np.size(times)],
                   (np.size(gt_zero_idx[0]), 1)).T.flatten()

    return (tiled_idx[0], tiled_idx[1], tiled_idx[2],
            cost_per_order.T.flatten())


def block_cost(times, costs, order):
    """Generate the cost matrix for each segment.

    Richter, Bry, Roy; Polynomial Trajectory Planning for Quadrotor Flight
    Equations 10-15; using joint formulation

    Args:
        times: time in which to complete each segment
        cost: weight in the sum of Hessian matrices (Equation 15)
        order: the order of the polynomial segments

    Returns:
        Block diagonal coo scipy sparse matrix for joint optimization
            before summing across cost terms (sum happens upon matrix conversion
            to CSR or CSC format)
    Raises:
    """

    n_coeff = utils.n_coeffs_free_ders(order)[0]
    n_seg = np.size(times)
    block_cost_sparse = block_cost_per_order(times, costs, order)

    # Weighted sum of Hessian cost for each order

    # derivatives from r = 0 to polyOrder where cost is not zero
    # numpy.nonzero returns a tuple; one element for each dimension of input
    # array
    der = np.nonzero(costs)[0]
    costs = np.array(costs)[der]

    # TODO(mereweth@jpl.nasa.gov) duplicate entries are not summed until
    # converted to CSR/CSC - is this ok?
    return sp.sparse.coo_matrix((costs[block_cost_sparse[0]] *
                                 block_cost_sparse[3],
                                 (block_cost_sparse[1],
                                  block_cost_sparse[2])),
                                shape=(n_coeff * n_seg, n_coeff * n_seg))


def main():
    costs = [0, 0, 1, 0, 1]

    times = [2, 1]
    order = 7
    print("Joint cost matrix for seg times {}; cost {}; poly order {}\n".format(
        times, costs, order))
    result = block_cost(times, costs, order)
    print(result.todense())

    costs = [1, 0, 0, 0, 0]
    times = [2,2,2]#,2,2]
    order = 3
    print("Joint cost matrix for seg times {}; cost {}; poly order {}\n".format(
        times, costs, order))
    result = block_cost(times, costs, order).tocsc(copy=True)
    print(result.todense())
    res1 = result.copy()

    # result = sp.sparse.block_diag((result,result,result))
    import pdb; pdb.set_trace()
    # update(result,[0],7,costs,order)
    # print(result.todense())

    # res2 = delete(result,2,3,costs,order).copy()
    res2 = insert(result,2,[3,3],costs,order)


    # new_res = insert(result, 0, 0.2, costs, order)
    print(res2.todense())

    costs = [1, 0, 0, 0, 0]
    times = [2,2,2,2,2]#,2,2]
    order = 3
    result = block_cost(times, costs, order).tocsc()
    res5 = insert(result,3,[3,3],costs,order)
    res4 = delete(res5,3,2,costs,order)
    # res5 = insert(res4,2,[2,2],costs,order)
    # res4 = delete(res5,1,3,costs,order)



    import pdb; pdb.set_trace()
if __name__ == '__main__':
    main()
