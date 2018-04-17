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

# import voxblox

# class constraintArray(object):
#     def __init__(self,size=1):
#         """
#         An array of constraints
#         """


class constraintBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self,weight=1.0,keep_out=True,der=0,dynamic_weighting=False,custom_weighting=False):

        # self.fcnID = 1

        self.der = der # Derivative to operate on
        self.weight = weight

        # Boolean flags
        self.moving     = False
        self.curv_func  = False
        self.keep_out   = keep_out
        self.active_seg = None

        self.feasible = False
        self.seg_feasible = False

        self.constraint_type = "null"

        self.dynamic_weighting = dynamic_weighting

        self.custom_weighting = custom_weighting

        # self.cost = 0
        # self.grad = np.array([])
        # self.curvature = np.array([])


    @abc.abstractmethod
    def cost_grad_curv(self,state, seg=0,doGrad=True,doCurv=False):
        """ Compute the constraint cost, gradient and curvature given the trajectory
            stored in state

            Args:
                state: The trajectory to evaluate for obstacle costs. np.array with dimensions (nder,nsamp,nseg).
                        A dict for each dimension
                seg: Which segment to act on
                doGrad: boolean to select whether or not to evaluate the gradient
                doCurv: boolean to select whether or not to evaluate the curvature

            Outputs:
                cost: the cost for the constraint (single number)
                grad: the cost gradient for the constraint. A dict for each dimension,
                                in each an np.array of length N (number of coefficients for each dimension)
                curv: the cost curvature for the constraint. A dict for each dimension,
                                in each an np.array of length N (number of coefficients for each dimension)
                max_ID: Index of the maximum violation
        """
        return

    def compute_constraint_cost_grad_curv(self, state, state_scaled, doGrad, doCurv=False, path_cost=0.0):
        """
        Computes the cost and gradient (df/dC) for one constraint

        Input:
            state: dict for each dim, 3D np.array([n_der,n_samp,n_seg]). Stores all timesteps for each dimension for each segment
            state_scaled: dict for each dimension: the scaled polynomial evaluations at each timestep.
                               3D np.array([n_samp,number of coefficients (N), n_der])

        Output:
            constr_cost: the cost for the constraint (single number)
            constr_cost_grad: the cost gradient for the constraint. A dict for each dimension,
                            in each an np.array of length N x n_seg

        """

        # Number of segments
        n_seg = state['x'].shape[2]

        # Initialise
        constr_cost_grad = dict()
        constr_grad_check = dict()
        constr_cost_curv = dict()
        constr_cost = 0.0;
        constr_cost_sqrt = 0.0;

        feasible = True;

        if path_cost == 0.0:
            self.dynamic_weighting = False

        # initialise gradient dictionary
        for key in state.keys():
            # Number of coefficients per dimension
            N = state_scaled[key].shape[1]
            constr_cost_grad[key] = np.zeros(N*n_seg)
            constr_grad_check[key] = np.zeros(N*n_seg)
            constr_cost_curv[key] = np.zeros((N*n_seg,N*n_seg))

        # Loop for each segment
        for i in range(0,n_seg):

            if self.active_seg is not None and i != self.active_seg:
                # Not an active segment, return zero cost and gradient
                continue

            # Compute df/dx for the constraint
            cost, grad, curv, max_ID = self.cost_grad_curv(state,i,doGrad=doGrad,doCurv=doCurv)

            if self.dynamic_weighting and cost > 0.0:
                cost_weight = 10**(np.round(np.log10(path_cost))-np.round(np.log10(cost**2))+1)
                if not self.keep_out:
                    cost_weight = 10**(np.round(np.log10(path_cost/n_seg)))#-np.round(np.log10(cost**2))+2)
            elif self.custom_weighting and cost > 0.0:
                cost_weight = 1.0 # Temporary - set at the end
            else:
                cost_weight = self.weight

            # Track overall feasibility for the constraint
            feasible = feasible and self.seg_feasible

            # Add the cost for the constraint
            # if not doCurv or self.keep_out is True:
            if self.cost_type is "squared":
                constr_cost += cost_weight * cost**2
                constr_cost_sqrt += cost

            elif self.cost_type is "not_squared":
                # Cost is just f
                constr_cost += cost_weight * cost

            # print("obstacle cost is {}".format(constr_cost))
            # Compute gradient if constraint is active
            if doGrad and max_ID[0] != -1 and self.constraint_type is not "esdf_check":

                # Loop for each dimension
                for key in ['x','y','z']:
                    # Number of coefficients per dimension
                    N = state_scaled[key].shape[1]
                    nder = state_scaled[key].shape[2]

                    # Form the sum of scaled coefficients weighted by the gradient.  This
                    # is equivalent to (df/dx)*(dX/dC)  = df/dC
                    base = np.zeros((N,1)) # initialise
                    if doCurv:
                        crv_base = np.zeros((N,N))

                    for ID in np.array(max_ID):
                        # Loop for each derivative
                        for j in range(0,nder):
                            # Combine together
                            base += (grad[key][j,ID]*state_scaled[key][ID,:,j,i]).reshape((N,1))
                            if doCurv:
                                # crv_base += curv[key][j,ID]*np.array(np.matrix(state_scaled[key][ID,:,j,i]).reshape((N,1))*np.matrix(state_scaled[key][ID,:,j,i]))
                                if key is not 'yaw': # TODO include yaw if it is included in a cost function
                                    # print("Key is {}\nN is {}\nstate_scaled size is x{}, y{}, z{}".format(key,N,np.shape(state_scaled['x']),np.shape(state_scaled['y']),np.shape(state_scaled['z'])))
                                    crv_base += curv[key][j,0,ID]*np.array(np.matrix(state_scaled['x'][ID,:,j,i]).reshape((N,1))*np.matrix(state_scaled[key][ID,:,j,i]))
                                    crv_base += curv[key][j,1,ID]*np.array(np.matrix(state_scaled['y'][ID,:,j,i]).reshape((N,1))*np.matrix(state_scaled[key][ID,:,j,i]))
                                    crv_base += curv[key][j,2,ID]*np.array(np.matrix(state_scaled['z'][ID,:,j,i]).reshape((N,1))*np.matrix(state_scaled[key][ID,:,j,i]))

                    # Store for current segment and dimension
                    # constr_cost_grad[key][N*i:N*(i+1)] = base
                    if doCurv:
                        constr_cost_curv[key][N*i:N*(i+1),N*i:N*(i+1)] = crv_base*cost_weight

            # else: Zeros if gradient not needed
                    constr_grad_check[key][N*i:N*(i+1)] = base.reshape(N)
                    # For cost that is W*f**2, grad is W*2*f*(df/Dx)' * dX/dC, curvature is 2*[(df/dX)' * dX/dC]^2 + 2*f*(d^2f/dX^2)(dX/dC)^2
                    # For cost that is W*f, then no more computations are required TODO CHECK THIS
                    if self.cost_type is "squared":
                        # constr_cost_curv[key] = 2*constr_cost_curv[key]*constr_cost + 2*constr_cost_grad[key]**2
                        constr_cost_grad[key][N*i:N*(i+1)] = (2*base*cost*cost_weight).reshape(N)
                    elif self.cost_type is "not_squared":
                        constr_cost_grad[key][N*i:N*(i+1)] = (base*cost_weight).reshape(N)

        # if not doCurv or self.keep_out is True:
        #     for key in constr_grad_check.keys():
        #         constr_grad_check[key] = 2*constr_grad_check[key]*constr_cost*cost_weight

        if self.custom_weighting:
            # Set one weighting for all segments, on the first iteration, based on the path cost
            if self.constraint_type is "esdf":
                # cost_weight = 10**(np.round(np.log10(path_cost)*3.0)-np.round(np.log10(constr_cost)*0.9))# 198
                # cost_weight = 10**(np.round(np.log10(path_cost)*3.25)-np.round(np.log10(constr_cost)*0.6))# 344
                cost_weight = 10**(np.round(np.log10(path_cost)*1.5)-np.round(np.log10(constr_cost)*0.1))# Unreal
                # import pdb; pdb.set_trace()
                print("\n\nCustom Cost Weight is {}\n\n".format(cost_weight))
                # import pdb; pdb.set_trace()
            elif self.constraint_type is "cylinder":
                cost_weight = 10**(np.round(np.log10(path_cost)*0.5)-np.round(np.log10(constr_cost)*2.0))
                print("\n\nCustom Cost Weight is {}\n\n".format(cost_weight))
            else:
                cost_weight = self.weight
                # import pdb; pdb.set_trace()
            # cost_weight = 10**(np.round(np.log10(path_cost)*1.5)-6)#np.round(np.log10(constr_cost))+2)

            # apply weight
            constr_cost *= cost_weight
            for key in constr_cost_grad.keys():
                constr_cost_grad[key] *= cost_weight
                constr_cost_curv[key] *= cost_weight

            self.custom_weighting = False
            self.weight = cost_weight

        # Feasibility check
        self.feasible = feasible

        return constr_cost, constr_cost_grad, constr_cost_curv



class ellipsoid_constraint(constraintBase):

    def __init__(self,weight,keep_out,der,x0,A,rot_mat,dynamic_weighting=False,doCurv=False,sum_func = True,custom_weighting=False):
        """

        """

        # Initialise parent case class
        super(ellipsoid_constraint,self).__init__(weight,keep_out,der,dynamic_weighting,custom_weighting=custom_weighting)

        self.constraint_type = "ellipsoid"

        self.seg_feasible = True

        self.x0 = x0

        self.A = A
        self.rot_mat = rot_mat

        self.sum_func = sum_func

        self.curv_func = doCurv

        if self.keep_out:
            self.in_out_scale = -1
        else:
            self.in_out_scale = 1

        if not self.curv_func or self.keep_out is True:
            self.cost_type = "squared"
        else:
            self.cost_type = "not_squared"




    def cost_grad_curv(self, state, seg = 0, doGrad=True, doCurv=False):
        """
        Computes cost, cost gradient and cost curvature of the ellipsoid constraint. A dictionary for each dimension

        Cost gradient and curvature are returned for use in optimisation steps (e.g. linesearch)

        Args:
            state: The trajectory to evaluate for obstacle costs. np.array with dimensions (nder,nsamp,nseg).
                    A dict for each dimension
            seg:    The segment to calculate the cost and gradient for
            doGrad: boolean to select whether or not to evaluate the gradient
            doCurv: boolean to select whether or not to evaluate the curvature

        Uses:
            self.
            moving: Flag to indicate if the obstacle is moving
            x0: position of the centre of the ellipsoid obstacle (np.array, 3 by nsmap)
            der: Derivative to operate on
            in_out_scale: +1 or -1 to indicate if the obstacle is a keep out (-1) or keep in (+1) obstacle
            A: Shape matrix for the ellipsoid

        Outputs:
            cost: the cost for the constraint (single number)
            grad: the cost gradient for the constraint. A dict for each dimension,
                            in each an np.array of length N (number of coefficients for each dimension)
            curv: the cost curvature for the constraint. A dict for each dimension,
                            in each an np.array of length N (number of coefficients for each dimension)
            max_ID: Index of the maximum violation

        """

        # Number of samples in a segment
        nder = state['x'].shape[0]
        nsamp = state['x'].shape[1]

        # Initialise grad: df/dx at the maximum, so for each dimension is an array of the number of derivatives (x includes the derivatives)
        # grad = dict(x=np.zeros((nder,1)),y=np.zeros((nder,1)),z=np.zeros((nder,1)),yaw=np.zeros((nder,1)))
        # curv = dict(x=np.zeros((nder,1)),y=np.zeros((nder,1)),z=np.zeros((nder,1)),yaw=np.zeros((nder,1)))

        grad = dict(x=np.zeros((nder,nsamp)),y=np.zeros((nder,nsamp)),z=np.zeros((nder,nsamp)),yaw=np.zeros((nder,nsamp)))
        curv = dict(x=np.zeros((nder,3,nsamp)),y=np.zeros((nder,3,nsamp)),z=np.zeros((nder,3,nsamp)),yaw=np.zeros((nder,3,nsamp)))

        if self.moving:
            if self.x0.shape[1] == nsamp:
                x0 = self.x0
            else:
                print("Error: need centre to be defined as a trajectory the same size as state")
                return
        else:
            # Copy centre for the computations
            x0 = np.repeat(np.reshape(self.x0,(3,1)),nsamp,axis=1)

        # normalise state from the centre
        x = np.matrix(np.zeros((3,nsamp)))
        x[0,:] = state['x'][self.der,:,seg] - x0[0,:]
        x[1,:] = state['y'][self.der,:,seg] - x0[1,:]
        x[2,:] = state['z'][self.der,:,seg] - x0[2,:]

        # Calculate cost
        cost_tmp = np.zeros(nsamp)
        for i in range(nsamp):
            cost_tmp[i] = self.in_out_scale*(x[:,i].T*self.A*x[:,i] - 1)[0,0]

        if self.sum_func:
            # Summ all costs
            if self.keep_out is True:
                cost_tmp[cost_tmp < 0.0] = 0.0
                max_ID = np.where(cost_tmp>0.0)[0]
            else:
                max_ID = np.where(cost_tmp>-np.inf)[0]

            max_cost = np.sum(cost_tmp)

            max_check = np.amax(cost_tmp)

        else:
            # Maximum violation
            max_cost = np.amax(cost_tmp)
            max_check = max_cost
            if np.sum(np.isclose(cost_tmp,max_cost))<1:
                print("ERROR - can't match max in cost_tmp")
            max_ID = np.where(np.isclose(cost_tmp,max_cost))[0][0]

        # Check non-inflated feasibility
        if max_check <= 0:
            self.seg_feasible = True # TODO - do this for the other constraints
            # print("ESDF Constraint in segment is feasible. Distance is {} (m), gradient is: {}".format(max_check, -grad[:,max_ID]))
            print("Ellipsoid Constraint in segment is feasible. Distance is {} (m)".format(max_check))
        else:
            print("Ellipsoid Constraint in segment is NOT feasible. Distance is {} (m)".format(max_check))
            import pdb; pdb.set_trace()
            self.seg_feasible = False

        # if np.size(max_ID) > 1:
        #     max_ID = max_ID[0]
        if max_cost <= 0:
            # Constraint not active - no need to compute gradient.
            # Set max ID to negative as a flag
            max_ID = -1
            max_cost = 0.0
            return max_cost, grad, curv, np.atleast_1d(max_ID)

        # if not self.keep_out:
        #     print("max cost from accel constraint is >0: {}".format(max_cost))#,np.linalg.norm(x[:,max_ID])))

        # Compute the gradient
        if doGrad:
            # grad_tmp = self.in_out_scale*2*self.A*x[:,max_ID]
            grad_tmp = self.in_out_scale*2*self.A*x[:,max_ID]
            grad['x'][self.der,max_ID] = grad_tmp[0,:]
            grad['y'][self.der,max_ID] = grad_tmp[1,:]
            grad['z'][self.der,max_ID] = grad_tmp[2,:]

        if doCurv:
            # crv_tmp = np.diagonal(self.in_out_scale*2*self.A)
            crv_tmp = self.in_out_scale*2*self.A
            # crv_tmp = np.diagonal(self.in_out_scale*2*self.A)
            curv['x'][self.der,:,:] = np.dot(crv_tmp[0,:].reshape((3,1)),np.ones((1,nsamp)))
            curv['y'][self.der,:,:] = np.dot(crv_tmp[1,:].reshape((3,1)),np.ones((1,nsamp)))
            curv['z'][self.der,:,:] = np.dot(crv_tmp[2,:].reshape((3,1)),np.ones((1,nsamp)))
            # np.repeat(crv_tmp[0,:].reshape((3,1)),np.size(max_ID),axis=1)#np.dot(crv_tmp[0,:].reshape((3,1)),np.ones((1,np.size(max_ID))))

            # curv['x'][self.der,:] = crv_tmp[0]
            # curv['y'][self.der,:] = crv_tmp[1]
            # curv['z'][self.der,:] = crv_tmp[2]

        return max_cost, grad, curv, np.atleast_1d(max_ID)


class cylinder_constraint(constraintBase):

    def __init__(self,weight,keep_out,der,x1,x2,r,l=0.01,active_seg=None,dynamic_weighting=False,
    doCurv=False,sum_func = True,inflate_buffer=None,custom_weighting=False):
        """

        """

        # Initialise parent case class
        super(cylinder_constraint,self).__init__(weight,keep_out,der,dynamic_weighting,custom_weighting=custom_weighting)

        self.constraint_type = "cylinder"

        self.seg_feasible = True

        # WHich segments it applies to
        self.active_seg = active_seg

        self.x1 = x1
        self.x2 = x2
        self.r = r
        self.l = l # endcap outward radius

        if inflate_buffer == None:
            # Set to radius
            self.inflate_buffer = self.r*1.0
        else:
            self.inflate_buffer = inflate_buffer

        # Geometry computations
        a = x2-x1
        ahat = a/np.linalg.norm(a)
        zhat = np.array([0,0,1])

        rotVec = np.cross(ahat,zhat)# khat * sin(theta)
        rotNorm = np.linalg.norm(rotVec) # sin(theta)
        rotDot = np.dot(ahat,zhat) # cos(theta)

        # Compute the rotation matrix for the end-caps
        # Vector of rotation
        if (rotNorm > 1e-10):
            k = rotVec/rotNorm # normalize to get rotation axis (khat)
        else:
            k = np.array([0,1,0]) * np.sign(rotDot)

        # Angle of rotation
        th = np.arctan2(rotNorm,rotDot) # theta= atan(sin(theta)/cos(theta))

        #Find rotation matrix with Rodrigues's Rotation formula
        kcross = np.matrix([[0, -k[2], k[1]],[ k[2], 0, -k[0]],[ -k[1], k[0], 0]]) # Cross product matrix for the rotation vector
        rot_mat = np.identity(3) + np.sin(th)*kcross + (1-np.cos(th))*(kcross*kcross) # Rodrigues's Rotation formula - fixed 20160502

        self.rot_mat = rot_mat
        self.A = np.array(rot_mat.T*np.diag([1/r**2, 1/r**2, 1/l**2])*rot_mat)

        self.a = a
        self.c = np.dot(a,a) # Distance between end caps squared - precompute

        # print("a is {}\nc is {}\nrot_mat is {}\n A is {}".format(a,self.c,rot_mat,self.A))

        self.sum_func = sum_func

        self.curv_func = doCurv

        if self.keep_out:
            self.in_out_scale = -1
        else:
            self.in_out_scale = 1

        # if not self.curv_func or self.keep_out:
        #     self.cost_type = "squared"
        # else:
        self.cost_type = "not_squared"




    def cost_grad_curv(self, state, seg = 0, doGrad=True, doCurv=False):
        """
        Computes cost, cost gradient and cost curvature of the cylinder constraint. A dictionary for each dimension

        Cost gradient and curvature are returned for use in optimisation steps (e.g. linesearch)

        Args:
            state: The trajectory to evaluate for obstacle costs. np.array with dimensions (nder,nsamp,nseg).
                    A dict for each dimension
            seg:    The segment to calculate the cost and gradient for
            doGrad: boolean to select whether or not to evaluate the gradient
            doCurv: boolean to select whether or not to evaluate the curvature

        Uses:
            self.
            moving: Flag to indicate if the obstacle is moving
            x1: position of the centre of one end the cylinder obstacle (np.array, 3 by nsamp)
            x2: position of the centre of the other end the cylinder obstacle (np.array, 3 by nsamp)
            der: Derivative to operate on
            in_out_scale: +1 or -1 to indicate if the obstacle is a keep out (-1) or keep in (+1) obstacle
            A: Shape matrix for the ellipsoid end caps on the cylinders
            r: the radius of the cylinder
            a: the vecotr between end caps
            c: the norm squared of the vector a

        Outputs:
            cost: the cost for the constraint (single number)
            grad: the cost gradient for the constraint. A dict for each dimension,
                            in each an np.array of length N (number of coefficients for each dimension)
            curv: the cost curvature for the constraint. A dict for each dimension,
                            in each an np.array of length N (number of coefficients for each dimension)
            max_ID: Index of the maximum violation

        """

        # Number of samples in a segment
        nder = state['x'].shape[0]
        nsamp = state['x'].shape[1]

        # Initialise grad: df/dx at the maximum, so for each dimension is an array of the number of derivatives (x includes the derivatives)
        # grad = dict(x=np.zeros((nder,1)),y=np.zeros((nder,1)),z=np.zeros((nder,1)),yaw=np.zeros((nder,1)))
        # curv = dict(x=np.zeros((nder,1)),y=np.zeros((nder,1)),z=np.zeros((nder,1)),yaw=np.zeros((nder,1)))

        grad = dict(x=np.zeros((nder,nsamp)),y=np.zeros((nder,nsamp)),z=np.zeros((nder,nsamp)),yaw=np.zeros((nder,nsamp)))
        curv = dict(x=np.zeros((nder,3,nsamp)),y=np.zeros((nder,3,nsamp)),z=np.zeros((nder,3,nsamp)),yaw=np.zeros((nder,3,nsamp)))

        # Extend vectors for number of states
        x1 = np.repeat(np.reshape(self.x1,(3,1)),nsamp,axis=1)
        x2 = np.repeat(np.reshape(self.x2,(3,1)),nsamp,axis=1)
        a = np.repeat(np.reshape(self.a,(3,1)),nsamp,axis=1)

        # Vectors
        # Vector between end points
        x1_x = np.matrix(np.zeros((3,nsamp)))
        x1_x[0,:] = state['x'][self.der,:,seg] - x1[0,:]
        x1_x[1,:] = state['y'][self.der,:,seg] - x1[1,:]
        x1_x[2,:] = state['z'][self.der,:,seg] - x1[2,:]
        # Vector between end points
        x2_x = np.matrix(np.zeros((3,nsamp)))
        x2_x[0,:] = state['x'][self.der,:,seg] - x2[0,:]
        x2_x[1,:] = state['y'][self.der,:,seg] - x2[1,:]
        x2_x[2,:] = state['z'][self.der,:,seg] - x2[2,:]

        # Determine which cost to apply for each part of the path
        dot_bot = self.long_dot(a,x1_x)
        dot_top = self.long_dot(-a,x2_x) # negative to reverse the direction of a so it comes from the same point as x2_x

        x_endcap1 = x1_x[:,dot_bot<0]
        x_endcap2 = x2_x[:,dot_top<0]
        x_cylinder = x1_x[:,(dot_bot>=0)*(dot_top>=0)]

        ### COSTS ###
        # Ellipsoid endcap costs
        cost_tmp_bot = np.zeros(np.shape(x_endcap1)[1])
        cost_tmp_top = np.zeros(np.shape(x_endcap2)[1])
        for i in range(np.shape(x_endcap1)[1]):
            cost_tmp_bot[i] = self.in_out_scale*(x_endcap1[:,i].T*self.A*x_endcap1[:,i] - 1 + self.in_out_scale*self.inflate_buffer**2/self.r**2)[0,0]
        for i in range(np.shape(x_endcap2)[1]):
            cost_tmp_top[i] = self.in_out_scale*(x_endcap2[:,i].T*self.A*x_endcap2[:,i] - 1 + self.in_out_scale*self.inflate_buffer**2/self.r**2)[0,0]


        # Cylinder
        a2 = np.repeat(np.reshape(self.a,(3,1)),np.shape(x_cylinder)[1],axis=1) #a*ones(1,length(x(1,:))); # Repeat in matrix
        b = self.long_cross(a2,x_cylinder)#This gives |a||x1_x|sin(theta), for each set of points

        #Distance to the line squared is |b|^2 / |a|^2, which leaves
        # |x2_x|^2*sin^2(theta) = d^2
        #Cost function is line d^2 - radius^2 (positive if outside)
        # cost_tmp_mid = self.in_out_scale*(self.long_dot(b,b)/self.c - (self.r**2 + self.inflate_buffer**2)*np.ones((1,np.shape(b)[1])))
        cost_tmp_mid = self.in_out_scale*(self.long_dot(b,b)/self.c/self.r**2 - (1 - self.in_out_scale*self.inflate_buffer**2/self.r**2)*np.ones((1,np.shape(b)[1])))
        # cost_tmp_mid = self.in_out_scale*(self.long_dot(b,b)/self.c/self.r**2  - np.ones((1,np.shape(b)[1])))
        # cost_tmp_mid = in_out_scale.*(dot(b,b)./constraint.c./constraint.r^2 - ones(1,length(b(1,:))));

        # Combine costs
        # Initialise
        cost_tmp = np.zeros(np.shape(x1_x)[1])

        # Add ellipsoid endcap costs
        cost_tmp[dot_bot<0] = cost_tmp_bot
        cost_tmp[dot_top<0] = cost_tmp_top

        # Add cylinder cost
        cost_tmp[(dot_bot>=0)*(dot_top>=0)] = np.reshape(cost_tmp_mid,np.size(cost_tmp_mid))

        # Get out the max cost or summed cost
        if self.sum_func:
            # Summ all costs
            if self.keep_out is True:
                cost_tmp[cost_tmp < 0.0] = 0.0
                max_ID = np.where(cost_tmp>0.0)[0]
            else:
                max_ID = np.where(cost_tmp>-np.inf)[0]
            max_cost = np.sum(cost_tmp)

            # Get cost to check
            max_check = np.amax(cost_tmp) - self.in_out_scale*self.inflate_buffer**2/self.r**2
        else:
            # Maximum violation
            max_cost = np.amax(cost_tmp)
            max_check = max_cost - self.in_out_scale*self.inflate_buffer**2/self.r**2
            if np.sum(np.isclose(cost_tmp,max_cost))<1:
                print("ERROR - can't match max in cost_tmp")
            max_ID = np.atleast_1d(np.where(np.isclose(cost_tmp,max_cost))[0][0])


        # Check non-inflated feasibility
        if max_check <= 0:
            self.seg_feasible = True # TODO - do this for the other constraints
            print("Cylinder Constraint in segment is feasible. Cost check value is {}".format(max_check))
        else:
            print("Cylinder Constraint in segment is NOT feasible. Cost check value  is {}".format(max_check))
            self.seg_feasible = False

        # if np.size(max_ID) > 1:
        #     max_ID = max_ID[0]
        if max_cost <= 0 and not self.sum_func:
            # Constraint not active - no need to compute gradient.
            # Set max ID to negative as a flag
            max_ID = -1
            max_cost = 0.0
            return max_cost, grad, curv, np.atleast_1d(max_ID)


        # Compute the gradient and curvature
        if doGrad:
            a = self.a
            for ID in max_ID:
                if np.dot(self.a,x1_x[:,ID]) < 0: # bottom ellipsoid
                    grad_tmp = (self.in_out_scale*2*self.A*x1_x[:,ID]).T

                    if doCurv:
                        # crv_tmp = np.diagonal(self.in_out_scale*2*self.A)
                        crv_tmp = self.in_out_scale*2*self.A
                elif np.dot(-self.a,x2_x[:,ID]) < 0: # top ellipsoid
                    grad_tmp = (self.in_out_scale*2*self.A*x2_x[:,ID]).T

                    if doCurv:
                        # crv_tmp = np.diagonal(self.in_out_scale*2*self.A)
                        crv_tmp = self.in_out_scale*2*self.A
                else: # Cylinder
                    b = np.cross(a,x1_x[:,ID].T)
                    grad_tmp = self.in_out_scale*2*np.cross(b,a)/self.c
                    if doCurv:
                        # crv_tmp = self.in_out_scale*2/self.c*np.array([a[1]**2+a[2]**2,a[0]**2+a[2]**2,a[0]**2+a[1]**2])
                        crv_tmp = self.in_out_scale*2/self.c*np.array([[a[1]**2+a[2]**2,-a[0]*a[1],-a[0]*a[2]],
                                                                    [-a[0]*a[1],a[0]**2+a[2]**2,-a[1]*a[2]],
                                                                    [-a[0]*a[2],-a[1]*a[2],a[0]**2+a[1]**2]])

                grad['x'][self.der,ID] = grad_tmp[0,0]
                grad['y'][self.der,ID] = grad_tmp[0,1]
                grad['z'][self.der,ID] = grad_tmp[0,2]

                if doCurv:
                    curv['x'][self.der,:,ID] = crv_tmp[0,:]
                    curv['y'][self.der,:,ID] = crv_tmp[1,:]
                    curv['z'][self.der,:,ID] = crv_tmp[2,:]

        return max_cost, grad, curv, np.atleast_1d(max_ID)

    def long_dot(self,v1,v2):
        """ DOt product of a vector of vectors, 3xnsamp """

        nsamp = np.shape(v1)[1]

        dot_out = np.zeros(nsamp)

        for i in range(nsamp):
            dot_out[i] = np.dot(v1[:,i],v2[:,i])

        return dot_out

    def long_cross(self,v1,v2):
        """ Cross product of two vector of vectors, 3xnsamp """

        nsamp = np.shape(v1)[1]

        cross_out = np.zeros((3,nsamp))

        for i in range(nsamp):
            cross_out[:,i] = np.cross(v1[:,i],v2[:,i].transpose())

        return cross_out



class esdf_constraint(constraintBase):

    def __init__(self,weight,esdf,quad_buffer=0.0,inflate_buffer = 0.0,
    dynamic_weighting=False,sum_func=False,custom_weighting=True,feasibility_checker=False):
        """

        """

        # Initialise parent case class
        super(esdf_constraint,self).__init__(weight,dynamic_weighting=dynamic_weighting,custom_weighting=custom_weighting)

        if feasibility_checker:
            self.constraint_type = "esdf_check"
        else:
            self.constraint_type = "esdf"
        # self.keep_out = True

        self.seg_feasible = True

        self.esdf = esdf

        self.quad_buffer = quad_buffer + inflate_buffer
        self.inflate_buffer = inflate_buffer
        print("quad_buffer in esdf is {}".format(quad_buffer))

        self.unknown_is_free = True

        if self.keep_out:
            self.in_out_scale = -1
        else:
            self.in_out_scale = 1

        self.cost_type = "squared"

        self.sum_func = sum_func

    def cost_grad_curv(self, state, seg = 0, doGrad=True, doCurv=False):
        """
        Computes cost, cost gradient and cost curvature of the esdf constraint. A dictionary for each dimension

        Cost gradient and curvature are returned for use in optimisation steps (e.g. linesearch)

        Args:
            state: The trajectory to evaluate for obstacle costs. np.array with dimensions (nder,nsamp,nseg).
                    A dict for each dimension
            seg:    The segment to calculate the cost and gradient for
            doGrad: boolean to select whether or not to evaluate the gradient
            doCurv: boolean to select whether or not to evaluate the curvature

        Uses:
            self.
            map: A stored esdf map of Voxblox format

        Outputs:
            cost: the cost for the constraint (single number)
            grad: the cost gradient for the constraint. A dict for each dimension,
                            in each an np.array of length N (number of coefficients for each dimension)
            curv: the cost curvature for the constraint. A dict for each dimension,
                            in each an np.array of length N (number of coefficients for each dimension)
            max_ID: Index of the maximum violation

        """

        # Number of samples in a segment
        nder = state['x'].shape[0]
        nsamp = state['x'].shape[1]

        # Initialise grad: df/dx at the maximum, so for each dimension is an array of the number of derivatives (x includes the derivatives)
        # grad_out = dict(x=np.zeros((nder,1)),y=np.zeros((nder,1)),z=np.zeros((nder,1)),yaw=np.zeros((nder,1)))
        # curv = dict(x=np.zeros((nder,1)),y=np.zeros((nder,1)),z=np.zeros((nder,1)),yaw=np.zeros((nder,1)))

        grad_out = dict(x=np.zeros((nder,nsamp)),y=np.zeros((nder,nsamp)),z=np.zeros((nder,nsamp)),yaw=np.zeros((nder,nsamp)))
        curv = dict(x=np.zeros((nder,3,nsamp)),y=np.zeros((nder,3,nsamp)),z=np.zeros((nder,3,nsamp)),yaw=np.zeros((nder,3,nsamp)))

        # get x, y, z trajectories
        x = state['x'][0,:,seg]
        y = state['y'][0,:,seg]
        z = state['z'][0,:,seg]

        # Create query points
        query = np.matrix(np.zeros([3,x.size]),dtype='double')
        dist = np.matrix(np.zeros((np.shape(query)[1],1)),dtype='double')
        obs = np.matrix(np.zeros((np.shape(query)[1],1)),dtype='int32')

        # load in x, y, z points
        query[0,:] = np.around(x,4)
        query[1,:] = np.around(y,4)
        query[2,:] = np.around(z,4)

        # Query the database
        # if doGrad:
        grad = np.matrix(np.zeros(np.shape(query)),dtype='double')
        self.esdf.getDistanceAndGradientAtPosition(query, dist, grad, obs)
        # print("Getting gradient: {}".format(grad))
        # else:
        #     self.esdf.getDistanceAtPosition(query, dist, obs)
        # Add buffer on quad:
        dist -= self.quad_buffer

        if self.unknown_is_free:
            dist[obs != 1] = self.quad_buffer - self.inflate_buffer#2.0

        # else:
            # dist[obs != 1] = -2.0


        if self.sum_func:
            # Sum all costs
            # dist[-dist < 0.0] = 0.0
            max_cost = np.sum(-dist)
            max_check = np.amax(-dist) - self.inflate_buffer
            max_ID = np.where(dist>-np.inf)[0]
            # max_ID = np.where(-dist>0.0)[0]
        else:
            # Maximum violation
            max_cost = np.amax(-dist) # Negative as violations are negative distances
            max_check = max_cost - self.inflate_buffer
            # cost_tmp = -np.sign(dist)*dist**2
            max_ID = np.where((-dist)==max_cost)[0][0]

        # Check non-inflated feasibility
        if max_check <= 0:
            self.seg_feasible = True # TODO - do this for the other constraints
            # print("ESDF Constraint in segment is feasible. Distance is {} (m), gradient is: {}".format(max_check, -grad[:,max_ID]))
            print("ESDF Constraint in segment is feasible. Distance is {} (m)".format(max_check))
        else:
            print("ESDF Constraint in segment is NOT feasible. Distance is {} (m)".format(max_check))
            self.seg_feasible = False

        if max_cost <= 0 and not self.sum_func:
            # Constraint not active - no need to compute gradient.
            # Set max ID to negative as a flag
            # print("Max esdf cost, {}, is < 0".format(max_cost))
            max_ID = -1
            max_cost = 0.0
            return max_cost, grad_out, curv, np.atleast_1d(max_ID)
        else:
            # max_cost = max_cost
            # print("Max cost from ESDF is {}\n".format(max_cost))
            pass


        # Compute the gradient
        if doGrad:
            # print("doing grad max_cost is {}".format(max_cost))
            grad_tmp = -grad[:,max_ID]
            grad_out['x'][0,max_ID] = grad_tmp[0,:]
            grad_out['y'][0,max_ID] = grad_tmp[1,:]
            grad_out['z'][0,max_ID] = grad_tmp[2,:]
            # print("grad at max is {}".format(grad_out))

        return max_cost, grad_out, curv, np.atleast_1d(max_ID)

def main():
    from astro import traj_qr

    waypoints = dict()
    waypoints['x'] = np.zeros([5,2])
    waypoints['y'] = np.zeros([5,2])
    waypoints['z'] = np.zeros([5,2])
    waypoints['yaw'] = np.zeros([3,2])

    waypoints['x'][0,:] = np.array([-1.0,1.0])
    waypoints['y'][0,:] = np.array([0.4,0.5])
    waypoints['z'][0,:] = np.array([-0.6,-0.5])

    traj = traj_qr.traj_qr(waypoints,seed_avg_vel=1.0)

    # traj.initial_guess()
    # traj.get_trajectory()

    traj.run_astro()

    weight = 1.0
    x1 = np.array([0.0,0.0,-0.5])
    x2 = np.array([0.0,1.0,-0.5])
    r = 0.3
    l = 0.01
    x1 = np.array([-10.0,0.0,-0.])
    x2 = np.array([10.0,0.0,-0.])
    x1 = np.array([-10.0,-2.0,-4.])
    x2 = np.array([10.0,2.0,4.])
    r = 1.0
    l = 0.01
    keep_out = False
    der = 0

    constr = cylinder_constraint(weight,keep_out,der,x1,x2,r,l,active_seg = 0,dynamic_weighting=False,doCurv=True,sum_func = False)
    nsamp = 50
    cost = np.zeros(nsamp)
    state_test = dict()
    for key in traj.state.keys():
        state_test[key] = np.zeros((1,nsamp,1))

    state_test_store = np.array([np.linspace(0.5,0.5,nsamp),
                                np.linspace(-3.,3,nsamp),
                                np.linspace(-0.5,-0.9,nsamp)])

    for i in range(nsamp):
        state_test['x'][0,:,0] = state_test_store[0,i]
        state_test['y'][0,:,0] = state_test_store[1,i]
        state_test['z'][0,:,0] = state_test_store[2,i]

        cost[i], grad, curv, max_ID = constr.cost_grad_curv(state_test, 0, doGrad=True, doCurv=True)


    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D

    # PLOT FIGURE
    fig = plt.figure(figsize=(10,10))
    fig.suptitle('Static 3D Plot', fontsize=20)
    ax = fig.add_subplot(111,projection='rectilinear')
    ax.set_xlabel('dist', fontsize=14)
    ax.set_ylabel('cost', fontsize=14)

    # plot Trajectory
    ax.plot(state_test_store[1,:],cost)

    # plot waypoints
    plt.show()

    import pdb; pdb.set_trace()
    cost, grad, curv = constr.compute_constraint_cost_grad_curv(traj.state, traj.poly.state_scaled, doGrad=True, doCurv=True)



    import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    A = np.identity(3)
    A[0,0] = 1/0.2**2
    A[1,1] = 1/0.5**2
    A[2,2] = 1/0.3**2
    weight = 1.0
    x0 = np.array([0.0,0.5,-0.5])
    keep_out = True
    der = 0
    rot_mat = np.identity(3)

    constr = ellipsoid_constraint(weight,keep_out,der,x0,A,rot_mat)

    cost, grad, curv = constr.compute_constraint_cost_grad_curv(traj.state, traj.poly.state_scaled, doGrad=True, doCurv=True)

    import pdb; pdb.set_trace()

    test_list = [ellipsoid_constraint(weight,keep_out,der,x0,A),ellipsoid_constraint(weight*2.0,keep_out,der,x0+1.0,A)]


if __name__ == '__main__':
    main()
