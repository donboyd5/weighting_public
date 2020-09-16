# -*- coding: utf-8 -*-
"""
Reweight class
  https://pythonhosted.org/ipopt/reference.html#reference

@author: donbo
"""

from __future__ import print_function, unicode_literals
import os
import numpy as np
import ipopt


class Reweight(ipopt.problem):
    """
    Class for reweighting microdata file.

    More documentation here.
    """

    def __init__(self, wh, xmat, targets):
        """Define values needed on initialization."""
        self._wh = wh
        # self._wh = np.array(wh)
        self._xmat = xmat
        self._targets = targets  # length must be _m, flatten if needed
        self._n = xmat.shape[0]
        self._m = xmat.shape[1]

    def reweight(self,
                 xlb=0.1,
                 xub=100,
                 crange=.03,
                 max_iter=100,
                 ccgoal=1,
                 objgoal=100,
                 quiet=True):
        r"""
        Build and solve the reweighting NLP.

        Good general settings seem to be:
            get_ccscale - use ccgoal=1, method='mean'
            get_objscale - use xbase=1.2, objgoal=100
            no other options set, besides obvious ones

        Important resources:
            https://pythonhosted.org/ipopt/reference.html#reference
            https://coin-or.github.io/Ipopt/OPTIONS.html
            ..\cyipopt\ipopt\ipopt_wrapper.py to see code from cyipopt author

        Returns
        -------
        x : TYPE
            DESCRIPTION.
        info : TYPE
            DESCRIPTION.

        """
        # constraint coefficients (constant)
        # cc = self._xmat * self._wh[:, None]
        # cc = self._xmat * self._wh
        cc = (self._xmat.T * self._wh).T

        # scale constraint coefficients and targets
        ccscale = self.get_ccscale(cc, ccgoal=ccgoal, method='mean')
        # print(ccscale)
        # ccscale = 1
        cc = cc * ccscale  # mult by scale to have avg derivative meet our goal
        targets = self._targets * ccscale

        # IMPORTANT: define callbacks AFTER we have scaled cc and targets
        # because callbacks must be initialized with scaled cc
        self.callbacks = Reweight_callbacks(cc, quiet)

        # x vector starting values, and lower and upper bounds
        x0 = np.ones(self._n)
        lb = np.full(self._n, xlb)
        ub = np.full(self._n, xub)

        # constraint lower and upper bounds
        cl = targets - abs(targets) * crange
        cu = targets + abs(targets) * crange

        nlp = ipopt.problem(
            n=self._n,
            m=self._m,
            problem_obj=self.callbacks,
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu)

        # objective function scaling
        objscale = self.get_objscale(objgoal=objgoal, xbase=1.2)
        # print(objscale)
        nlp.addOption('obj_scaling_factor', objscale)  # multiplier

        # define additional options as a dict
        opts = {
            'print_level': 5,
            'file_print_level': 5,
            'jac_d_constant': 'yes',
            'hessian_constant': 'yes',
            'max_iter': max_iter,
            'mumps_mem_percent': 100,  # default 1000
            'linear_solver': 'mumps',
            }

        # TODO: check against already set options, etc. see ipopt_wrapper.py
        for option, value in opts.items():
            nlp.addOption(option, value)

        # outfile = 'test4.out'
        # if os.path.exists(outfile):
        #     os.remove(outfile)
        # nlp.addOption('output_file', outfile)
        # nlp.addOption('derivative_test', 'first-order')  # second-order

        # nlp_scaling_method: default gradient-based
        # equilibration-based needs MC19
        # nlp.addOption('nlp_scaling_method', 'equilibration-based')
        # nlp.addOption('nlp_scaling_max_gradient', 1e-4)  # 100 default
        # nlp.addOption('mu_strategy', 'adaptive')  # not good
        # nlp.addOption('mehrotra_algorithm', 'yes')  # not good
        # nlp.addOption('mumps_mem_percent', 100)  # default 1000
        # nlp.addOption('mumps_pivtol', 1e-4)  # default 1e-6; 1e-2 is SLOW
        # nlp.addOption('mumps_scaling', 8)  # 77 default

        x, info = nlp.solve(x0)
        return x, info

    def get_ccscale(self, cc, ccgoal, method='mean'):
        """
        Create multiplicative scaling vector ccscale.

        For scaling the constraint coefficients and the targets.

        Parameters
        ----------
        ccgoal : TYPE
            DESCRIPTION.
        method : TYPE, optional
            DESCRIPTION. The default is 'mean'.

        Returns
        -------
        ccscale vector.

        """
        # use mean or median as the denominator
        if(method == 'mean'):
            denom = cc.sum(axis=0) / cc.shape[0]
        elif(method == 'median'):
            denom = np.median(cc, axis=0)

        ccscale = np.absolute(ccgoal / denom)
        # ccscale = ccscale / ccscale
        return ccscale

    def get_objscale(self, objgoal, xbase):
        """
        Calculate objective scaling factor.

        Returns
        -------
        objscale : TYPE
            DESCRIPTION.

        """
        xbase = np.full(self._n, xbase)
        objbase = self.callbacks.objective(xbase)
        objscale = objgoal / objbase
        # convert to python float from Numpy float as that is what
        # cyipopt requires
        objscale = objscale.item()
        # print(objscale)
        return objscale




class Reweight_callbacks(object):
    """
    Must have:
        objective
        constraints
        gradient
        jacobian
        jacobianstructure
        hessian
        hessianstructure
        intermediate
    """
    def __init__(self, cc, quiet):
        self._cc = cc
        self._n = cc.shape[0]
        self._m = cc.shape[1]
        self._quiet = quiet

    def objective(self, x):
        """Calculate objective function."""
        return np.sum((x - 1)**2)

    def constraints(self, x):
        """
        Calculate constraints for a given set of x values.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return np.dot(x, self._cc)

    def gradient(self, x):
        """Calculate gradient of objective function."""
        return 2 * x - 2


    def jacobian(self, x):
        """
        Calculate nonzero elements of Jacobian, return in sparse format.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        row, col = self.jacobianstructure()
        return self._cc.T[row, col]

    def jacobianstructure(self):
        """
        Define sparse structure of Jacobian.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return np.nonzero(self._cc.T)

    def hessian(self, x, lagrange, obj_factor):
        """
        Calculate the Hessian matrix in sparse form.

        In this problem the Hessian is a constant 2 (2nd derivative of
        objective) with nothing added for the constraints, multiplied by the
        internal Ipopt variable obj_factor. Ipopt also requires that its
        internal variable lagrange be passed to this function, although it
        is not needed in this problem.


        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        lagrange : TYPE
            DESCRIPTION.
        obj_factor : TYPE
            DESCRIPTION.

        Returns
        -------
        H : TYPE
            DESCRIPTION.

        """
        H = np.full(self._n, 2) * obj_factor
        return H

    def hessianstructure(self):
        """
        Row and column indexes of nonzero elements of hessian.

        A tuple of two arrays: one for row indexes and one for column indexes.
        In this problem the hessian has nonzero elements only on the diagonal
        so this returns an array of row indexes of arange(0, n) where n is
        the number of rows (and columns) in the square hessian matrix, and
        the same array for the column index column indexes.

        These indexes must correspond to the order of the elements returned
        from the hessian function. That requirement is enforced in that
        function.

        Note: The cyipopt default hessian structure is a lower triangular
        matrix, so if that is what the hessian function produces, this
        function is not needed.

        Returns
        -------
        hstruct : tuple:
            First array has row indexes of nonzero elements of the hessian
            matrix.
            Second array has column indexes for these elements.

        """
        hidx = np.arange(0, self._n, dtype='int64')
        hstruct = (hidx, hidx)
        return hstruct

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):
        """
        Print intermediate results after each iteration.

        Parameters
        ----------
        alg_mod : TYPE
            DESCRIPTION.
        iter_count : TYPE
            DESCRIPTION.
        obj_value : TYPE
            DESCRIPTION.
        inf_pr : TYPE
            DESCRIPTION.
        inf_du : TYPE
            DESCRIPTION.
        mu : TYPE
            DESCRIPTION.
        d_norm : TYPE
            DESCRIPTION.
        regularization_size : TYPE
            DESCRIPTION.
        alpha_du : TYPE
            DESCRIPTION.
        alpha_pr : TYPE
            DESCRIPTION.
        ls_trials : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # print("Objective value at iteration #%d is - %g"
        #     % (iter_count, obj_value))
        if(not self._quiet):
            print("Iter, obj, infeas #%d %g %g"
                  % (iter_count, obj_value, inf_pr))
