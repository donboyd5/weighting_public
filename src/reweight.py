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
        self._xmat = xmat
        self._targets = targets
        self._n = xmat.shape[0]
        self._m = xmat.shape[1]

    def reweight(self):
        """


        Returns
        -------
        x : TYPE
            DESCRIPTION.
        info : TYPE
            DESCRIPTION.

        """
        # call the solver
        # https://pythonhosted.org/ipopt/reference.html#reference
        # constraint coefficients (constant)
        cc = self._xmat * self._wh[:, None]

        # create multiplicative scaling vector ccscale, for scaling
        # the constraint coefficients and the targets
        ccgoal = 1
        # use mean or median as the denominator
        denom = ccgoal / (cc.sum(axis=0) / cc.shape[0])  # mean
        # denom = np.median(cc, axis = 0)
        ccscale = np.absolute(ccgoal / denom)

        cc = cc * ccscale  # mult by scale to have avg derivative meet our goal
        targets = self._targets * ccscale

        # x vector: starting values and lower and upper bounds
        x0 = np.ones(self._n)
        lb = np.full(self._n, 0.1)
        ub = np.full(self._n, 100)

        # constraint lower and upper bounds
        cl = targets * .97
        cu = targets * 1.03

        rwobject = Reweight_callbacks(cc)

        nlp = ipopt.problem(
            n=self._n,
            m=self._m,
            problem_obj=rwobject,
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu)

        # nlp.addOption('file_print_level', 6)
        # outfile = 'test.out'
        # if os.path.exists(outfile):
        #     os.remove(outfile)
        # nlp.addOption('output_file', outfile)
        nlp.addOption('jac_d_constant', 'yes')
        nlp.addOption('hessian_constant', 'yes')
        nlp.addOption('max_iter', 100)

        xtest = np.full(x0.size, 1.15)
        objbase = rwobject.objective(xtest)
        # objbase
        objgoal = 10
        objscale = objgoal / objbase
        objscale = objscale.item()
        nlp.addOption('obj_scaling_factor', objscale)  # multiplier

        x, info = nlp.solve(x0)

        return x, info


















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
    def __init__(self, cc):
        self._cc = cc
        self._n = cc.shape[0]
        self._m = cc.shape[1]

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
        print("Iter, obj, infeas #%d %g %g" % (iter_count, obj_value, inf_pr))
