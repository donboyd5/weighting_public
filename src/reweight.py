# -*- coding: utf-8 -*-
"""
Reweight class

@author: donbo
"""

from __future__ import print_function, unicode_literals
import numpy as np


class Reweight(object):
    """
    Class for reweighting microdata file.

    More documentation here.
    """

    def __init__(self, cc):
        """Define values needed on initialization."""
        self.cc = cc
        self.n = cc.shape[0]
        self.m = cc.shape[1]

    def objective(self, x):
        """Calculate objective function."""
        return np.sum((x - 1)**2)

    def gradient(self, x):
        """Calculate gradient of objective function."""
        return 2 * x - 2

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
        return np.dot(x, self.cc)

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
        return self.cc.T[row, col]

    def jacobianstructure(self):
        """
        Define sparse structure of Jacobian.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return np.nonzero(self.cc.T)

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
        H = np.full(self.n, 2) * obj_factor
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
        hidx = np.arange(0, self.n, dtype='int64')
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
