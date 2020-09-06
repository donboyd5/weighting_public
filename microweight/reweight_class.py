# -*- coding: utf-8 -*-
"""
Reweight class

@author: donbo
"""

import numpy as np
import timeit

a = np.array([1., 5.])
a.sum()
np.square(a).sum()
# a.square() NO
#  %timeit np.square(a)

def objective(x):
        #
        # The callback for calculating the objective
        #
        return np.square(x).sum()
objective(a)

class Reweight(object):
    def __init__(self):
        pass

    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        return np.square(x).sum()

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        return np.array([
                    x[0] * x[3] + x[3] * np.sum(x[0:3]),
                    x[0] * x[3],
                    x[0] * x[3] + 1.0,
                    x[0] * np.sum(x[0:3])
                    ])

    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #
        return np.array((np.prod(x), np.dot(x, x)))

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #
        return np.concatenate((np.prod(x) / x, 2*x))

    def hessianstructure(self):
        #
        # The structure of the Hessian
        # Note:
        # The default hessian structure is of a lower triangular matrix. Therefore
        # this function is redundant. I include it as an example for structure
        # callback.
        #

        return np.nonzero(np.tril(np.ones((4, 4))))

    def hessian(self, x, lagrange, obj_factor):
        #
        # The callback for calculating the Hessian
        #
        H = obj_factor*np.array((
                (2*x[3], 0, 0, 0),
                (x[3],   0, 0, 0),
                (x[3],   0, 0, 0),
                (2*x[0]+x[1]+x[2], x[0], x[0], 0)))

        H += lagrange[0]*np.array((
                (0, 0, 0, 0),
                (x[2]*x[3], 0, 0, 0),
                (x[1]*x[3], x[0]*x[3], 0, 0),
                (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))

        H += lagrange[1]*2*np.eye(4)

        row, col = self.hessianstructure()

        return H[row, col]

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

        #
        # Example for the use of the intermediate callback.
        #
        # print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))
        print("Iter, obj, infeas #%d %g %g" % (iter_count, obj_value, inf_pr))



class hs071(object):
    def __init__(self):
        pass

    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        return x[0] * x[3] * np.sum(x[0:3]) + x[2]

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        return np.array([
                    x[0] * x[3] + x[3] * np.sum(x[0:3]),
                    x[0] * x[3],
                    x[0] * x[3] + 1.0,
                    x[0] * np.sum(x[0:3])
                    ])

    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #
        return np.array((np.prod(x), np.dot(x, x)))

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #
        return np.concatenate((np.prod(x) / x, 2*x))

    def hessianstructure(self):
        #
        # The structure of the Hessian
        # Note:
        # The default hessian structure is of a lower triangular matrix. Therefore
        # this function is redundant. I include it as an example for structure
        # callback.
        #

        return np.nonzero(np.tril(np.ones((4, 4))))

    def hessian(self, x, lagrange, obj_factor):
        #
        # The callback for calculating the Hessian
        #
        H = obj_factor*np.array((
                (2*x[3], 0, 0, 0),
                (x[3],   0, 0, 0),
                (x[3],   0, 0, 0),
                (2*x[0]+x[1]+x[2], x[0], x[0], 0)))

        H += lagrange[0]*np.array((
                (0, 0, 0, 0),
                (x[2]*x[3], 0, 0, 0),
                (x[1]*x[3], x[0]*x[3], 0, 0),
                (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))

        H += lagrange[1]*2*np.eye(4)

        row, col = self.hessianstructure()

        return H[row, col]

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

        #
        # Example for the use of the intermediate callback.
        #
        # print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))
        print("Iter, obj, infeas #%d %g %g" % (iter_count, obj_value, inf_pr))

