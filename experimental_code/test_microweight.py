# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 05:16:40 2020

@author: donbo
"""

# %% notes
# TODO
#   classes within classes; memory management
#   weight objective by size of weight


# %% imports
from __future__ import print_function, unicode_literals
import os
import numpy as np
import pandas as pd
from numpy.random import seed

import scipy  # needed for sparse matrices
from scipy.optimize import lsq_linear
# from scipy.optimize import least_squares  # nonlinear least squares

from timeit import default_timer as timer

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

import ipopt

import src.microweight as mw
import src.make_test_problems as mtp
import src.reweight as rw


# %% test geoweight problems of arbitrary size
mtp.Problem.help()

p = mtp.Problem(h=100, s=5, k=2)
# p = mtp.Problem(h=20000, s=30, k=10)  # moderate-sized problem, < 1 min

# I don't think our problems for a single AGI range will get bigger
# than the one below:
#   30k tax records, 50 states, 30 characteristics (targets) per state
# but problems will be harder to solve with real data
# p = mtp.Problem(h=30000, s=50, k=30) # took 31 mins on my computer

mw.Microweight.help()

g1 = mw.Microweight(p.wh, p.xmat, p.targets)

# look at the inputs
g1.wh
g1.xmat
g1.geotargets

# solve for state weights
g1.geoweight()

# examine results
g1.elapsed_minutes
g1.result  # this is the result returned by the solver
dir(g1.result)
g1.result.cost  # objective function value at optimum
g1.result.message

# optimal values
g1.beta_opt  # beta coefficients, s x k
g1.delta_opt  # delta constants, 1 x h
g1.whs_opt  # state weights
g1.geotargets_opt

# ensure that results are correct
# did we hit targets? % differences:
pdiff = (g1.geotargets_opt - g1.geotargets) / g1.geotargets * 100
pdiff
np.square(pdiff).sum()  # would like it to be approx zero

# do state weights sum to national weights for every household?
g1.whs_opt  # optimal state weights
wh_opt = g1.whs_opt.sum(axis=1)
wh_opt  # sum of optimal state weights for each household
np.square(wh_opt - g1.wh).sum()  # should be approx zero


# %% test linear least squares
# here we test ability to hit national (not state) targets, creating
# weights that minimize sum of squared differences from targets

p = mtp.Problem(h=10, s=1, k=2)
p = mtp.Problem(h=100, s=1, k=4)
p = mtp.Problem(h=3000, s=1, k=10)
p = mtp.Problem(h=30000, s=1, k=20)
p = mtp.Problem(h=300000, s=1, k=30)
p = mtp.Problem(h=500000, s=1, k=50)

seed(1)
r = np.random.randn(p.targets.size) / 50  # random normal
q = [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]
np.quantile(r, q)
targets = (p.targets * (1 + r)).flatten()
diff_weights = np.where(targets != 0, 100 / targets, 1)

# we are solving Ax = b, where
#   b are the targets and
#   A x multiplication gives calculated targets
# using sparse matrix As instead of A

b = targets * diff_weights
b

wmat = p.xmat * diff_weights
At = np.multiply(p.wh.reshape(p.h, 1), wmat)
# At = np.multiply(p.wh.reshape(p.h, 1), p.xmat)
A = At.T
As = scipy.sparse.coo_matrix(A)

# calculate starting percent differences
Atraw = np.multiply(p.wh.reshape(p.h, 1), p.xmat)
# compare sdiff -- starting differences - to res.fun
sdiff = (np.dot(np.full(p.h, 1), Atraw) - targets) / targets * 100
sdiff

lb = np.full(p.h, 0.25)
ub = np.full(p.h, 4)

# lb = np.full(p.h, 0)
# ub = np.full(p.h, np.inf)

p.h
p.k

start = timer()
res = lsq_linear(As, b, bounds=(lb, ub),
                 method='trf',
                 tol=1e-6,
                 #lsmr_tol='auto',
                 max_iter=50, verbose=2)
end = timer()

print(end - start)

np.abs(sdiff).max()
np.abs(res.fun).max()

np.square(sdiff).sum()
np.square(res.fun).sum()

# compare to cost function
np.square(sdiff).sum() / 2
res.cost
# np.square(res.fun).sum() / 2

sdiff
res.fun
q = [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]
np.quantile(res.x, q)

n, bins, patches = plt.hist(res.x, 50, density=True, facecolor='g', alpha=0.75)


res.x
res.cost
res.fun
b
pdiff = res.fun / b * 100
np.abs(pdiff)
np.abs(pdiff).max()

res.x

targets_calc = np.dot(res.x, Atraw)
targets
targets_calc
targets_calc - targets
(targets_calc - targets) / targets * 100
pdiff


# %% reweighting classes

# https://pythonhosted.org/ipopt/tutorial.html

# The constructor of the ipopt.problem class requires
#   n: the number of variables in the problem,
#   m: the number of constraints in the problem,
#   lb and ub: lower and upper bounds on the variables,
#   cl and cu: lower and upper bounds of the constraints
# problem_obj is an object whose methods implement the objective,
# gradient, constraints, jacobian, and hessian of the problem.

# The intermediate() method if defined is called every iteration.
# The jacobianstructure() and hessianstructure() methods if defined
# should return a tuple which lists the non zero values of the jacobian
# and hessian matrices respectively.
# If not defined then these matrices are assumed to be dense.
# The jacobian() and hessian() methods should return the non zero values
# as a flattened array.
# If the hessianstructure() method is not defined then the hessian()
# method should return a lower traingular matrix (flattened).


class Rw1(object):
    def __init__(self, cc):
        self.cc = cc
        self.n = cc.shape[0]
        self.m = cc.shape[1]

    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        return np.sum((x - 1)**2)

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        return 2 * x - 2

    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #
        return np.dot(x, self.cc)

    # def jacobian(self, x):
    #     #
    #     # The callback for calculating the Jacobian
    #     #  return nonzero elements, flattened
    #     #
    #     return self.cc[self.cc != 0]

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #  return nonzero elements, flattened
        #
        return self.cc

    # def jacobianstructure(self):
    #     return np.nonzero(self.cc)

    # def hessian(self, x, lagrange, obj_factor):
    #     #
    #     # The callback for calculating the Hessian
    #     #
    #     H = np.full(self.n, 2) * obj_factor
    #     return H

    # def hessianstructure(self):
    #     #
    #     # The structure of the Hessian
    #     # Note:
    #     # The default hessian structure is of a lower triangular matrix. Therefore
    #     # this function is redundant. I include it as an example for structure
    #     # callback.
    #     #
    #     # np.tril returns the lower triangle (including diagonal) of an array.
    #     # numpy.nonzero returns the indices of the elements that are non-zero.
    #     #  returns a tuple of arrays, one for each dimension of a,
    #     #  containing the indices of the non-zero elements in that dimension
    #     #  The values in a are always tested and returned in
    #     #  row-major, C-style order.
    #     #
    #     return np.nonzero(np.eye(self.n))

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



class Rw2(object):
    def __init__(self):
        pass

    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        return np.sum((x - 1)**2)

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        return 2 * x - 2

    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #
        return np.dot(x, cc)

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #  return nonzero elements, flattened
        #
        return cc[cc != 0]

    def jacobianstructure(self):
        return np.nonzero(cc)

    def hessian(self, x, lagrange, obj_factor):
        #
        # The callback for calculating the Hessian
        #
        H = np.full(x0.size, 2) * obj_factor
        return H

    def hessianstructure(self):
        #
        # The structure of the Hessian
        # Note:
        # The default hessian structure is of a lower triangular matrix. Therefore
        # this function is redundant. I include it as an example for structure
        # callback.
        #
        # np.tril returns the lower triangle (including diagonal) of an array.
        # numpy.nonzero returns the indices of the elements that are non-zero.
        #  returns a tuple of arrays, one for each dimension of a,
        #  containing the indices of the non-zero elements in that dimension
        #  The values in a are always tested and returned in
        #  row-major, C-style order.
        #
        return np.nonzero(np.eye(x0.size))

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


class Rw3(ipopt.problem):
    def __init__(self, cc):
        self._cc = cc
        self._n = cc.shape[0]
        self._m = cc.shape[1]

        #
        # The constraint functions are bounded from below by zero.
        #
        # cl = np.zeros(2*self._m)

        # super(Rw3, self).__init__(
        #                     2*self._m,
        #                     2*self._m,
        #                     cl=cl
        #                     )

        #
        # Set solver options
        #
        # self.addOption('derivative_test', 'second-order')
        # self.addOption('jac_d_constant', 'yes')
        # self.addOption('hessian_constant', 'yes')
        # self.addOption('mu_strategy', 'adaptive')
        self.addOption('max_iter', 100)
        self.addOption('tol', 1e-8)

    # def solve(self, _lambda):

    #     x0 = np.concatenate((np.zeros(m), np.ones(m)))
    #     self._lambda = _lambda
    #     x, info = super(lasso, self).solve(x0)

    #     return x[:self._m]

    def objective(self, x):

        w = x[:self._m].reshape((-1, 1))
        u = x[self._m:].reshape((-1, 1))

        return np.linalg.norm(self._y - np.dot(self._A, w))**2/2 + self._lambda * np.sum(u)

    def constraints(self, x):

        w = x[:self._m].reshape((-1, 1))
        u = x[self._m:].reshape((-1, 1))

        return np.vstack((u + w,  u - w))

    def gradient(self, x):

        w = x[:self._m].reshape((-1, 1))

        g = np.vstack((np.dot(-self._A.T, self._y - np.dot(self._A, w)), self._lambda*np.ones((self._m, 1))))

        return g

    def jacobianstructure(self):

        #
        # Create a sparse matrix to hold the jacobian structure
        #
        return np.nonzero(np.tile(np.eye(self._m), (2, 2)))

    def jacobian(self, x):

        I = np.eye(self._m)

        J = np.vstack((np.hstack((I, I)), np.hstack((-I, I))))

        row, col = self.jacobianstructure()

        return J[row, col]

    def hessianstructure(self):

        h = np.zeros((2*self._m, 2*self._m))
        h[:self._m, :self._m] = np.tril(np.ones((self._m, self._m)))

        #
        # Create a sparse matrix to hold the hessian structure
        #
        return np.nonzero(h)

    def hessian(self, x, lagrange, obj_factor):

        H = np.zeros((2*self._m, 2*self._m))
        H[:self._m, :self._m] = np.tril(np.tril(np.dot(self._A.T, self._A)))

        row, col = self.hessianstructure()

        return obj_factor*H[row, col]


class Rw4(object):
    def __init__(self, cc):
        self.cc = cc
        self.n = cc.shape[0]
        self.m = cc.shape[1]

    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        return np.sum((x - 1)**2)

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        return 2 * x - 2

    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #
        return np.dot(x, self.cc)

    # def jacobian(self, x):
    #     #
    #     # The callback for calculating the Jacobian
    #     #  return nonzero elements, flattened
    #     #
    #     return self.cc[self.cc != 0]

    # def jacobianstructure(self):
    #     return np.nonzero(self.cc)

    # def jacobian(self, x):
    #     #
    #     # The callback for calculating the Jacobian
    #     #  return nonzero elements, flattened
    #     #
    #     row, col = self.jacobianstructure()
    #     return self.cc[row, col]

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #  return nonzero elements, flattened
        #
        return self.cc

    def hessian(self, x, lagrange, obj_factor):
        #
        # The callback for calculating the Hessian
        #
        H = np.full(self.n, 2) * obj_factor
        return H

    def hessianstructure(self):
        #
        # The structure of the Hessian
        # Note:
        # The default hessian structure is of a lower triangular matrix. Therefore
        # this function is redundant. I include it as an example for structure
        # callback.
        #
        # np.tril returns the lower triangle (including diagonal) of an array.
        # numpy.nonzero returns the indices of the elements that are non-zero.
        #  returns a tuple of arrays, one for each dimension of a,
        #  containing the indices of the non-zero elements in that dimension
        #  The values in a are always tested and returned in
        #  row-major, C-style order.
        #
        return np.nonzero(np.eye(self.n))

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


# TODO:
#   get sparse jacobian working
#   why is inf_pr from callback different from terminal??
#   compile with hsl

class Rw5(object):
    def __init__(self, cc):
        self.cc = cc
        self.n = cc.shape[0]
        self.m = cc.shape[1]

    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        return np.sum((x - 1)**2)

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        return 2 * x - 2

    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #
        return np.dot(x, self.cc)

    # def jacobian(self, x):
    #     #
    #     # The callback for calculating the Jacobian
    #     #  return nonzero elements, flattened
    #     #
    #     return self.cc[self.cc != 0]

    # def jacobianstructure(self):
    #     return np.nonzero(self.cc)

    # def jacobian(self, x):
    #     #
    #     # The callback for calculating the Jacobian
    #     #  return nonzero elements, flattened
    #     #
    #     row, col = self.jacobianstructure()
    #     return self.cc[row, col]

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #  return nonzero elements, flattened
        #
        return self.cc.T

    def hessian(self, x, lagrange, obj_factor):
        #
        # The callback for calculating the Hessian
        #
        H = np.full(self.n, 2) * obj_factor
        return H

    def hessianstructure(self):
        #
        # The structure of the Hessian
        # Note:
        # The default hessian structure is of a lower triangular matrix. Therefore
        # this function is redundant. I include it as an example for structure
        # callback.
        #
        # np.tril returns the lower triangle (including diagonal) of an array.
        # numpy.nonzero returns the indices of the elements that are non-zero.
        #  returns a tuple of arrays, one for each dimension of a,
        #  containing the indices of the non-zero elements in that dimension
        #  The values in a are always tested and returned in
        #  row-major, C-style order.
        #
        return np.nonzero(np.eye(self.n))

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


class Rw6(object):
    def __init__(self, cc):
        self.cc = cc
        self.n = cc.shape[0]
        self.m = cc.shape[1]

    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        return np.sum((x - 1)**2)

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        return 2 * x - 2

    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #
        return np.dot(x, self.cc)

    # def jacobian(self, x):
    #     #
    #     # The callback for calculating the Jacobian
    #     #  return nonzero elements, flattened
    #     #
    #     return self.cc[self.cc != 0]

    def jacobianstructure(self):
        return np.nonzero(self.cc.T)

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #  return nonzero elements, flattened
        #
        row, col = self.jacobianstructure()
        return self.cc.T[row, col]

    # def jacobian(self, x):
    #     #
    #     # The callback for calculating the Jacobian
    #     #  return nonzero elements, flattened
    #     #
    #     return self.cc.T

    def hessian(self, x, lagrange, obj_factor):
        #
        # The callback for calculating the Hessian
        #
        H = np.full(self.n, 2) * obj_factor
        return H

    def hessianstructure(self):
        #
        # The structure of the Hessian
        # Note:
        # The default hessian structure is of a lower triangular matrix. Therefore
        # this function is redundant. I include it as an example for structure
        # callback.
        #
        # np.tril returns the lower triangle (including diagonal) of an array.
        # numpy.nonzero returns the indices of the elements that are non-zero.
        #  returns a tuple of arrays, one for each dimension of a,
        #  containing the indices of the non-zero elements in that dimension
        #  The values in a are always tested and returned in
        #  row-major, C-style order.
        #
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
        #
        # Example for the use of the intermediate callback.
        #
        # print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))
        print("Iter, obj, infeas #%d %g %g" % (iter_count, obj_value, inf_pr))



# %% test reweighting


p = mtp.Problem(h=10, s=1, k=2)
p = mtp.Problem(h=100, s=1, k=4)
p = mtp.Problem(h=3000, s=1, k=10)
p = mtp.Problem(h=30000, s=1, k=20)
p = mtp.Problem(h=100000, s=1, k=25)
p = mtp.Problem(h=300000, s=1, k=30)
p = mtp.Problem(h=500000, s=1, k=50)
p = mtp.Problem(h=1000000, s=1, k=100)

p.wh
p.xmat
cc = p.xmat * p.wh[:, None]  # multiply each column of xmat by wh
# get multiplicative scaling factors - make avg derivative 1 (or other goal)
ccgoal = 1
# use mean or median as the denominator
denom = cc.sum(axis=0) / cc.shape[0]  # mean
# denom = np.median(cc, axis = 0)
ccscale = np.absolute(ccgoal / denom)
cc = cc * ccscale  # mult by scale to have avg derivative meet our goal


p.targets
seed(1)
r = np.random.randn(p.targets.size) / 50  # random normal
q = [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]
np.quantile(r, q)
targets = (p.targets * (1 + r)).flatten()
targets = targets * ccscale
targets

# x0 = np.ones(p.xmat.shape[0]) * 1.035
x0 = np.ones(p.xmat.shape[0])
lb = np.full(x0.size, 0.1)
ub = np.full(x0.size, 100)
cl = targets * .97
cu = targets * 1.03

myobj = Rw6(cc)
myobj.constraints(x0)
cl
cu

nlp = ipopt.problem(
            n=len(x0),
            m=len(cl),
            problem_obj=myobj,  # Rw1(cc)
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu
            )

nlp.addOption('file_print_level', 6)
outfile = 'test.out'
if os.path.exists(outfile):
    os.remove(outfile)
nlp.addOption('output_file', outfile)

# nlp.addOption('mu_strategy', 'adaptive')
# nlp.addOption('mehrotra_algorithm', 'yes')

nlp.addOption('jac_d_constant', 'yes')
nlp.addOption('hessian_constant', 'yes')
nlp.addOption('max_iter', 100)

# nlp.addOption('tol', 1e-7)

xtest = np.full(x0.size, 1.15)
objbase = myobj.objective(xtest)
objbase
objgoal = 10
objscale = objgoal / objbase
objscale = objscale.item()
objscale
# if(x0.size < 1e4):
#     objscale = 1.0
# else:
#     objscale = objscale.item()  # native float is needed

# type(objscale)
# objscale = np.float32(objscale)
# objscale = np.double(objscale)
# objscale = np.single(objscale)

# type(1e-2)
# objscale = .0003
# objscale = .01

nlp.addOption('obj_scaling_factor', objscale)  # multiplier

# nlp.setProblemScaling(obj_scaling=1e2)  # multiplier
# nlp.setProblemScaling(obj_scaling=1)

# xscale = np.full(x0.shape, 1e-2)
# nlp.setProblemScaling(x_scaling=xscale)

# nlp.addOption('linear_solver', 'ma57')

x, info = nlp.solve(x0)

# iter, secs:
# 50, 305 far from solved - No scaling by me or ipopt

# 31, 174 my scaling 10 but no obj_scaling
# 31, 178 my scaling 10, obj_scaling 1e-2
# 31, 178  my scaling 10, obj_scaling 1e2

# 21, 120 optimal but slacks, my scaling 1, no obj_scaling
# 21, 120 optimal but slacks, my scaling 1, no obj_scaling, xscaling 1e-2
# myscale 1, objscaling 1e-2: 22, 125, no issues
# myscale 1, objscaling 1e-2, mustrat adaptive: 23, 142, slacks
# myscale 1, objscaling 1e-2, mehrotra: 24, 143, slacks

# myscale 1, objscaling 1e-4: 30, 166
# myscale 1, objscaling 1.05-based: 32, 176
# myscale 1, objscaling 1.05-based x 10: 28, 155
# myscale 1, objscaling 1.05-based x 100: 27, 150
# myscale 1, objscaling 1.05-based x 100, jac constant: 27, 142
# myscale 1, objscaling 1.05-based x 100, jac+hess constant: 27, 131

# 383396

# myobj.objective(x0)
# myobj.constraints(x0)
# myobj.constraints(x)
# targets
np.sum(x0 * p.wh)
np.sum(x * p.wh)

myobj.constraints(x) / targets * 100 - 100
np.quantile(x, q)
np.quantile(x, [.5, .9, .97, .98, .981, .982, .985, .99, .995, 1])

n, bins, patches = plt.hist(x, 100, density=True, facecolor='g', alpha=0.75)

myobj.hessian(x0, 1, 1)
myobj.hessianstructure()

j = myobj.jacobian(x0)
j
cc
j.size
cc.size
js = np.nonzero(cc)
row, col = js
j2 = cc[row, col]
j2.size
row, col = myobj.jacobianstructure()
row, col
cc[row, col]
cc

ccnz = cc[row, col]
ccnz.size
cc.size


# %% test Reweight class

def constraints(x):
    return np.dot(x * p.wh, p.xmat)


p = mtp.Problem(h=10, s=1, k=2)
p = mtp.Problem(h=200, s=1, k=4)
p = mtp.Problem(h=500, s=1, k=5)
p = mtp.Problem(h=1000, s=1, k=6)
p = mtp.Problem(h=3000, s=1, k=10)
p = mtp.Problem(h=30000, s=1, k=20)
p = mtp.Problem(h=100000, s=1, k=5)
p = mtp.Problem(h=100000, s=1, k=25)
p = mtp.Problem(h=100000, s=1, k=35)
p = mtp.Problem(h=300000, s=1, k=10)
p = mtp.Problem(h=300000, s=1, k=30)
p = mtp.Problem(h=500000, s=1, k=50)
p = mtp.Problem(h=1000000, s=1, k=100)


seed(1)
r = np.random.randn(p.targets.size) / 100  # random normal
q = [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]
np.quantile(r, q)
targets = (p.targets * (1 + r)).flatten()

x0 = np.ones(p.wh.size)
t0 = constraints(x0)
pdiff0 = t0 / targets * 100 - 100
pdiff0

rwp = rw.Reweight(p.wh, p.xmat, targets)
x, info = rwp.reweight(xlb=0.1, xub=10,
                       crange=.015,
                       ccgoal=10, objgoal=100,
                       max_iter=50)
info['status_msg']

np.quantile(x, [0, .1, .25, .5, .75, .9, 1])

t1 = constraints(x)
pdiff1 = t1 / targets * 100 - 100
pdiff1

pdiff0
pdiff1 - pdiff0
n, bins, patches = plt.hist(x, 100, density=True, facecolor='g', alpha=0.75)

np.quantile(x, [0, .1, .25, .5, .75, .9, 1])
np.quantile(p.wh, [0, .1, .25, .5, .75, .9, 1])
x0.sum()
x.sum()

# save a difficult problem to csv files
# we need p.wh, p.xmat, targets
# x, info = rwp.reweight(xlb=0.1, xub=10,
#                        crange=.02,
#                        ccgoal=10, objgoal=100,
#                        max_iter=100)
whdf = pd.DataFrame(data=p.wh, columns=["wh"])
whdf.to_csv("wh.csv", index=False)

xmatdf = pd.DataFrame(data=p.xmat, columns=range(1, p.k + 1))
xmatdf = xmatdf.add_prefix('x')
xmatdf.to_csv("xmat.csv", index=False)

targetsdf = pd.DataFrame(data=targets, columns=['target'])
targetsdf.to_csv('targets.csv', index=False)

constraints(x0)
constraints(x)

np.quantile(x, q)
np.quantile(x, [.5, .9, .97, .98, .981, .982, .985, .99, .995, 1])




# %% other stuff

def jacd(m):
    I = np.eye(m)
    J = np.vstack((np.hstack((I, I)), np.hstack((-I, I))))
    return J

def jacs(m):
    I = np.eye(m)
    J = np.vstack((np.hstack((I, I)), np.hstack((-I, I))))
    row, col = jacstr(m)
    return J[row, col]

def jacstr(m):
    return np.nonzero(np.tile(np.eye(m), (2, 2)))

m = 3
jacd(m)
jacstr(m)
jacs(m)


x0 = [1.0, 5.0, 5.0, 1.0]

lb = [1.0, 1.0, 1.0, 1.0]
ub = [5.0, 5.0, 5.0, 5.0]

cl = [25.0, 40.0]
cu = [2.0e19, 40.0]


p.wh
p.xmat
cc = p.xmat * p.wh[:, None]  # multiply each column of xmat by wh

p.targets
seed(1)
r = np.random.randn(p.targets.size) / 50  # random normal
q = [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]
np.quantile(r, q)
targets = (p.targets * (1 + r)).flatten()

x0 = np.ones(p.xmat.shape[0])
lb = np.full(x0.size, 0)
ub = np.full(x0.size, 100)
cl = targets * .9
cu = targets * 1.1

myobj = Rw(cc)

myobj.objective(x0)
myobj.constraints(x0)

nlp = ipopt.problem(
            n=len(x0),
            m=len(cl),
            problem_obj=Rw2(),
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu
            )


x, info = nlp.solve(x0)


def objective(x):
    #
    # The callback for calculating the objective
    #
    return np.sum((x - 1)**2)


def gradient(x):
    #
    # The callback for calculating the gradient
    #
    return 2 * x - 2


def constraints(x, cc):
    #
    # The callback for calculating the constraints
    #
    return np.dot(x, cc)


def jacobian(x, cc):
    #
    # The callback for calculating the Jacobian
    #  return nonzero elements, flattened
    #
    return cc[cc != 0]


def jacobianstructure(cc):
    return np.nonzero(cc)


def hessianstructure(self):
    #
    # The structure of the Hessian
    # Note:
    # The default hessian structure is of a lower triangular matrix. Therefore
    # this function is redundant. I include it as an example for structure
    # callback.
    #
    # np.tril returns the lower triangle (including diagonal) of an array.
    # numpy.nonzero returns the indices of the elements that are non-zero.
    #  returns a tuple of arrays, one for each dimension of a,
    #  containing the indices of the non-zero elements in that dimension
    #  The values in a are always tested and returned in
    #  row-major, C-style order.
    #
    return np.nonzero(np.eye(x.size))

np.nonzero(np.eye(x.size))



def hessian(self, x, lagrange, obj_factor):
    #
    # The callback for calculating the Hessian
    #
    H = np.full(x.size, 2) * obj_factor
    return H


row = ([4, 6])
col = ([2, 3])
row, col
p.xmat[row, col]
p.xmat


p.wh
p.xmat
cc = p.xmat * p.wh[:, None]  # multiply each column of xmat by wh
x = np.ones(p.xmat.shape[0])
constraints(x, cc)
jacobian(x, cc)
jacobianstructure(cc)
p.targets
seed(1)
r = np.random.randn(p.targets.size) / 50  # random normal
q = [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]
np.quantile(r, q)
targets = (p.targets * (1 + r)).flatten()








