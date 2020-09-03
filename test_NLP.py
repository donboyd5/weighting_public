# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 09:35:09 2020

@author: donbo
"""

# %% imports

from timeit import default_timer as timer
import numpy as np
from numpy import cosh, zeros_like, mgrid, zeros
from numpy.random import seed

import scipy.optimize
from scipy.optimize import least_squares
from scipy.optimize import lsq_linear
from scipy.optimize import newton_krylov
from scipy.sparse import rand

from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.optimize import BFGS

# conda install -c conda-forge scikit-sparse
# import scikit-sparse

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

import make_test_problems as mtp


# %% functions


def targs(x, wh, xmat):
    return np.dot(x * wh, xmat)


def diffs(x, wh, xmat, targets):
    diffs = targs(x, wh, xmat) - targets
    return diffs.reshape(diffs.size)


def ssd(x, wh, xmat, targets):
    diffs = targs(x, wh, xmat) - targets
    ssd = np.square(diffs).sum()
    return ssd


def dw(targets, goal=100):
    goalmat = np.full(targets.shape, goal)
    with np.errstate(divide='ignore'):  # turn off divide-by-zero warning
        diff_weights = np.where(targets != 0, goalmat / targets, 1)
    return diff_weights


def diffs2(x, wh, xmat, targets, dw):
    diffs = targs(x, wh, xmat) - targets
    diffs = diffs.flatten() * dw
    return diffs.reshape(diffs.size)

# weighting from scratch
# obj=priority.weight * {(target.calc / scale - target / scale)^2}

# wdf <- dplyr::tibble(wt.iter=w, wtnum=1:length(w))

# grad <- inputs$coeffs %>%
#   dplyr::left_join(wdf, by="wtnum") %>%
#   dplyr::group_by(obj.element) %>%
#   dplyr::mutate(calc=sum(wt.iter*coeff)) %>%
#   dplyr::mutate(grad={2 * coeff * priority.weight * (calc - target)} / {scale^2}) %>%
#   dplyr::group_by(wtnum) %>%
#   dplyr::summarise(grad=sum(grad)) %>% # sum across all of the object elements for this particular weight
#   ungroup


def jacfn(x, wh, xmat, targets, dw):
    j1 = np.multiply(wh.reshape(wh.size, 1), xmat)
    j2 = dw * j1
    jac = j2.T
    return jac


def jacfn2(x, wh, xmat, targets, dw):
    jac = jac2
    return jac


# %% analysis
jac2 = jacfn(x0, p.wh, p.xmat, targets, dwv)
jacfn(xr, p.wh, p.xmat, targets, dwv)

diffs2(x0, p.wh, p.xmat, targets, dwv)
jacfn(x0, p.wh, p.xmat, targets, dwv)
x0.shape
p.xmat.shape

x0

p.xmat
p.xmat.sum(axis = 0)
np.dot(x0 * , p.xmat)

x0.flatten() * p.xmat
p.xmat * x0
type(x0)
type(p.xmat)

mtp.Problem.help()
p = mtp.Problem(h=10, s=1, k=2)
p = mtp.Problem(h=3000, s=1, k=5)
p = mtp.Problem(h=30000, s=1, k=30)
p.wh
p.xmat
p.targets

# targets = p.targets * [.9, 1.1, .87]
# targets = p.targets * 1.1

seed(1)
r = np.random.randn(p.targets.size) / 100  # random normal
targets = p.targets * (1 + r)

x0 = np.full(p.wh.shape, 1)

targs(x0, p.wh, p.xmat)
targets
p.targets

diffs(x0, p.wh, p.xmat, targets)
ssd(x0, p.wh, p.xmat, targets)

dwv = dw(targets)

diffs2(x0, p.wh, p.xmat, targets, dwv)
np.square(diffs2(x0, p.wh, p.xmat, targets, dwv)).sum() / 2

start = timer()
result = least_squares(diffs2,
                       x0,
                       method='trf',
                       # jac='2-point',
                       jac = jacfn2,
                       # x_scale = 'jac',
                       bounds=(0.25, 4),
                       max_nfev=6000,
                       # f_scale=0.1,
                       verbose=2,
                       args=(p.wh, p.xmat, targets, dwv))
end = timer()
end - start

result.x
resultj.x

# 3-point: 1105 secs, 1594 nfev, cost 0.009662235266516418
# 2-point: 1162, 1749, 0.0096628



# automatic differentiation
# https://github.com/google/jax


p.targets
targets
targs(result.x, p.wh, p.xmat)

diff = targs(result.x, p.wh, p.xmat) - targets
diff / targets * 100
result.x

start = timer()
result = least_squares(diffs,
                       x0,
                       method='trf',
                       jac='cs',
                       # x_scale = 'jac',
                       bounds=(0.1, 10),
                       max_nfev = 1000,
                       f_scale=0.1,
                       verbose=1,
                       args=(p.wh, p.xmat, targets))
end = timer()
end - start



result = least_squares(ssd, x0, method='trf', jac='2-point', verbose=,
                       args=(p.wh, p.xmat, p.targets))







def F(x):
    return np.cos(x) + x[::-1] - [1, 2, 3, 4]


x = scipy.optimize.broyden1(F, [1, 1, 1, 1], f_tol=1e-14)
x

np.cos(x) + x[::-1]


# %% large-scale least squares
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html


np.random.seed(0)

m = 20000
n = 10000

A = rand(m, n, density=1e-4)
b = np.random.randn(m)

m = 50
n = 200000

# A = rand(m, n, density=1)
A = rand(m, n, density=1)
A = rand(m, n)

A1 = A.toarray()


m = 50
n = 200000

A = np.random.rand(m, n)
As = scipy.sparse.coo_matrix(A)
b = np.random.randn(m)

lb = np.random.randn(n)
ub = lb + 1

start = timer()
res = lsq_linear(As, b, bounds=(lb, ub), lsmr_tol='auto',
                 max_iter=1000, verbose=2)
end = timer()
end - start

# %% test linear least squares

p = mtp.Problem(h=10, s=1, k=2)
p = mtp.Problem(h=100, s=1, k=4)
p = mtp.Problem(h=3000, s=1, k=10)
p = mtp.Problem(h=30000, s=1, k=20)
p = mtp.Problem(h=300000, s=1, k=30)
p = mtp.Problem(h=500000, s=1, k=50)

seed(1)
r = np.random.randn(p.targets.size) / 1000  # random normal
targets = (p.targets * (1 + r)).flatten()
diff_weights = np.where(targets != 0, 100 / targets, 1)
b = targets * diff_weights
b

wmat = p.xmat * diff_weights
At = np.multiply(p.wh.reshape(p.h, 1), wmat)
# At = np.multiply(p.wh.reshape(p.h, 1), p.xmat)
A = At.T
As = scipy.sparse.coo_matrix(A)

# lb = np.zeros(p.h)
# ub = lb + 10
lb = np.full(p.h, 0.5)
ub = np.full(p.h, 1.5)

lb = np.full(p.h, 0)
ub = np.full(p.h, 100)

p.h
p.k

start = timer()
res = lsq_linear(As, b, bounds=(lb, ub),
                 method='trf',
                 tol=1e-5,
                 lsmr_tol='auto',
                 max_iter=100, verbose=2)
end = timer()

print(end - start)
np.abs(res.fun).max()
res.fun
q = [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]
np.quantile(res.x, q)

Atraw = np.multiply(p.wh.reshape(p.h, 1), p.xmat)
sdiff = (np.dot(np.full(p.h, 1), Atraw) - targets) / targets * 100  # compare res.fun
sdiff

np.square(sdiff).sum()
np.square(res.fun).sum()

n, bins, patches = plt.hist(res.x, 500, density=True, facecolor='g', alpha=0.75)



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


# %% large-scale problem
# parameters
nx, ny = 75, 75
hx, hy = 1./(nx-1), 1./(ny-1)

P_left, P_right = 0, 0
P_top, P_bottom = 1, 0


def residual(P):
    d2x = zeros_like(P)
    d2y = zeros_like(P)

    d2x[1:-1] = (P[2:] - 2*P[1:-1] + P[:-2]) / hx/hx
    d2x[0] = (P[1] - 2*P[0] + P_left)/hx/hx
    d2x[-1] = (P_right - 2*P[-1] + P[-2])/hx/hx

    d2y[:, 1:-1] = (P[:, 2:] - 2*P[:, 1:-1] + P[:, :-2])/hy/hy
    d2y[:,0] = (P[:, 1] - 2*P[:,0] + P_bottom)/hy/hy
    d2y[:,-1] = (P_top - 2*P[:,-1] + P[:,-2])/hy/hy

    return d2x + d2y - 10*cosh(P).mean()**2

# solve
guess = zeros((nx, ny), float)
sol = newton_krylov(residual, guess, method='lgmres', verbose=1)
print('Residual: %g' % abs(residual(sol)).max())

# visualize
x, y = mgrid[0:1:(nx*1j), 0:1:(ny*1j)]
plt.pcolormesh(x, y, sol, shading='gouraud')
plt.colorbar()
plt.show()


# %% large-scale with constraints - setup
# see https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize

def cons_f(x):
    return [x[0]**2 + x[1], x[0]**2 - x[1]]

def cons_J(x):
    return [[2*x[0], 1], [2*x[0], -1]]


def cons_H(x, v):
    return v[0]*np.array([[2, 0], [0, 0]]) + v[1]*np.array([[2, 0], [0, 0]])


def cons_H_sparse(x, v):
    return v[0]*csc_matrix([[2, 0], [0, 0]]) + v[1]*csc_matrix([[2, 0], [0, 0]])

def cons_H_linear_operator(x, v):
    def matvec(p):
        return np.array([p[0]*2*(v[0]+v[1]), 0])
    return LinearOperator((2, 2), matvec=matvec)


# nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess=cons_H)

# nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1,
#                                            jac=cons_J, hess=cons_H_sparse)

# nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1,
#                                           jac=cons_J, hess=cons_H_linear_operator)

#nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess=BFGS())
#nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess='2-point')
nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1, jac='2-point', hess=BFGS())

# %% rosenbrock

def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

res1 = minimize(rosen, x0, method='nelder-mead',
               options={'xatol': 1e-8, 'disp': True})
res1.x
res1.fun

res2 = minimize(rosen, x0, method='BFGS', jac=rosen_der,
               options={'disp': True})
res2.x
res2.fun

def rosen_hess(x):
    x = np.asarray(x)
    H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200*x[0]**2-400*x[1]+2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
    H = H + np.diag(diagonal)
    return H

res3 = minimize(rosen, x0, method='Newton-CG',
               jac=rosen_der, hess=rosen_hess,
               options={'xtol': 1e-8, 'disp': True})

res3 = minimize(rosen, x0, method='Newton-CG',
               jac=rosen_der, hess=rosen_hess,
               options={'xtol': 1e-8, 'disp': True})
res3.x
res3.fun


def rosen_hess_p(x, p):
    x = np.asarray(x)
    Hp = np.zeros_like(x)
    Hp[0] = (1200*x[0]**2 - 400*x[1] + 2)*p[0] - 400*x[0]*p[1]
    Hp[1:-1] = -400*x[:-2]*p[:-2]+(202+1200*x[1:-1]**2-400*x[2:])*p[1:-1] \
               -400*x[1:-1]*p[2:]
    Hp[-1] = -400*x[-2]*p[-2] + 200*p[-1]
    return Hp
res4 = minimize(rosen, x0, method='Newton-CG',
               jac=rosen_der, hessp=rosen_hess_p,
               options={'xtol': 1e-8, 'disp': True})
res4.x
res4.fun

res5 = minimize(rosen, x0, method='trust-ncg',
               jac=rosen_der, hess=rosen_hess,
               options={'gtol': 1e-8, 'disp': True})
res5.x
res5.fun

res6 = minimize(rosen, x0, method='trust-ncg',
               jac=rosen_der, hessp=rosen_hess_p,
               options={'gtol': 1e-8, 'disp': True})
res6.x
res6.fun

res7 = minimize(rosen, x0, method='trust-krylov',
               jac=rosen_der, hess=rosen_hess,
               options={'gtol': 1e-8, 'disp': True})
res7.x
res7.fun

res8 = minimize(rosen, x0, method='trust-krylov',
               jac=rosen_der, hessp=rosen_hess_p,
               options={'gtol': 1e-8, 'disp': True})
res8.x
res8.fun

# All methods Newton-CG, trust-ncg and trust-krylov are suitable for dealing
# with large-scale problems (problems with thousands of variables). That is
# because the conjugate gradient algorithm approximately solve the trust-region
# subproblem (or invert the Hessian) by iterations without the explicit Hessian
# factorization. Since only the product of the Hessian with an arbitrary vector
# is needed, the algorithm is specially suited for dealing with sparse
# Hessians, allowing low storage requirements and significant time savings for
# those sparse problems.

# The minimize function provides algorithms for constrained minimization, namely
# 'trust-constr' , 'SLSQP' and 'COBYLA'. They require the constraints to be
# defined using slightly different structures. The method 'trust-constr' requires
# the constraints to be defined as a sequence of objects LinearConstraint and
# NonlinearConstraint. Methods 'SLSQP' and 'COBYLA', on the other hand, require
# constraints to be defined as a sequence of dictionaries, with keys type, fun
# and jac.


x0 = np.array([0.5, 0])
res = minimize(rosen, x0, method='trust-constr', jac=rosen_der, hess=rosen_hess,
               constraints=[linear_constraint, nonlinear_constraint],
               options={'verbose': 1}, bounds=bounds)

print(res.x)


# %% large scale NLP with constraints djb

# warm up - simple
def f(x):   # The rosenbrock function
    return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2

# no constraints, no jacobian


minimize(f, [2, -1], method="CG")

# no constraints, but with jacobian


def jacobian(x):
    return np.array((-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2),
                     2*(x[1] - x[0]**2)))


minimize(f, [2, 1], method="CG", jac=jacobian)


# no constraints, jacobian and hessian and different method


def hessian(x): # Computed with sympy
    return np.array(((1 - 4*x[1] + 12*x[0]**2,
                      -4*x[0]), (-4*x[0], 2)))

minimize(f, [2, -1], method="Newton-CG", jac=jacobian, hess=hessian)

minimize(f, [2, -1], method="L-BFGS-B", jac=jacobian)


# %% new problem, bounds and constraints

# this is a lasso problem, maybe faster ways to solver

def f(x):
    return np.sqrt((x[0] - 3)**2 + (x[1] - 2)**2)


minimize(f, np.array([0, 0]), bounds=((-1.5, 1.5), (-1.5, 1.5)))


def constraint(x):
    return np.atleast_1d(1.5 - np.sum(np.abs(x)))


x0 = np.array([0, 0])
constraint(x0)
constraint([1, -1])
minimize(f, x0, constraints={"fun": constraint, "type": "ineq"})


# %% test for real - setup


def f(x):
    # x is an array
    diffs = (x - 1)
    obj = np.square(diffs).sum()
    return (obj)


def jac(x):
    # jacobian (gradiant) of objective function
    return 2 * x - 2


def hess(x):
    # hessian of objective function
    return hmat

def hess_sparse(x):
    # hessian of objective function
    return hmat_sparse


hmat = np.identity(x0.size) * 2

hmat_sparse = scipy.sparse.identity(x0.size) * 2
# hmat_sparse.toarray()

# %% test for real - implementation

# methods available for minimization, POTENTIALLY w/bounds and constraints:
#  SLSQP -- default w/constraints and bounds (option verbose not available)
#  trust-constr -- handles constraints and bounds, option verbose available
#  trust-exact -- ??handles constraints & bounds, requires jacobian & hessian
#  trust-krylov -- ??handles constraints & bounds, requires jacobian & hessian

# other:
#  BFGS -- cannot handle constraints or bounds
#  CG -- cannot handle constraints or bounds
#  COBYLA -- cannot handle bounds (POSSIBLE option nonetheless)
#  dogleg -- cannot handle constraints or bounds, requires jacobian
#  L-BFGS-B -- cannot handle constraints (but handles bounds)
#  Nelder-Mead -- cannot handle constraints or bounds
#  Powell -- cannot handle constraints
#  TNC -- cannot handle constraints
#  trust-ncg --cannot handle constraints or bounds, requires jacobian

# Constraints for COBYLA, SLSQP are defined as a list of dictionaries. Each
#  dictionary with fields:
# type str Constraint type: ‘eq’ for equality, ‘ineq’ for inequality.
# fun callable The function defining the constraint.
# jac callable, optional The Jacobian of fun (only for SLSQP).
# argssequence, optional
# Extra arguments to be passed to the function and Jacobian.


p = mtp.Problem(h=100, s=1, k=5)
p = mtp.Problem(h=1000, s=1, k=10)
p = mtp.Problem(h=10000, s=1, k=10)
p.wh
p.xmat
p.targets

p.wh.shape
p.xmat.shape

At = p.xmat * p.wh[:, None]  # elementwise multiplication of matrix and vector
# A = np.multiply(p.wh.reshape(-1, 1), p.xmat)  # also works
A = At.T

x0 = np.ones(p.wh.size)
np.dot(A, x0)
p.targets

# add some noise to the targets so we have to work for a solution
r = np.random.randn(p.targets.size) / 100  # random normal
targets = (p.targets * (1 + r)).flatten()

# set up the problem
lb = 0
ub = 10
bounds = Bounds(lb, ub)

# constraints setup for SLSQP


def cons_sqp(x):
    return A.dot(x) - targets


# cons_sqp(x0)


#  constraints input for SLSQP:
# cons = ({'type': 'eq', 'fun': cons_sqp})

# ineq_cons = {'type': 'ineq',
#              'fun' : lambda x: np.array([1 - x[0] - 2*x[1],
#                                          1 - x[0]**2 - x[1],
#                                          1 - x[0]**2 + x[1]]),
#              'jac' : lambda x: np.array([[-1.0, -2.0],
#                                          [-2*x[0], -1.0],
#                                          [-2*x[0], 1.0]])}


# constraint setup for trust-constr
lc = LinearConstraint(A, targets * .98, targets * 1.02)
lc = LinearConstraint(A, targets, targets)
lc
# dir(lc)
lc.A
lc.lb
lc.ub
lc.A.shape

p.h
p.k

# SLSQP - only good for smaller problems
start = timer()
resc1 = minimize(f, x0, jac=jac,
                 bounds=bounds, constraints=cons,
                 method='SLSQP')
end = timer()
end - start
resc1.x
resc1.message
np.dot(A, resc1.x)
targets
np.dot(A, resc1.x) / targets * 100 - 100

# trust-constr -- better for larger problems
start = timer()
resc2 = minimize(f, x0, jac=jac,
                 hess=hess_sparse,  # hess, hess_sparse,
                 bounds=bounds, constraints=lc,
                 method='trust-constr',
                 options={'verbose': 2, 'maxiter': 20, 'disp': True})
end = timer()
end - start
# resc2.x
resc2.message
np.dot(A, resc2.x)
targets
diffs = np.dot(A, resc2.x) - targets
np.abs(diffs).sum()
np.dot(A, resc2.x) / targets * 100 - 100

start = timer()
resc3 = minimize(f, x0, jac=jac,  # hess='2-point',
                 bounds=bounds, constraints=lc,
                 method='trust-constr',
                 options={'verbose': 2, 'maxiter': 5, 'disp': True})
end = timer()
end - start

f(resc1.x)
f(resc2.x)
f(resc3.x)
resc1.message
resc2.message

np.dot(A, resc1.x)
targets


# use min <= calc <= max for constraints
# where we have a matrix that is post multiplied by X for calc
# and vectors (arrays) for min max

# djb come back here

linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])
linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])

minimize(f, x0, constraints={"fun": constraint, "type": "ineq"})

bounds = Bounds([0, -0.5], [1.0, 2.0])
linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])

x0 = np.array([0.5, 0])
res = minimize(rosen, x0, method='trust-constr', jac=rosen_der, hess=rosen_hess,
               constraints=[linear_constraint, nonlinear_constraint],
               options={'verbose': 1}, bounds=bounds)








f(np.array([0, 1, 2, 3]))

x0 = np.array([3, 4, 5])
f(x0)
jac(x0)
minimize(f, x0)


lb = 1.5
ub = 2
bounds = Bounds(lb, ub)
bounds = Bounds([0, 2, 0], [0.5, np.Inf, 7])
res = minimize(f, x0, bounds=bounds)
res

xmat = np.array([10, 10, 10,
                20, 20, 20]).reshape(3, 2)
xmat
xmat.T
np.dot(xmat.T, x0)
np.dot(xmat.T, res.x)


lc = LinearConstraint(xmat.T, [46, 64], [50, 68])
lc
dir(lc)
lc.A
lc.lb
lc.ub


resc = minimize(f, x0, bounds=bounds, constraints=lc,
                method='SLSQP', options={'verbose': 2})

resc = minimize(f, x0, jac=jac, bounds=bounds, constraints=lc,
                method='SLSQP', options={'verbose': 2})


resc
np.dot(xmat.T, x0)
np.dot(xmat.T, resc.x)
lc.lb
lc.ub
bounds.lb
bounds.ub
resc.x


# something bigger

