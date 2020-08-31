# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 09:35:09 2020

@author: donbo
"""

# %% imports
import numpy as np
import make_test_problems as mtp
import scipy.optimize
from scipy.optimize import least_squares
from scipy.optimize import lsq_linear
from timeit import default_timer as timer

from scipy.optimize import newton_krylov
from numpy import cosh, zeros_like, mgrid, zeros

import matplotlib.pyplot as plt

from numpy.random import seed

# %% test NLP


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

jac2 = jacfn(x0, p.wh, p.xmat, targets, dwv)

def jacfn2(x, wh, xmat, targets, dw):
    jac = jac2
    return jac


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
from scipy.sparse import rand
from scipy.optimize import lsq_linear

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
p = mtp.Problem(h=3000, s=1, k=5)
p = mtp.Problem(h=30000, s=1, k=20)
p = mtp.Problem(h=300000, s=1, k=30)
p = mtp.Problem(h=500000, s=1, k=50)

seed(1)
r = np.random.randn(p.targets.size) / 100  # random normal
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
lb = np.full(p.h, 0.1)
ub = np.full(p.h, 10)

start = timer()
res = lsq_linear(As, b, bounds=(lb, ub),
                 method='trf',
                 tol=1e-3,
                 lsmr_tol='auto',
                 max_iter=100, verbose=2)
end = timer()
end - start

tmp = end - start
tmp

res.x
res.cost
res.fun
b
pdiff = res.fun / b * 100
np.abs(pdiff)
np.abs(pdiff).max()

res.x
Atraw = np.multiply(p.wh.reshape(p.h, 1), p.xmat)
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
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.optimize import BFGS


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


bounds = Bounds([0, -0.5], [1.0, 2.0])
linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])

# nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess=cons_H)

# nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1,
#                                            jac=cons_J, hess=cons_H_sparse)

# nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1,
#                                           jac=cons_J, hess=cons_H_linear_operator)

#nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess=BFGS())
#nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess='2-point')
nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1, jac='2-point', hess=BFGS())

# %% NLP solve
from scipy.optimize import minimize

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

