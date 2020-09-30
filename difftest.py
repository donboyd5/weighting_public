# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 12:00:05 2020

@author: donbo
"""

def get_diffs(x, xmat, targets):
    whs = x.reshape((xmat.shape[0], targets.shape[0]))
    diffs = np.dot(whs.T, xmat) - targets
    return diffs

def f(x, xmat, targets, objscale, diff_weights):
    whs = x.reshape((xmat.shape[0], targets.shape[0]))
    diffs = np.dot(whs.T, xmat) - targets
    diffs = diffs * diff_weights
    obj = np.square(diffs).sum() * objscale
    return obj

def gfun(x, xmat, targets, objscale, diff_weights):
    whs = x.reshape((xmat.shape[0], targets.shape[0]))
    diffs = np.dot(whs.T, xmat) - targets
    diffs = diffs * diff_weights
    return 2 * xmat.dot(diffs.T)


h = p.xmat.shape[0]
s = p.targets.shape[0]
p.targets.size
p.whs.mean()

p = mtp.Problem(h=5, s=4, k=3)
x1 = np.full(p.h * p.s, 15)
dw = np.full((p.s, p.k), 1)
f(x1, p.xmat, p.targets, 1, dw)
gcheck = egfn(x1, p.xmat, p.targets, 1, dw)  # returns vector size h x s -- change obj for each x
diffs = get_diffs(x1, p.xmat, p.targets) # 1 difference per target (s x k), 1 row per state, 1 col per var

2 * p.xmat.dot(diffs.T)
gcheck.reshape((p.h, p.s))

diffs

gcheck * 100


h = 0; s = 1
2 * p.xmat[h, 0] * diffs[s, 0] + 2 * p.xmat[h, 1] * diffs[s, 1]
2 * (p.xmat[h, 0] * diffs[s, 0] + p.xmat[h, 1] * diffs[s, 1])

2 * np.dot(p.xmat, diffs.T)


# 2*xmat[1, 1]*diff[1] + 2*xmat[1, 2]*diff[2] for 2 diffs (2 targets), for person 1 state 1 ()

def gfun(x, xmat, targets, objscale, diff_weights):
    diffs = get_diffs(x, xmat, targets)
    g = diffs
    return g

gfun(x1, p.xmat, p.targets, 1, 1)


I managed to solve this for my problem by writing my own Newton optimisation:

import numpy as np
from sksparse.cholmod import cholesky

f = np.zeros(n_data)

difference = 999

while difference > 1e-5:

    cur_hess = hess(f)
    cur_jac = jac(f)

    cur_chol = cholesky(cur_hess)
    sol = cur_chol.solve_A(cur_jac)

    new_f = f - sol
    difference = np.linalg.norm(f - new_f)

    f = new_f
I'm using scikit-sparse to do a sparse Cholesky decomposition with CHOLMOD here. This works really well for my problem and typically converges within a few iterations. Hope it's of use to some!


Alternatively, it is also possible to define the Hessian  as a sparse matrix,

>>>
from scipy.sparse import csc_matrix
def cons_H_sparse(x, v):
    return v[0]*csc_matrix([[2, 0], [0, 0]]) + v[1]*csc_matrix([[2, 0], [0, 0]])
nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1,
                                           jac=cons_J, hess=cons_H_sparse)

