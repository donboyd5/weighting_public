# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 05:16:40 2020

@author: donbo
"""

# %% imports
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

import microweight.microweight as mw
import microweight.make_test_problems as mtp


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
r = np.random.randn(p.targets.size) / 100  # random normal
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
                 max_iter=40, verbose=2)
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



