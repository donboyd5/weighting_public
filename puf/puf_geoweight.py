# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 06:56:25 2020

Parallel group_by?
https://pandas.pydata.org/pandas-docs/stable/ecosystem.html?highlight=parallel

https://stackoverflow.com/questions/26187759/parallelize-apply-after-pandas-groupby

https://stackoverflow.com/questions/1704401/is-there-a-simple-process-based-parallel-map-for-python
https://towardsdatascience.com/10x-faster-parallel-python-without-python-multiprocessing-e5017c93cce1
https://github.com/ray-project/ray

@author: donbo
"""

# Notes about eclipse


# %% set working directory if not already set
import os
os.getcwd()
os.chdir('C:/programs_python/weighting')
os.getcwd()

# %% imports
import sys
import requests
import pandas as pd
import numpy as np
import src.microweight as mw
import src.make_test_problems as mtp


# %% constants
WEBDIR = 'https://www.irs.gov/pub/irs-soi/'
DOWNDIR = 'C:/programs_python/weighting/puf/downloads/'
DATADIR = 'C:/programs_python/weighting/puf/data/'
# PUFDIR = 'C:/programs_python/weighting/puf/'
HDFDIR = 'C:/programs_python/weighting/puf/ignore/'

HT2_2018 = "18in55cmagi.csv"

# agi stubs
# AGI groups to target separately
IRS_AGI_STUBS = [-9e99, 1.0, 5e3, 10e3, 15e3, 20e3, 25e3, 30e3, 40e3, 50e3,
                 75e3, 100e3, 200e3, 500e3, 1e6, 1.5e6, 2e6, 5e6, 10e6, 9e99]
HT2_AGI_STUBS = [-9e99, 1.0, 10e3, 25e3, 50e3, 75e3, 100e3,
                 200e3, 500e3, 1e6, 9e99]


# %% crosswalks
# This is Peter's xwalk plus mine -- it includes more than we will use
PUFHT2_XWALK = {
    'nret_all': 'N1',  # Total population
    'nret_mars1': 'MARS1',  # Single returns number
    'nret_mars2': 'MARS2',  # Joint returns number
    'c00100': 'A00100',  # AGI amount
    'e00200': 'A00200',  # Salary and wage amount
    'e00200_n': 'N00200',  # Salary and wage number
    'e00300': 'A00300',  # Taxable interest amount
    'e00600': 'A00600',  # Ordinary dividends amount
    'c01000': 'A01000',  # Capital gains amount
    'c01000_n': 'N01000',  # Capital gains number
    # check Social Security
    # e02400 is Total Social Security
    # A02500 is Taxable Social Security benefits amount
    'e02400': 'A02500',  # Social Security total (2400)
    'c04470': 'A04470',  # Itemized deduction amount (0 if standard deduction)
    'c04470_n': 'N04470',  # Itemized deduction number (0 if standard deduction)
    'c17000': 'A17000',  # Medical expenses deducted amount
    'c17000_n': 'N17000',  # Medical expenses deducted number
    'c04800': 'A04800',  # Taxable income amount
    'c04800_n': 'N04800',  # Taxable income number
    'c05800': 'A05800',  # Regular tax before credits amount
    'c05800_n': 'N05800',  # Regular tax before credits amount
    'c09600': 'A09600',  # AMT amount
    'c09600_n': 'N09600',  # AMT number
    'e00700': 'A00700',  # SALT amount
    'e00700_n': 'N00700',  # SALT number
    # check pensions
    # irapentot: IRAs and pensions total e01400 + e01500
    # A01750: Taxable IRA, pensions and annuities amount
    'irapentot': 'A01750',
}
PUFHT2_XWALK
# CAUTION: reverse xwalk relies on having only one keyword per value
HT2PUF_XWALK = {val: kw for kw, val in PUFHT2_XWALK.items()}
HT2PUF_XWALK
list(HT2PUF_XWALK.keys())


# %% utility functions


def getmem(objects=dir()):
    """Memory used, not including objects starting with '_'.

    Example:  getmem().head(10)
    """
    mb = 1024**2
    mem = {}
    for i in objects:
        if not i.startswith('_'):
            mem[i] = sys.getsizeof(eval(i))
    mem = pd.Series(mem) / mb
    mem = mem.sort_values(ascending=False)
    return mem


# %% program functions

def wsum(grp, sumvars, wtvar):
    """ Returns data frame row with weighted sums of selected variables.

        grp: a dataframe (typically a dataframe group)
        sumvars: the variables for which we want weighted sums
        wtvar:  the weighting variable
    """
    return grp[sumvars].multiply(grp[wtvar], axis=0).sum()


def constraints(x, wh, xmat):
    return np.dot(x * wh, xmat)


# %% ONETIME download Historical Table 2
files = [HT2_2018]

for f in files:
    print(f)
    url = WEBDIR + f
    path = DOWNDIR + f
    r = requests.get(url)
    print(r.status_code)
    with open(path, "wb") as file:
        file.write(r.content)


# %% read and adjust Historical Table 2

ht2 = pd.read_csv(DOWNDIR + HT2_2018, thousands=',')
ht2
ht2.info()
ht2.STATE.describe()  # 54 -- states, DC, US, PR, OA
ht2.STATE.value_counts().sort_values()
ht2.groupby('STATE').STATE.count()  # alpha order
ht2.head()
ht2.columns.to_list()
# # convert all strings to numeric
# stn = ht2raw.columns.to_list()
# stn.remove('STATE')
# ht2[stn] = ht2raw[stn].apply(pd.to_numeric, errors='coerce', axis=1)
ht2

h2stubs = pd.DataFrame([
    [0, 'All income ranges'],
    [1, 'Under $1'],
    [2, '$1 under $10,000'],
    [3, '$10,000 under $25,000'],
    [4, '$25,000 under $50,000'],
    [5, '$50,000 under $75,000'],
    [6, '$75,000 under $100,000'],
    [7, '$100,000 under $200,000'],
    [8, '$200,000 under $500,000'],
    [9, '$500,000 under $1,000,000'],
    [10, '$1,000,000 or more']],
    columns=['h2stub', 'h2range'])
h2stubs
# h2stubs.info()


# %% get reweighted national puf
PUF_RWTD = HDFDIR + 'puf2018_reweighted' + '.h5'
pufrw = pd.read_hdf(PUF_RWTD)  # 1 sec
pufrw.columns.sort_values()
pufrw


# %% prepare puf subset and weighted sums
# create a subset with ht2stub variable
pufsub = pufrw.copy()
pufsub['HT2_STUB'] = pd.cut(
    pufsub['c00100'],
    HT2_AGI_STUBS,
    labels=list(range(1, len(HT2_AGI_STUBS))),
    right=False)
pufsub.columns.sort_values().tolist()  # show all column names

pufsub[['pid', 'c00100', 'HT2_STUB', 'IRS_STUB']].sort_values(by='c00100')

# create list of target vars
alltargvars = ['nret_all', 'nret_mars1', 'nret_mars2',
            'c00100', 'e00200', 'e00300', 'e00600',
            'c01000', 'e02400', 'c04800', 'irapentot']
alltargvars

pufsums = pufsub.groupby('HT2_STUB').apply(wsum,
                                            sumvars=alltargvars,
                                            wtvar='wtnew')
pufsums = pufsums.append(pufsums.sum().rename(0)).sort_values('HT2_STUB')
pufsums['HT2_STUB'] = pufsums.index
pufsums


# %% prepare compatible ht2 subset
# we'll rename HT2 columns to be like puf

# prepare a list of column names we want from ht2
ht2_all = list(HT2PUF_XWALK.keys())  # superset
ht2_all
prefixes = ('N0')  # must be tuple, not a list
ht2_use = [x for x in ht2_all if not x.startswith(prefixes)]
drops = ['N17000', 'A17000', 'A04470', 'A05800', 'A09600', 'A00700']
ht2_use = [x for x in ht2_use if x not in drops]
ht2_use
# ht2_use.remove('N17000')
ht2_use.append('STATE')
ht2_use.append('AGI_STUB')
ht2_use

# get those columns, rename as needed, and create new columns
ht2_sub = ht2[ht2_use].copy()
ht2_sub = ht2_sub.rename(columns=HT2PUF_XWALK)
ht2_sub = ht2_sub.rename(columns={'AGI_STUB': 'HT2_STUB'})
# multiply dollar values by 1000
ht2_sub.columns
dollcols = ['c00100', 'e00200', 'e00300',
            'e00600', 'c01000', 'e02400', 'c04800', 'irapentot']
dollcols
ht2_sub[dollcols] = ht2_sub[dollcols] * 1000
ht2_sub


# %% compare pufsums to HT2 for US
ht2sums = ht2_sub.query('STATE=="US"')
ht2sums = ht2sums.drop(columns=['STATE'])
ht2sums.columns

pufsums.columns

ht2sums
pufsums

pd.options.display.max_columns
pd.options.display.max_columns = 99
round(pufsums.drop(columns='HT2_STUB') / ht2sums.drop(columns='HT2_STUB') * 100 - 100)
pd.options.display.max_columns = 0
# e02400 is way off, c04800 has some trouble, and irapentot is way off, so don't use them
# the rest look good

# %% targvars and states definitions
targvars = ['nret_all', 'nret_mars1', 'nret_mars2', 'c00100', 'e00200',
            'e00300', 'e00600']
targvars + ['HT2_STUB']

targstates = ['CA', 'FL', 'NY', 'TX']
targstates = ['CA']
targstates = ['CA', 'FL']
targstates = ['CA', 'NY']
targstates = ['CA', 'TX']
targstates = ['FL', 'TX']
targstates = ['FL', 'NY', 'TX']
targstates = ['CA', 'FL', 'NY', 'OH', 'PA', 'TX']
targstates = ['CA', 'CT', 'FL', 'GA', 'MA', 'NY', 'OH', 'OR', 'PA', 'TX', 'WA']

targstates = ['AL', 'AR', 'CA', 'CT', 'FL', 'GA', 'MA', 'MN', 'NJ', 'NY', 'OH', 'OR', 'PA', 'TX', 'WA']
targstates = ['AK', 'AL', 'AR', 'CA', 'CT', 'FL', 'GA', 'MD',
              'MA', 'MN', 'NH', 'NJ', 'NY', 'OH', 'OR', 'PA', 'TN', 'TX', 'VT', 'WA']


# %% geoweight just one stub
pufsub.columns
pufsub[['HT2_STUB', 'pid']].groupby(['HT2_STUB']).agg(['count'])

stub = 9
pufstub = pufsub.query('HT2_STUB == @stub')[['pid', 'HT2_STUB', 'wtnew'] + targvars]
pufstub

ht2stub = ht2_sub.query('HT2_STUB == @stub & STATE != "US"')[['STATE', 'HT2_STUB'] + targvars]
ht2stub
# show average target value per return times 100
round(ht2stub[targvars].div(ht2stub.nret_all, axis=0) * 100, 1)

# create an adjusted ht2stub that only has target states
ht2stub_adj = ht2stub.copy()
mask = np.logical_not(ht2stub_adj['STATE'].isin(targstates))
column_name = 'STATE'
ht2stub_adj.loc[mask, column_name] = 'XX'
ht2stub_adj[['STATE', 'HT2_STUB']].groupby(['STATE']).agg(['count'])
ht2stub_adj = ht2stub_adj.groupby(['STATE', 'HT2_STUB']).sum()
ht2stub_adj.info()
# average target value per return
round(ht2stub_adj.div(ht2stub_adj.nret_all, axis=0), 1)
ht2stub_adj.sum()
ht2stub_adj
# pufsums.query('HT2_STUB == @stub')[targvars]
# np.array(ht2stub_adj.sum())
# ratios = pufsums.query('HT2_STUB == @stub')[targvars] / np.array(ht2stub_adj.sum())
# ratios = np.array(ratios)


wh = pufstub.wtnew.to_numpy()
xmat = np.asarray(pufstub[targvars], dtype=float)
xmat.shape
# use one of the following
targets = ht2stub.drop(columns=['STATE', 'HT2_STUB'])
# targets = ht2stub_adj # .drop(columns=['STATE', 'HT2_STUB'])
targets = np.asarray(ht2stub_adj, dtype=float)
targets
# targets_scaled = targets * ratios
# targets.shape
# targets_scaled.shape

# targets_scaled / targets

# scale targets by ratio of pufsums to HT2

g = mw.Microweight(wh, xmat, targets)
# g = mw.Microweight(wh, xmat, targets_scaled)

# look at the inputs
g.wh
g.xmat
g.geotargets

# g.wh.shape
# g.xmat.shape
# g.geotargets.shape

# solve for state weights
g.geoweight()

# examine results
g.elapsed_minutes
np.round(g.result.fun.reshape(targets.shape), 1)
g.result  # this is the result returned by the solver
dir(g.result)
g.result.cost  # objective function value at optimum
g.result.message

# optimal values
g.beta_opt  # beta coefficients, s x k
g.delta_opt  # delta constants, 1 x h
g.whs_opt  # state weights
g.whs_opt.shape
g.geotargets_opt

g.geotargets_opt.sum(axis=0)


np.round(g.result.fun, 1)
np.round(g.result.fun.reshape(targets.shape), 1)
round(ht2stub_adj.div(ht2stub_adj.nret_all, axis=0), 1)
# np.round(g.result.fun.reshape(7, 5), 1)

# %% let's try fmin_slsqp, completely different way
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_slsqp.html#scipy.optimize.fmin_slsqp
# minimize the least squares diff using the result above as starting point
# with weight adding-up constraint
# https://www.programcreek.com/python/example/114542/scipy.optimize.fmin_slsqp

import scipy as sp
from scipy.optimize import fmin_slsqp
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import show_options
show_options(solver="minimize", method="trust-ncg")

# constraints
# scipy.optimize.LinearConstraint(A, lb, ub, keep_feasible=False)[source]
# where:
# lb <= A.dot(x) <= ub
# A is (m, n) where m=number constraints, n=number variables
# this gives res[1] = A[1, 1] * x[1] + A[1, 2] * x[2] + ...A[1, n] * x[m]
# x is whs.flatten()
# lb, ub, are wh
# so we have h rows in A, whs.size columns in A
# A will have 1s in the appropriate places!!
xx = p.whs.flatten() # xx[0:2] = p.whs [0, 0:2], xx[3:5] = p.whs[1, 0:2]
# A:
# 1 1 1 0000000000000000
# 0 0 0, 1 1 1, 000000000000
# 000, 000, 111, 0000000000
nwh = p.whs.shape[0]  # int(xx.size)
ns = p.whs.shape[1]
A = np.zeros((nwh, nwh * ns))  # Pre-allocate matrix
for i in range(0, nwh):
    A[i, range(i*ns, i*ns+ns)] = 1
A
A.dot(xx)
p.wh
lincon = sp.optimize.LinearConstraint(A, p.wh, p.wh)

from scipy.sparse import lil_matrix
# from scipy.sparse.linalg import spsolve
# from numpy.linalg import solve, norm
# from numpy.random import rand
A2 = lil_matrix((nwh, nwh * ns))
for i in range(0, nwh):
    A2[i, range(i*ns, i*ns+ns)] = 1
A2
A2.dot(xx)
lincon2 = sp.optimize.LinearConstraint(A2, p.wh, p.wh)


A = lil_matrix((1000, 1000))
A[0, :100] = rand(100)
A[1, 100:200] = A[0, :100]
A.setdiag(rand(1000))
Now convert it to CSR format and solve A x = b for x:

>>>
A = A.tocsr()



wh
xmat
targets

p = mtp.Problem(h=20, s=3, k=2)

p = mtp.Problem(h=40, s=5, k=2)

p.whs.shape
np.array([1, 2, 3]).shape
xx = np.array([1, 2, 3])
zz = p.whs.dot(xx) # (20, 3) dot (3, ) gives (20, )
zz = p.whs.dot(xx.reshape(3, 1)) # (20, 3) dot (3, 1) gives (20, 1)
# values are the same
zz
zz.shape

p.wh
p.targets
p.xmat
p.whs
diffs = np.dot(p.whs.T, p.xmat) - p.targets
np.square(diffs).sum()

import autograd.numpy as np
# import autograd.numpy.random as npr
# import autograd.scipy.signal
from autograd import grad




# def rosen_der(x):
#     xm = x[1:-1]
#     xm_m1 = x[:-2]
#     xm_p1 = x[2:]
#     der = np.zeros_like(x)
#     der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
#     der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
#     der[-1] = 200*(x[-1]-x[-2]**2)
#     return der



gx = mw.Microweight(p.wh, p.xmat, p.targets)
gx.geoweight()
gx.whs_opt
gx.whs_opt.flatten()

# lb <= A.dot(x) <= ub
# x is whs.flatten()
# lb, ub, are wh
# so we have h rows in A, whs.size columns in A


x= p.whs.flatten()
x.reshape((p.xmat.shape[0], p.targets.shape[0]))
f(p.whs, p.xmat, p.targets)
x0 = np.full(p.whs.size, 0)
x0 = np.full(p.whs.size, 1)
x0 = np.full(p.whs.size, np.mean(p.wh))
x0 = gx.whs_opt.flatten()
# ans = fmin_slsqp(f, x0, args=(p.xmat, p.targets), iprint=2)
# options={'func': None, 'maxiter': 100, 'ftol': 1e-06, 'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08, 'finite_diff_rel_step': None}
# res = ((test_tup, ) * N)
bnds = [(14.0, 19.0)] * x0.size
bnds = ((0, 120),) * x0.size
bnds

bnds = sp.optimize.Bounds(5.0, 25.0)
bnds = sp.optimize.Bounds(13.0, 19.0, keep_feasible=True)
bnds = sp.optimize.Bounds(0.0, np.inf)
p.whs.min()
p.whs.max()

minimize(f, x0, method='SLSQP',
               bounds=bnds,
               args=(p.xmat, p.targets),
               options={'iprint': 2, 'disp': True})

# fmin_slsqp(f, x0, args=(p.xmat, p.targets), iprint=2)
bnds = [(0, 30) for _ in x0]
bnds = [(13, 30.0)] * x0.size # zero lbound does not work, 14 lbound does not work
ans = fmin_slsqp(f, x0, bounds=bnds, args=(p.xmat, p.targets), iprint=2)

# it seems like slsqp has trouble with bounds, so must use trust approach
# do not use method='SLSQP', let it choose , options={'iprint': 2, 'disp': True}

def targs(x, xmat, targets):
    whs = x.reshape((xmat.shape[0], targets.shape[0]))
    return np.dot(whs.T, xmat)

def f(x, xmat, targets, objscale):
    whs = x.reshape((xmat.shape[0], targets.shape[0]))
    diffs = np.dot(whs.T, xmat) - targets
    # diffs = diffs * diff_weights
    obj = np.square(diffs).sum() * objscale
    return(obj)

# define gradient of objective function
gfn = grad(f)


p = mtp.Problem(h=1000, s=20, k=10)

A2 = lil_matrix((p.h, p.h * p.s))
for i in range(0, p.h):
    A2[i, range(i*p.s, i*p.s + p.s)] = 1
A2
b=A2.todense()
lincon2 = sp.optimize.LinearConstraint(A2, p.wh, p.wh)

bnds = sp.optimize.Bounds(0, np.inf)
# objscale = 1e-6
xcheck = np.full(p.whs.size, np.mean(p.wh))
objscale = 1 / f(xcheck, p.xmat, p.targets, 1) * 1e4
objscale
f(xcheck, p.xmat, p.targets, objscale)

x0 = np.full(p.whs.size, 0)
x0 = np.full(p.whs.size, 1)
x0 = np.full(p.whs.size, np.mean(p.wh))

f(x0, p.xmat, p.targets, objscale)

x0 = np.full(p.whs.size, np.mean(p.wh))
res = minimize(f, x0,
               method='trust-constr',
               # method='SLSQP',
               bounds=bnds,
               constraints=lincon2,
               jac=gfn,
               args=(p.xmat, p.targets, objscale),
               options={'verbose': 2})

res2

res = res2
res = res3
res.fun
res.jac
res.message
res.nfev
res.nit
res.njev
res.status
res.x
res.x.min()
res.x.max()

np.round(A2.dot(res.x) - p.wh, 1)  # sum of state weights minus national weights
# targs(res.x, p.xmat, p.targets) - p.targets  # calculated targets minus targets
np.round((targs(res.x, p.xmat, p.targets) - p.targets) / p.targets * 100, 4)  # pct diff

res2.message
res3.message
res2.x
res2.x.min()
res2.x.max()


# compare to p.whs
p.whs
res.x.reshape(p.whs.shape)
gx.whs_opt

out, fx, its, imode = fmin_slsqp(f, x0, args=(p.xmat, p.targets), iprint=2)
dir(ans)
dir(out)
type(out)
out.out
out.fx

# res = minimize(rosen, x0, method='SLSQP', jac=rosen_der,
#                constraints=[eq_cons, ineq_cons], options={'ftol': 1e-9, 'disp': True},
#                bounds=bounds)

# out ndarray of float The final minimizer of func.
# fx ndarray of float, if full_output is true The final value of the objective function.
# its int, if full_output is true The number of iterations.
# imode int, if full_output is true The exit mode from the optimizer (see below).
# smode string, if full_output is true Message describing the exit mode from the optimizer
# -1 : Gradient evaluation required (g & a)
#  0 : Optimization terminated successfully
#  1 : Function evaluation required (f & c)
#  2 : More equality constraints than independent variables
#  3 : More than 3*n iterations in LSQ subproblem
#  4 : Inequality constraints incompatible
#  5 : Singular matrix E in LSQ subproblem
#  6 : Singular matrix C in LSQ subproblem
#  7 : Rank-deficient equality constraint subproblem HFTI
#  8 : Positive directional derivative for linesearch
#  9 : Iteration limit reached

# %% cvxpy
# https://www.cvxpy.org/

import cvxpy as cp

def diffs(x, xmat, targets):
    whs = x.reshape((xmat.shape[0], targets.shape[0]))
    diffs = np.dot(whs.T, xmat) - targets
    return(diffs)

p = mtp.Problem(h=20, s=3, k=2)
x = cp.Variable(p.whs.size)
# constraints = [0 <= x, x <= 1]
# objective = cp.Minimize(cp.sum_squares(A*x - b))


# Problem data.
m = 30
n = 20
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

# Construct the problem.
x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A@x - b))
constraints = [0 <= x, x <= 1]
prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
# The optimal value for x is stored in `x.value`.
print(x.value)
# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
print(constraints[0].dual_value)


# %% get new national weights - loop through all states in a stub and calculate weights
# prepare the stub
stub = 9
pufstub = pufsub.query('HT2_STUB == @stub')[['pid', 'HT2_STUB', 'wtnew'] + targvars]
ht2stub = ht2_sub.query('HT2_STUB == @stub & STATE != "US"')[['STATE', 'HT2_STUB'] + targvars]
# begin the loop (must loopify)
# create an adjusted ht2stub that only has target state
tstate = 'NY'
ht2stub_adj = ht2stub.copy()
mask = np.logical_not(ht2stub_adj['STATE'] == tstate)
column_name = 'STATE'
ht2stub_adj.loc[mask, column_name] = 'XX'
# ht2stub_adj[['STATE', 'HT2_STUB']].groupby(['STATE']).agg(['count'])
ht2stub_adj = ht2stub_adj.groupby(['STATE', 'HT2_STUB']).sum()
# ht2stub_adj.info()
# ht2stub_adj
# average target value per return
# round(ht2stub_adj.div(ht2stub_adj.nret_all, axis=0), 1)
# ht2stub_adj.sum()
# ht2stub_adj
# pufsums.query('HT2_STUB == @stub')[targvars]
# np.array(ht2stub_adj.sum())
# ratios = pufsums.query('HT2_STUB == @stub')[targvars] / np.array(ht2stub_adj.sum())
# ratios = np.array(ratios)
# looks like int income in OA is a problem

wh = pufstub.wtnew.to_numpy()
xmat = np.asarray(pufstub[targvars], dtype=float)
# xmat.shape

targets = np.asarray(ht2stub_adj, dtype=float)
targets

g = mw.Microweight(wh, xmat, targets)
# g = mw.Microweight(wh, xmat, targets_scaled)

# look at the inputs
g.wh
g.xmat
g.geotargets

# solve for state weights
g.geoweight()
# g.beta_opt
# g.delta_opt  # delta constants, 1 x h
pd.DataFrame(g.beta_opt.flatten()).describe()
pd.DataFrame(g.delta_opt).describe()

# After 10 iterations
# C:\programs_python\weighting\src\microweight.py:103: RuntimeWarning: overflow encountered in exp
#   beta_x = np.exp(np.dot(beta, xmat.T))
# C:\programs_python\weighting\src\microweight.py:105: RuntimeWarning: divide by zero encountered in true_divide
#   delta = np.log(wh / beta_x.sum(axis=0))  # axis=0 gives colsums
# C:\programs_python\weighting\src\microweight.py:105: RuntimeWarning: overflow encountered in true_divide
#   delta = np.log(wh / beta_x.sum(axis=0))  # axis=0 gives colsums
# C:\programs_python\weighting\src\microweight.py:105: RuntimeWarning: divide by zero encountered in log
#   delta = np.log(wh / beta_x.sum(axis=0))  # axis=0 gives colsums
# C:\Users\donbo\anaconda3\envs\analysis\lib\site-packages\numpy\core\_methods.py:47: RuntimeWarning: overflow encountered in reduce
#   return umr_sum(a, axis, dtype, out, keepdims, initial, where)

# examine results
g.elapsed_minutes

# it must be possible to make percent errors better
np.round(g.result.fun.reshape(targets.shape), 1)
# x0 values and results
# x0 0
# array([[  5.2, -13.6,   8.8,   1.4,  -5.2,   0.7,   0.9],
#        [  0.2,   5.2,  -0.5,   0.6,   1.9,   5.6,   1.6]])
# x0 1e-9
# array([[ 2.1,  0.3, -0.5, -1.3,  0.1,  0.6,  0.2],
#        [ 0.5,  3.4,  0.3,  0.8,  1.3,  5.6,  1.7]])
# x0 1e-12
# array([[ 0.7,  0.3, -0.3, -0.1,  0.1,  0.6,  0.2],
#        [ 0.6,  3.4,  0.3,  0.7,  1.3,  5.6,  1.7]])


g.result  # this is the result returned by the solver
dir(g.result)
g.result.cost  # objective function value at optimum
g.result.message


# %% geoweight
mtp.Problem.help()

p = mtp.Problem(h=10000, s=50, k=10)
# p = mtp.Problem(h=20000, s=30, k=10)  # moderate-sized problem, < 1 min

# I don't think our problems for a single AGI range will get bigger
# than the one below:
#   30k tax records, 50 states, 30 characteristics (targets) per state
# but problems will be harder to solve with real data
# p = mtp.Problem(h=30000, s=50, k=30) # took 31 mins on my computer

mw.Microweight.help()

p.xmat.shape
p.targets.shape
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




# %% Peter's  crosswalks
# Peter's mappings of puf to historical table 2
# "n1": "N1",  # Total population
# "mars1_n": "MARS1",  # Single returns number
# "mars2_n": "MARS2",  # Joint returns number
# "c00100": "A00100",  # AGI amount
# "e00200": "A00200",  # Salary and wage amount
# "e00200_n": "N00200",  # Salary and wage number
# "c01000": "A01000",  # Capital gains amount
# "c01000_n": "N01000",  # Capital gains number
# "c04470": "A04470",  # Itemized deduction amount (0 if standard deduction)
# "c04470_n": "N04470",  # Itemized deduction number (0 if standard deduction)
# "c17000": "A17000",  # Medical expenses deducted amount
# "c17000_n": "N17000",  # Medical expenses deducted number
# "c04800": "A04800",  # Taxable income amount
# "c04800_n": "N04800",  # Taxable income number
# "c05800": "A05800",  # Regular tax before credits amount
# "c05800_n": "N05800",  # Regular tax before credits amount
# "c09600": "A09600",  # AMT amount
# "c09600_n": "N09600",  # AMT number
# "e00700": "A00700",  # SALT amount
# "e00700_n": "N00700",  # SALT number

    # Maps PUF variable names to HT2 variable names
VAR_CROSSWALK = {
    "n1": "N1",  # Total population
    "mars1_n": "MARS1",  # Single returns number
    "mars2_n": "MARS2",  # Joint returns number
    "c00100": "A00100",  # AGI amount
    "e00200": "A00200",  # Salary and wage amount
    "e00200_n": "N00200",  # Salary and wage number
    "c01000": "A01000",  # Capital gains amount
    "c01000_n": "N01000",  # Capital gains number
    "c04470": "A04470",  # Itemized deduction amount (0 if standard deduction)
    "c04470_n": "N04470",  # Itemized deduction number (0 if standard deduction)
    "c17000": "A17000",  # Medical expenses deducted amount
    "c17000_n": "N17000",  # Medical expenses deducted number
    "c04800": "A04800",  # Taxable income amount
    "c04800_n": "N04800",  # Taxable income number
    "c05800": "A05800",  # Regular tax before credits amount
    "c05800_n": "N05800",  # Regular tax before credits amount
    "c09600": "A09600",  # AMT amount
    "c09600_n": "N09600",  # AMT number
    "e00700": "A00700",  # SALT amount
    "e00700_n": "N00700",  # SALT number
}

