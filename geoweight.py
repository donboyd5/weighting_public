# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:17:48 2020

@author: donbo
"""
# %% notes
# https://www.w3schools.com/python


# %% imports
import numpy as np
# import random
from numpy.random import seed
from numpy.random import rand

from scipy.optimize import least_squares

from timeit import default_timer as timer


# %% functions


def get_delta(wh, beta, xmat):
    """
    Get vector of constants, 1 per household.

    Definitions:
    h: number of households
    k: number of characteristics each household has
    s: number of states or geographic areas

    Keyword arguments:
    wh: 1 x h vector of weights for each household
    beta: s x k matrix of poisson model coefficients
        (same for all households)
    xmat: h x k matrix of characteristics for each household

    See (Khitatrakun, Mermin, Francis, 2016, p.5)

    Note: we cannot let beta %*% xmat get too large!! or exp will be Inf and
    problem will bomb. It will get large when a beta element times an
    xmat element is large, so either beta or xmat can be the problem.
    """
    beta_x = np.exp(np.dot(beta, xmat.T))

    delta = np.log(wh / beta_x.sum(axis=0))  # axis=0 gives colsums
    return delta


def get_diff_weights(targets, goal=100):
    """
    difference weights - a weight to be applied to each target in the
      difference function so that it hits its goal
      set the weight to 1 if the target value is zero

    do this in a vectorized way
    """

    # avoid divide by zero or other problems

    # numerator = np.full(targets.shape, goal)
    # with np.errstate(divide='ignore'):
    #     dw = numerator / targets
    #     dw[targets == 0] = 1

    goalmat = np.full(targets.shape, goal)
    with np.errstate(divide='ignore'):  # turn off divide-by-zero warning
        diff_weights = np.where(targets != 0, goalmat / targets, 1)

    return diff_weights


def get_diff_weights_vec(targets, goal=100):
    """
    difference weights - a weight to be applied to each target in the
      difference function so that it hits its goal
      set the weight to 1 if the target value is zero

    do this in a vectorized way
    """

    # avoid divide by zero or other problems

    # numerator = np.full(targets.shape, goal)
    # with np.errstate(divide='ignore'):
    #     dw = numerator / targets
    #     dw[targets == 0] = 1

    goalmat = np.full(targets.shape, goal)
    with np.errstate(divide='ignore'):  # turn off divide-by-zero warning
        diff_weights = np.where(targets != 0, goalmat / targets, 1)

    return diff_weights.flatten(order='F')


def get_targets(beta, wh, xmat):
    """
    Calculate matrix of target values by state and characteristic.

    Definitions:
    h: number of households
    k: number of characteristics each household has
    s: number of states or geographic areas

    Parameters
    ----------
    beta : matrix
        s x k matrix of coefficients for the poisson function that generates
        state weights.
    wh : vector
         1 x h vector of weights for each household.
    xmat : matrix
        h x k matrix of characteristics (data) for households.

    Returns
    -------
    targets_mat : matrix
        s x k matrix of target values.

    """
    delta = get_delta(wh, beta, xmat)
    whs = get_weights(beta, delta, xmat)
    targets_mat = np.dot(whs.T, xmat)
    return targets_mat


def get_weights(beta, delta, xmat):
    """
    Calculate state-specific weights for each household.

    Definitions:
    h: number of households
    k: number of characteristics each household has
    s: number of states or geographic areas

    See (Khitatrakun, Mermin, Francis, 2016, p.4)

    Parameters
    ----------
    beta : matrix
        s x k matrix of coefficients for the poisson function that generates
        state weights.
    delta : vector
        h-length vector of constants (one per household) for the poisson
        function that generates state weights.
    xmat : matrix
        h x k matrix of characteristics (data) for households.

    Returns
    -------
    matrix of dimension h x s.

    """
    # begin by calculating beta_x, an s x h matrix:
    #   each row has the sum over k of beta[s_i, k] * x[h_j, k]
    #     for each household where s_i is the state in row i
    #   each column is a specific household
    beta_x = np.dot(beta, xmat.T)

    # add the delta vector of household constants to every row
    # of beta_x and transpose
    # beta_xd <- apply(beta_x, 1, function(mat) mat + delta)
    beta_xd = (beta_x + delta).T

    weights = np.exp(beta_xd)

    return weights


def targets_diff(beta_obj, wh, xmat, targets, diff_weights):
    '''
    Calculate difference between calculated targets and desired targets.

    Parameters
    ----------
    beta_obj: vector or matrix
        if vector it will have length s x k and we will create s x k matrix
        if matrix it will be dimension s x k
        s x k matrix of coefficients for the poisson function that generates
        state weights.
    wh: array-like
        DESCRIPTION.
    xmat: TYPE
        DESCRIPTION.
    targets: TYPE
        DESCRIPTION.
    diff_weights: TYPE
        DESCRIPTION.

    Returns
    -------
    matrix of dimension s x k.

    '''
    if beta_obj.ndim == 1:
        beta = beta_obj.reshape(targets.shape)
    elif beta_obj.ndim == 2:
        beta = beta_obj

    targets_calc = get_targets(beta, wh, xmat)
    diffs = targets_calc - targets
    diffs = diffs * diff_weights

    # if this was called with a vector, return a vector
    if beta_obj.ndim == 1:
        diffs = diffs.flatten()

    return diffs

# make_problem <- function(h, s, k) {
#     # create a problem of a chosen size h: # of households s: # of states k: # of characteristics per household

#     # returns a list with items created below

#     # example call: make_problem(8, 3, 2)

#     set.seed(1234)
#     xmat <- matrix(stats::runif(h * k), nrow = h, byrow = TRUE)

#     set.seed(1234)
#     whs <- matrix(stats::runif(h * s, 10, 20), nrow = h, byrow = TRUE)

#     wh = rowSums(whs)
#     ws = colSums(whs)

#     targets <- t(whs) %*% xmat  # s x k

#     keepnames <- c("h", "s", "k", "xmat", "wh", "ws", "whs", "targets")
#     problem <- list()
#     for (var in keepnames) problem[[var]] <- get(var)
#     problem
# }


class Problem:
    """Problem elements."""

    def __init__(self, h, s, k):
        import numpy as np
        from numpy.random import seed
        self.h = h
        self.s = s
        self.k = k

        # prepare xmat
        seed(1)
        r = np.random.randn(h, k) / 100  # random normal)
        xmean = 100 + 20 * np.arange(0, k)
        self.xmat = xmean * (1 + r)

        self.whs = 10 + 10 * np.random.rand(h, s)
        self.wh = self.whs.sum(axis=1)
        self.ws = self.whs.sum(axis=0)
        self.targets = np.dot(self.whs.T, self.xmat)



# %% define data for simple r problem -- geoweight example


#  see cell further below for expected results
h = 10
s = 3
k = 2

wh = [43.45278, 51.24605, 39.08130, 47.52817, 44.98483,
      43.90340, 37.35561, 35.01735, 45.55096, 47.91773]

x1 = [0.113703411, 0.609274733, 0.860915384, 0.009495756, 0.666083758,
      0.693591292, 0.282733584, 0.292315840, 0.286223285, 0.186722790]

x2 = [0.6222994, 0.6233794, 0.6403106, 0.2325505, 0.5142511,
      0.5449748, 0.9234335, 0.8372956, 0.2668208, 0.2322259]

xmat = np.array([x1, x2]).T

targets = np.array([[55.50609, 73.20929],
                    [61.16143, 80.59494],
                    [56.79071, 75.41574]])

# for test purpose, also define a matrix that has some zeroes
targetsz = np.array([[55.50609, 73.20929],
                     [0, 80.59494],
                     [56.79071, 0]])


# %% check, using data from r problem
wh
xmat
xmat.T
targets
targetsz

np.sum(xmat, axis=0)  # colsums
xmat.sum(axis=0)
np.sum(xmat, axis=1)  # rowsums

dw = get_diff_weights(targets)
dw
targets * dw
# get_diff_weights(targetsz)  # test the case where we have zeroes
targetsz * get_diff_weights(targetsz)

beta0 = np.zeros([s, k])
beta0

delta0 = get_delta(wh, beta0, xmat)
delta0

whs0 = get_weights(beta0, delta0, xmat)
whs0

targs0 = get_targets(beta0, wh, xmat)
targs0

targs0 - targets

# we can pass either a matrix or an array to targets_diff
# return type will be same as input type
targets_diff(beta0, wh, xmat, targets, diff_weights=dw)
targets_diff(beta0.flatten(), wh, xmat, targets, diff_weights=dw)

# bv2 = np.array(np.arange(1, 13))
# bv2

res = least_squares(targets_diff, beta0.flatten(),
                    args=(wh, xmat, targets, dw))
dir(res)
res.nfev
res.njev
res.fun
res.cost
res.message
res.optimality
res.x

# res = res1
beta_opt = res.x.reshape(targets.shape)
beta_opt
delta_opt = get_delta(wh, beta_opt, xmat)
delta_opt
whs_opt = get_weights(beta_opt, delta_opt, xmat)
whs_opt
targets_opt = get_targets(beta_opt, wh, xmat)
targets_opt
targets
targets_opt - targets
targets_diff(beta_opt, wh, xmat, targets, diff_weights=dw)

# method{'trf', 'dogbox', 'lm'}, optional
# dogbox is a lot more nfev
res1 = least_squares(targets_diff, beta0.flatten(),
                     args=(wh, xmat, targets, dw))
res1 = least_squares(targets_diff, beta0.flatten(),
                     method='lm',
                     args=(wh, xmat, targets, dw))
res1 = least_squares(targets_diff, beta0.flatten(),
                     method='dogbox',
                     args=(wh, xmat, targets, dw))
res1 = least_squares(targets_diff, beta0.flatten(),
                     method='trf',
                     args=(wh, xmat, targets, dw))
res1 = least_squares(targets_diff, beta0.flatten(),
                     method='trf', jac='3-point', verbose=2,
                     args=(wh, xmat, targets, dw))
res1 = least_squares(targets_diff, beta0.flatten(),
                     method='trf', jac='cs', verbose=2,
                     args=(wh, xmat, targets, dw))
res1.nfev
res1.njev
res1.fun
res1.cost
res1.message



# %% results from r problem - for checking against

# dw from get_dweights should be:
# 1.801604 1.635017 1.760851 1.365947 1.240773 1.325983

# delta when the beta matrix is 0 should be:
# 2.673062, 2.838026, 2.567032, 2.762710, 2.707713,
#     2.683379, 2.521871, 2.457231, 2.720219, 2.770873

# state weights when beta is 0 and we use the associated delta:
# > whs0
#           [,1]     [,2]     [,3]
#  [1,] 14.48426 14.48426 14.48426
#  [2,] 17.08202 17.08202 17.08202
#  [3,] 13.02710 13.02710 13.02710
#  [4,] 15.84272 15.84272 15.84272
#  [5,] 14.99494 14.99494 14.99494
#  [6,] 14.63447 14.63447 14.63447
#  [7,] 12.45187 12.45187 12.45187
#  [8,] 11.67245 11.67245 11.67245
#  [9,] 15.18365 15.18365 15.18365
# [10,] 15.97258 15.97258 15.97258

# targets when beta is 0
#          [,1]     [,2]
# [1,] 57.81941 76.40666
# [2,] 57.81941 76.40666
# [3,] 57.81941 76.40666

# sse_weighted 5.441764e-21

# $beta_opt_mat
#             [,1]        [,2]
# [1,] -0.02736588 -0.03547895
# [2,]  0.01679640  0.08806331
# [3,] -0.05385230  0.03097379

# $targets_calc
#          [,1]     [,2]
# [1,] 55.50609 73.20929
# [2,] 61.16143 80.59494
# [3,] 56.79071 75.41574

# $whs (optimal)
#           [,1]     [,2]     [,3]
#  [1,] 13.90740 15.09438 14.45099
#  [2,] 16.34579 18.13586 16.76441
#  [3,] 12.42963 13.97414 12.67753
#  [4,] 15.60913 16.07082 15.84823
#  [5,] 14.44566 15.85272 14.68645
#  [6,] 14.06745 15.51522 14.32073
#  [7,] 11.70919 13.28909 12.35734
#  [8,] 11.03794 12.39991 11.57950
#  [9,] 14.90122 15.59650 15.05323
# [10,] 15.72018 16.31167 15.88589

# %% arbitrary problems
# p = Problem(h=1000, s=5, k=2)  # small test problem
p = Problem(h=30000, s=50, k=30)  # about as large as we'll need to solve
# time: 1890 secs trf 2-point
# nfev 19
# njev 12

betavec0 = np.zeros(p.s * p.k)
dw = get_diff_weights(p.targets)

start1 = timer()
res1 = least_squares(targets_diff, betavec0,
                     method='lm',
                     args=(p.wh, p.xmat, p.targets, dw))
end1 = timer()
print(end1 - start1)
res1.nfev
res1.njev

start2 = timer()
res2 = least_squares(targets_diff, betavec0,
                     method='trf', jac='2-point', verbose=2,
                     args=(p.wh, p.xmat, p.targets, dw))
end2 = timer()
print(end2 - start2)  # trf 3-point 470, cs ng, 2-point 193
res2.nfev  # trf 3-point 38, cs ng, 2-point 23
res2.njev  # trf 3-point 20, cs ng, 2-point 16

res = res2
beta_opt = res.x.reshape(p.targets.shape)
beta_opt
delta_opt = get_delta(p.wh, beta_opt, p.xmat)
delta_opt
whs_opt = get_weights(beta_opt, delta_opt, p.xmat)
whs_opt
targets_opt = get_targets(beta_opt, p.wh, p.xmat)
targets_opt
p.targets
diff = targets_opt - p.targets
np.square(diff).sum()
(diff**2).sum()
res.cost
res.nfev
res.njev
dir(res)
# targets_diff(beta_opt, wh, xmat, targets, diff_weights=dw)


# %% experiment


p = Problem(10, 3, 2)
p.xmat
p.whs
p.wh
p.ws
p.targets
p.h
p.s
p.k
p.xmat


xmat = np.array([[1, 2],
                 [4, 5],
                 [7, 8]])

x = np.array([[1, 2],
              [4, 5],
              [7, 8]])
np.ones(2)
x + np.ones(2)
x + np.array([7, 11])

wh = np.array(range(1, 4)).T
wh

h = 10
k = 2
s = 3
# wh = random.randrange(100, 1210)
seed(1)
wh = 100 * (1 + rand(h))
wh

# wh = [41, 50, 29, 37, 81, 30, 73, 63, 20, 35, 68, 22, 60, 31, 95]

seed(2)
beta = np.zeros([s, k])
beta
# xmat = np.random.rand(h, k)
x1 = 10 * (1 + (rand(h) - .5) / 10)
x2 = 30 * (1 + (rand(h) - .5) / 10)

xmat = np.array([x1, x2]).T
xmat

beta @ xmat.T  # matrix multiplication


xmat[1, ]

xmat
np.transpose(xmat)
xmat.T

wh * np.transpose(xmat)
np.dot(wh, np.transpose(xmat))

dir(xmat)
xmat[0]
xmat[2]
xmat[3]
xmat[0, 1]
xmat.shape
xmat.size
xmat.ndim
xmat.reshape(xmat.size)
xmat.flatten()

xvec = xmat.flatten()
xmat2 = xvec.reshape(xmat.shape)
xmat
xmat2


xmat.reshape(xmat.size, order='F')
