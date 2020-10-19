# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 08:58:32 2020

@author: donbo

qrake calculates geographic area weights for a national microdata file
such that:

    (1) the sum of the area weights for each household equals the household's
    national weight, and

    (2) the weighted totals for selected variables equal or come as close
    as practical to target totals, for each area, for each variable.

The method is based largely on the method described in:

    Randrianasolo, Toky, and Yves Tillé. “Small Area Estimation by Splitting
    the Sampling Weights.” Electronic Journal of Statistics 7, no. 0
    (2013): 1835–55. https://doi.org/10.1214/13-EJS827.

The code is based largely on R code provided by Toky Randrianasol to
Don Boyd by email on on October 1, 2020, and ported to python by Boyd
and subsequently revised by Boyd.

I would recommend citing the above paper and the authors' code in any
public work that uses the code below.

The main deviation from the R code is that the R code used the raking method
of the R function calib from the R package sampling (written by Yves Tillé).
Instead, I use the python function maybe_exact_calibrate from the python
package empirical_calibration.

The empirical_calibration package is described in the paper:

    Wang, Xiaojing, Jingang Miao, and Yunting Sun. “A Python Library For
    Empirical Calibration.” ArXiv:1906.11920 [Stat], July 25, 2019.
    http://arxiv.org/abs/1906.11920.

I use maybe_exact_calibrate in my main approach because:

    (1) It appears to be very robust. In particular, if it cannot find a
    solution that fully satisfies every target, it automatically relaxes
    constraints to find a very good solution that is close to satisfying
    targets.

    (2) It allows us to assign priority weights to targets, which may be
    useful in the future.

    (3) As with the calib raking method, it can solve for a set of weighting
    factors that are near to baseline weights.

    (4) It is very fast, although I have not tested it to see whether it
    is faster than calib's raking method.

The function also has an autoscale option intended to reduce potential
numerical difficulties, but in my experimentation it has not worked well
so I do not use this option in the code below.

One important adjustment: maybe_exact_calibrate solves for weights that
hit or come close to target weighted means, whereas normally we specify
our problems so that the targets are weighted totals. I take this into
account by:

    (1) Converting area target totals to are target means by dividing
    each area target total by the area target population,

    (2) Solving for mean-producing weights, and

    (3) Converting to sum-producing weights by multiplying the
    mean-producing weights by the area target population.

The python code for maybe_exact_calibrate is at:

    https://github.com/google/empirical_calibration/blob/master/empirical_calibration/core.py

I have also ported the R code for calib's raking method (but not other
methods) to python, and provide that as a backup method.

The empirical_calibration package can be installed with:

    pip install -q git+https://github.com/google/empirical_calibration


****** Here is a summary of how the code works. ******

N:  number of households (tax returns, etc.), corresponding index is i
D:  number of areas (states, etc.), corresponding index is j
w:  vector of national household weights of length N, indexed by i

[NOTE: must update code below to use proper i and j indexes]

Q:  an N x D matrix of proportions of national weights, where:
    Q[i, j] is the proportion of household i's national weight w[i] that is
        in area j

    The sum of each row of Q must be 1. This imposes the requirement that
    a household's area weights must sum to the household's national weights,
    for every household.

In the code below:

We need the equivalent of the g weights returned by R's calib function.
See https://www.rdocumentation.org/packages/sampling/versions/2.8/topics/calib.

Those values, when multiplied by population weights, give sample weights that
hit or come close to desired targets.

At each iteration, for each state i, we multiply Q[i, ] by g to get updated
Q[i, ] for next iteration, where Q[i, k] .

The return from ec.maybe_exact_calibrate

https://github.com/google/empirical_calibration/blob/master/empirical_calibration/core.py
https://github.com/google/empirical_calibration/blob/master/notebooks/survey_calibration_cvxr.ipynb

"""


# %% imports

import warnings
import numpy as np
import pandas as pd

import puf.puf_constants as pc
import src.make_test_problems as mtp
import src.microweight as mw
import experimental_code.geoweight as gw
from timeit import default_timer as timer

# pip install -q git+https://github.com/google/empirical_calibration
import empirical_calibration as ec


# %% constants

SMALL_POSITIVE = np.nextafter(np.float64(0), np.float64(1))
# not sure if needed: a small nonzero number that can be used as a divisor
SMALL_DIV = SMALL_POSITIVE * 1e16
# 1 / SMALL_DIV  # does not generate warning

QUADRATIC = ec.Objective.QUADRATIC
ENTROPY = ec.Objective.ENTROPY

STLIST = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
          'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
          'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
          'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
          'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']

qtiles = [0, .01, .1, .25, .5, .75, .9, .99, 1]


# %% classes and functions

class Result:
    pass


def get_geo_data():
    IGNOREDIR = 'C:/programs_python/weighting/puf/ignore/'

    ht2sub= pd.read_csv(IGNOREDIR + 'ht2sub.csv')
    ht2sums= pd.read_csv(IGNOREDIR + 'ht2sums.csv')

    pufsub = pd.read_csv(IGNOREDIR + 'pufsub.csv')
    pufsums= pd.read_csv(IGNOREDIR + 'pufsums.csv')
    return ht2sub, ht2sums, pufsub, pufsums


def prep_stub(stub, targvars, targstates, ht2sub, ht2sums, pufsub, pufsums):
    # get wh, xmat, and targets

    pufstub = pufsub.query('HT2_STUB == @stub')[['pid', 'HT2_STUB', 'wtnew'] + targvars]
    ptot = pufsums.query('HT2_STUB ==@stub')[targvars]

    ht2stub = ht2sub.query('HT2_STUB == @stub & STATE != "US"')[['STATE', 'HT2_STUB'] + targvars]
    htot = ht2sums.query('HT2_STUB ==@stub')[targvars]
    # ptot / htot

    # collapse ht2stub to target states and XX which will be all other
    mask = np.logical_not(ht2stub['STATE'].isin(targstates))
    ht2stub.loc[mask, 'STATE'] = 'XX'
    ht2stub = ht2stub.groupby(['STATE', 'HT2_STUB']).sum()

    # prepare the return values
    wh = pufstub.wtnew.to_numpy()
    xmat = np.asarray(pufstub[targvars], dtype=float)
    targets = np.asarray(ht2stub, dtype=float)
    return wh, xmat, targets

def gec(xmat, wh, targets,
        target_weights: np.ndarray = None,
        objective: ec.Objective = ec.Objective.ENTROPY,
        increment: float = 0.001):

    # small_positive = np.nextafter(np.float64(0), np.float64(1))
    wh = np.where(wh == 0, SMALL_POSITIVE, wh)

    pop = wh.sum()
    tmeans = targets / pop

    # ompw:  optimal means-producing weights
    ompw, l2_norm = ec.maybe_exact_calibrate(
        covariates=xmat,
        target_covariates=tmeans.reshape((1, -1)),
        baseline_weights=wh,
        # target_weights=np.array([[.25, .75]]), # target priorities
        target_weights=target_weights,
        autoscale=True,  # doesn't always seem to work well
        # note that QUADRATIC weights often can be zero
        objective=objective,  # ENTROPY or QUADRATIC
        increment=increment
    )
    # print(l2_norm)

    # wh, when multiplied by g, will yield the targets
    g = ompw * pop / wh
    g = np.array(g, dtype=float).reshape((-1, ))  # djb

    return g


def get_drops(targets, drop_dict):
    drops = np.zeros(targets.shape, dtype=bool) # start with all values False
    if drop_dict is not None:
        for row, cols in drop_dict.items():
            drops[row, cols] = True
    return drops

# STOP_TRUE = (10 < 20) and (5 > 6)


def qrake(Q, wh, xmat, targets,
          method='raking', maxiter=200, drops=None,
          objective: ec.Objective = ec.Objective.ENTROPY,):
    """

    Parameters
    ----------
    Q : 2d array
        DESCRIPTION.
    w : 1d array
        DESCRIPTION.
    xmat : TYPE
        DESCRIPTION.
        Note: this was Xs in the R code.
    targets : TYPE
        Note that this was TTT in the R code provided by Toky Randrianasolo.

    Returns
    -------
    Q : TYPE
        DESCRIPTION.

    """

    # TODO:
    #   customize stopping rule based only on max target pct diff

    def print_problem():
        print(' Number of households:                {:8,}'.format(wh.size))
        print(' Number of areas:                     {:8,d}'.format(m))
        print()
        print(' Number of targets per area:          {:8,d}'.format(nt_per_area))
        print(' Number of potential targets, total:  {:8,d}'.format(nt_possible))
        print(' Number of targets dropped:           {:8,d}'.format(nt_dropped))
        print(' Number of targets used:              {:8,d}'.format(nt_used))

    a = timer()

    if method == 'raking':
        gfn = rake
        objective = None
    elif method == 'raking-ec':
        gfn = gec

    # constants
    # EPS = 1e-5  # acceptable weightsum error (tolerance) - 1e-5 in R code
    TOL_WTDIFF = 0.0005  # tolerance for difference between weight sum and 1
    TOL_TARGPCTDIFF = 1.0  # tolerance for targets percent difference

    # initialize stopping criteria values
    ediff = 1  # error, called ver in Toky R code
    iter = 1  # initialize iteration count called k in Toky R. R code
    iter_best = iter

    # difference in weights - Toky R. used sum of absolute weight differences,
    # I use largest absolute weight difference
    max_weight_absdiff = 1e9  # initial maximum % difference between sums of weights for a household and 100
    max_targ_abspctdiff = 1e9  # initial maximum % difference vs targets
    max_diff_best = max_targ_abspctdiff

    m = targets.shape[0]  # number of states
    n = wh.size  # number of households
    wh = wh.reshape((-1, 1))  # ensure the proper shape

    # compute xmat_wh before loop to(calib calculates it in the loop)
    xmat_wh = xmat * wh  # shape -  n x number of targets

    # numbers of targets
    nt_per_area = targets.shape[1]
    nt_possible = nt_per_area * m
    if drops is None:
        drops= np.zeros(targets.shape, dtype=bool)  # all False
        nt_dropped = 0
    else:
        # nt_dropped = sum([len(x) for x in drops.values()])
        nt_dropped = drops.sum()
    nt_used = nt_possible - nt_dropped
    good_targets = np.logical_not(drops)

    def end_loop(iter, max_targ_abspctdiff):
        iter_rule = (iter > maxiter)
        target_rule = (max_targ_abspctdiff <= TOL_TARGPCTDIFF)
        no_more = iter_rule or target_rule
        return no_more

    # Making a copy of Q is crucial. We don't want to change the
    # original Q.
    Q = Q.copy()
    Q_best = Q.copy()

    print('')
    print_problem()

    h1 = "                  max weight      max target       p95 target"
    h2 = "   iteration        diff           pct diff         pct diff"
    print('\n')
    print(h1)
    print(h2, '\n')

    while not end_loop(iter, max_targ_abspctdiff):

        print(' '*3, end='')
        print('{:4d}'.format(iter), end='', flush=True)

        for j in range(m):  # j indexes areas

            good_cols = good_targets[j, :]

            g = gfn(xmat_wh[:, good_cols], Q[:, j], targets[j, good_cols], objective=objective)
            # print(type(g))
            # if method == 'raking' and g is None:
            #     # try to recover by using the alternate method
            #     g = gec(xmat_wh[:, good_cols], Q[:, j], targets[j, good_cols])
            if g is None:
                g = np.ones(n)

            if np.isnan(g).any() or np.isinf(g).any() or g.any() == 0:
                g = np.ones(g.size)
                # we'll need to do this one again
            else:
                pass
                # print("done with this area")
            Q[:, j] = Q[:, j] * g.reshape(g.size, )  # end for loop

        # we have completed all areas for this iteration
        # calc max weight difference BEFORE recalibrating Q
        abswtdiff = np.abs(Q.sum(axis=1) - 1)  # sum of weight-shares for each household
        max_weight_absdiff = abswtdiff.max()  # largest sum across all households
        print(' '*11, end='')
        print(f'{max_weight_absdiff:8.4f}', end='')
        if np.isinf(abswtdiff).any():
            # these weight shares are not good, do another iteration
            # ediff = EPS
            max_weight_absdiff = TOL_WTDIFF
            print("Existence of infinite coefficients --> non-convergence.")

        #print("Weight sums max percent difference: {}".format(maxadiff))  # ediff
        Q = Q / Q.sum(axis=1)[:, None]  # Recalibrate Q. Note None so that we have proper broadcasting

        # calculate targets pct diff AFTER recalibrating Q
        whs = np.multiply(Q, wh.reshape((-1, 1)))  # faster
        diff = np.dot(whs.T, xmat) - targets
        abspctdiff = np.abs(diff / targets * 100)
        max_targ_abspctdiff = abspctdiff[good_targets].max()

        ptile = np.quantile(abspctdiff[good_targets], (.95))
        print(' '*6, end='')
        print(f'{max_targ_abspctdiff:8.2f} %', end='')
        print(' '*7, end='')
        print(f'{ptile:8.2f} %')

        # final processing before next iteration
        if max_targ_abspctdiff < max_diff_best:
            Q_best = Q.copy()
            max_diff_best = max_targ_abspctdiff.copy()
            iter_best = iter
        iter = iter + 1
        # end while loop

    # post-processing after exiting while loop, using Q_best, not Q
    whs_opt = np.multiply(Q_best, wh.reshape((-1, 1)))  # faster
    targets_opt = np.dot(whs_opt.T, xmat)
    diff = targets_opt - targets
    pctdiff = diff / targets * 100
    abspctdiff = np.abs(pctdiff)
    # calculate weight difference AFTER final calibration
    abswtdiff = np.abs(Q_best.sum(axis=1) - 1)  # sum of weight-shares for each household
    max_weight_absdiff = abswtdiff.max()  # largest sum across all households

    if iter > maxiter:
        print('\nMaximum number of iterations exceeded.\n')

    print('\n')
    print_problem()

    print(f'\nPost-calibration max abs diff between sum of household weights and 1, across households: {max_weight_absdiff:9.5f}')
    print()

    # compute and print good and all values for various quantiles
    p100a = abspctdiff.max()
    p100m = abspctdiff[good_targets].max()
    p99a = np.quantile(abspctdiff, (.99))
    p99m = np.quantile(abspctdiff[good_targets], (.99))
    p95a = np.quantile(abspctdiff, (.95))
    p95m = np.quantile(abspctdiff[good_targets], (.95))
    print('Results for calculated targets versus desired targets:')
    print( '                                                              good             all\n')
    print(f'    Max abs percent difference                           {p100m:9.3f} %     {p100a:9.3f} %')
    print(f'    p99 of abs percent difference                        {p99m:9.3f} %     {p99a:9.3f} %')
    print(f'    p95 of abs percent difference                        {p95m:9.3f} %     {p95a:9.3f} %')
    print(f'\nNumber of iterations:                       {iter - 1:5d}')
    print(f'Best target difference found at iteration:  {iter_best:5d}')

    b = timer()
    print('\nElapsed time: {:8.1f} seconds'.format(b - a))

    # create a dict of items to return
    res = {}
    res['esecs'] = b - a
    res['Q_opt'] = Q_best
    res['whs_opt'] = whs_opt
    res['targets_opt'] = targets_opt
    res['pctdiff'] = pctdiff
    res['iter_opt'] = iter_best

    return res


def rake(Xs, d, total, q=1, objective=None):
    # this is a direct translation of the raking code of the calib function
    # in the R sampling package, as of 10/3/2020
    # Xs the matrix of covariates
    # d vector of initial weights
    # total vector of targets
    # q vector or scalar related to heteroskedasticity
    # returns g, which when multiplied by the initial d gives the new weight
    EPS = 1e-15  # machine double precision used in R
    EPS1 = 1e-8  # R calib uses 1e-6
    max_iter = 10

    # make sure inputs all have the right shape
    d = d.reshape((-1, 1))
    total = total.reshape((-1, 1))

    lam = np.zeros((Xs.shape[1], 1))  # lam is k x 1
    w1 = d * np.exp(np.dot(Xs, lam) * q) # h(n) x 1

    # set initial value for g (djb addition to program)
    g = np.ones(w1.size)

    # operands could not be broadcast together with shapes (20,1) (100,1)
    for i in range(max_iter):
        phi = np.dot(Xs.T, w1) - total  # phi is 1 col matrix
        T1 = (Xs * w1).T # T1 has k(m) rows and h(n) columns
        phiprim = np.dot(T1, Xs) # phiprim is k x k
        lam = lam - np.dot(np.linalg.pinv(phiprim, rcond = 1e-15), phi) # k x 1
        w1 = d * np.exp(np.dot(Xs, lam) * q)  # h(n) x 1; in R this is a vector??
        if np.isnan(w1).any() or np.isinf(w1).any():
            warnings.warn("No convergence bad w1")
            g = None
            break
        tr = np.inner(Xs.T, w1.T) # k x 1
        if np.max(np.abs(tr - total) / total) < EPS1:
            break
        if i==max_iter:
            warnings.warn("No convergence after max iterations")
            g = None
        else:
            g = w1 / d  # djb: what if d has zeros?
        # djb temporary solution: force g to be float
        # TODO: explore where we have numerical problems and
        # fix them
        g = np.array(g, dtype=float)  # djb
        g = g.reshape((-1, ))
        # end of the for loop

    return g


# %% set up a random problem
p = mtp.Problem(h=20, s=3, k=2, xsd=.1, ssd=.5)
p = mtp.Problem(h=100, s=6, k=3, xsd=.1, ssd=.5)
p = mtp.Problem(h=1000, s=10, k=5, xsd=.1, ssd=.5)
p = mtp.Problem(h=10000, s=20, k=10, xsd=.1, ssd=.5)
p = mtp.Problem(h=20000, s=20, k=10, xsd=.1, ssd=.5)
p = mtp.Problem(h=30000, s=50, k=20, xsd=.1, ssd=.5)
p = mtp.Problem(h=300000, s=50, k=40, xsd=.1, ssd=.5)

targets = p.targets.copy()
xmat = p.xmat.copy()
wh = p.wh.copy()


s = 0
ratios = targets[s, ] / targets.sum(axis=0)
iwh = wh * ratios[0]
iwh[0] = 0

# pvals = np.ones((1, targets[s, ].size))
# pvals = np.array([[1, .1]])

t2 = targets[s, ].copy()
t2[0] = 0
t2[1] = 56.4e3
t2[1] = 57e3
t2
t2[1] = np.inf
t2[1] = 1e9

pvals = np.array([[1, 1e-9]])
pvals

# initial diffs and percent diffs
np.dot(xmat.T, iwh) - t2
np.round((np.dot(xmat.T, iwh) - t2) / t2 * 100, 3)

g0 = gec(xmat, iwh, targets[s, ]); g0
g1 = gec(xmat, iwh, t2, increment=.001); g1
g1 = gec(xmat, iwh, t2, objective=QUADRATIC, increment=.001); g1  # best, if we must keep a bad target
g1 = gec(xmat, iwh, t2, objective=ENTROPY, increment=.001); g1
g1 = rake(xmat, iwh, t2); g1
g1 = gec(xmat, iwh, t2, increment=.0001); g1
g1 = gec(xmat, iwh, t2, increment=1e-13); g1

g1 = np.where(np.isnan(g1), 0, g1)

np.dot(xmat.T, iwh * g1) - t2
np.round((np.dot(xmat.T, iwh * g1) - t2) / t2 * 100, 3)


# %% drop a target for this state
# make a list with row numbers of the targets matrix and column numbers we will drop
targets.shape
targets

# let's target state 3 (row 4) and cols 2 and 4, which we will set to really bad numbers
drops = {0: (1, 2),
         1: (1, 3),
         3: (2, 4)}

s = 3
badtargs = list(drops.get(s))  # IMPORTANT: convert to list or use lists from start
ratios = targets[s, ] / targets.sum(axis=0)
iwh = wh * ratios[0]
# iwh[0] = 0

# pvals = np.ones((1, targets[s, ].size))
# pvals = np.array([[1, .1]])

# now make a vector that has bad values for the target and another vector
# that drops those values, for which we will create an xmat with dropped cols
t2 = targets[s, ].copy()
t2[badtargs]
t2[badtargs].shape

t2[badtargs] * [.5, 1.5]
badvals = t2[badtargs] * (.5, 1.5)
t2[badtargs] = badvals

badtargs = list(drops.get(s))
t2 = targets[s, ].copy()
t3 = np.delete(t2, badtargs, 0)
xmat3 = np.delete(xmat, badtargs, 1)


# review
targets[s, ]
t2
t3
xmat.shape
xmat3.shape

# initial diffs and percent diffs
np.dot(xmat.T, iwh) - t2
np.round((np.dot(xmat.T, iwh) - t2) / t2 * 100, 3)

g0 = gec(xmat, iwh, targets[s, ]); g0
g0r = rake(xmat, iwh, t2); g0r
g2e = gec(xmat, iwh, t2, objective=ENTROPY, increment=.001); g2e # bad
g2q = gec(xmat, iwh, t2, objective=QUADRATIC, increment=.001); g2q  # best, if we must keep a bad target
g3e = gec(xmat3, iwh, t3, objective=ENTROPY, increment=.001); g3e
g3q = gec(xmat3, iwh, t3, objective=QUADRATIC, increment=.001); g3q
g3r = rake(xmat3, iwh, t3); g3r

# diffs and percent diffs with different g's (and xmat where appropriate)
np.dot(xmat.T, iwh * g0) - t2  # bad on the bad targets
np.dot(xmat.T, iwh * g0r) - t2  # no good
np.dot(xmat.T, iwh * g2e) - t2  # no good
np.dot(xmat.T, iwh * g2q) - t2  # better
np.round((np.dot(xmat.T, iwh * g2q) - t2) / t2 * 100, 3)

xmat.shape
xmat[:, 0:2].shape
np.dot(xmat.T, iwh * g0)

# import numpy.ma as ma
# look at masked arrays as possible alternative
# https://numpy.org/doc/stable/reference/maskedarray.generic.html


# create a mask-like array with shape of targets, where bad values are False
# bad value indices will be a list of tuples
# each tuple as an integer as its first value (a state index)
# and a tuple of integers as 2nd value (indexes of bad columns)
mask= np.ones(targets.shape, dtype=bool)
bads = [(1, (1)), (3, (1, 2))]
# mask[bads]  # we CANNOT do this
for ij in bads: mask[ij] = False
mask

targets[mask]
targets.sum()
targets[mask].sum()


# To create an array with the second element invalid, we would do:

# drops on the kept targets
np.dot(xmat3.T, iwh * g0) - t3
np.dot(xmat3.T, iwh * g3e) - t3
np.dot(xmat3.T, iwh * g3q) - t3
np.dot(xmat3.T, iwh * g3r) - t3  # good enough but not as close as others
np.round((np.dot(xmat3.T, iwh * g3r) - t3) / t3 * 100, 3)

# drops on all targets
np.dot(xmat.T, iwh * g2q) - t2  # best undropped
np.dot(xmat.T, iwh * g3e) - t2  # dropped

np.round((np.dot(xmat.T, iwh * g2q) - t2) / t2 * 100, 3)
np.round((np.dot(xmat.T, iwh * g3e) - t2) / t2 * 100, 3)

gec(xmat, iwh, t2, target_weights=None)

targets[s, ]
t2
np.dot(xmat.T, iwh * g0)
np.dot(xmat.T, iwh * g1)
np.dot(xmat.T, iwh * g1) - t2
np.round((np.dot(xmat.T, iwh * g1) - t2) / t2 * 100, 3)
np.dot(xmat.T, iwh * g2)


gec2(xmat, iwh, t2)
gec2(xmat, iwh, t2)


gec(xmat, iwh, targets[s, ], pvals)
gec2(xmat, iwh, targets[s, ])

%timeit gec(xmat, iwh, targets[s, ])
%timeit gec2(xmat, iwh, targets[s, ])


# %% create some drops
# { keyName1 : value1, keyName2: value2, keyName3: [val1, val2, val3] }
# which columns should we drop, for which states
drops = {'AK': (1, 2),
         'XX': (1, 3)}

drops = {0: (1, 2),
         1: (1, 3)}

drops
list(drops)
for k, v in drops.items():
    print(k, v)

drops.get("XX")
drops.get(0)
drops['XX']
drops[0]


# %% test rake vs ec

tnew = targets.sum(axis=0)
np.random.seed(1)
noise = np.random.normal(0, .01, tnew.size)
tnew2 = tnew * (1 + noise)

tnew.shape
tnew2.shape
xmat.shape

g1 = rake(xmat, wh, tnew)

ar = timer()
g2 = rake(xmat, wh, tnew2)  # .reshape((-1,))
br = timer()
br - ar
g1
g2

aec = timer()
g3 = gec(xmat, wh, tnew2)
bec = timer()
bec - aec


tnew - np.dot(xmat.T, wh * g1)
diffr = tnew2 - np.dot(xmat.T, wh * g2)
diffec = tnew2 - np.dot(xmat.T, wh * g3)
# np.dot(xmat.T, ompw*pop)
np.square(diffr).sum()
np.square(diffec).sum()

wdiffr = wh - wh * g2
wdiffec = wh - wh * g3
np.square(wdiffr).sum()
np.square(wdiffec).sum()

wdiffr = (wh / (wh * g2) - 1)
wdiffec = (wh / (wh * g3) - 1)
np.square(wdiffr).sum()
np.square(wdiffec).sum()

np.quantile(g2, qtiles)
np.quantile(g3, qtiles)

# quadratic
# array([0.        , 0.50555846, 0.74603538, 1.00321358, 1.25328969,
#        1.4856983 , 2.5232894 ])

# entropy
# array([0.20711579, 0.56915897, 0.72323304, 0.933203  , 1.19828925,
#        1.50852965, 4.22583624])


# %% check targets and data
# targ_US = targets.sum(axis=0)
# puf_US = np.dot(xmat.T, wh)
# puf_US / targ_US * 100 - 100
# targvars


# %% get all agi data

# stub counts
# HT2_STUB
# 1          5397
# 2         23527
# 3         35277
# 4         40316
# 5         26183
# 6         18302
# 7         31866
# 8         18052
# 9         12639
# 10        29686

ht2sub, ht2sums, pufsub, pufsums = get_geo_data()


# %% set up agistub problem for qrake (devised in puf_geoweight up)

stub = 7

# choose a definition of targvars and targstates
targvars = pc.targvars_all
targstates = pc.STATES

badstates = ['AK', 'ND', 'SD', 'UT', 'WY']
targstates = [x for x in pc.STATES if x not in badstates]

wh, xmat, targets = prep_stub(stub, targvars, targstates,
                              ht2sub=ht2sub, ht2sums=ht2sums,
                              pufsub=pufsub, pufsums=pufsums)
wh.shape
xmat.shape
targets.shape

# stubs, 7 variables, using all states
# qtiles of target pct diff [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
# 1
# init array([-67.1, -19.1,  -3.1,   5.4,  36.9,  84.7, 564.4])
# opt array([-67.6,  -1.3,  -0.4,   0.1,   1.7,   8.5,  67.7])
# array([-5.510e+01, -1.500e+00, -6.000e-01, -1.000e-01,  1.200e+00, 7.000e+00,  1.007e+02])
# opt array([-100. ,  -70.8,    3.3,   10.1,   17.3,   27.6, 2090.8])

# 2
# init array([-83.1, -14.3,  -3.6,  -0. ,  7.1,  26.9,  90.4])
# opt array([-0.3, -0.1, -0. , -0. ,  0. ,  0. ,  0.8])

# 3
# init array([-77.5, -20.7,  -5.2,  -0. ,   2.2,  17.3, 157.3])
# opt array([-0., -0., -0., -0., -0.,  0.,  1.])

# 4
# init array([-73.5, -15.1,  -4.6,  -0. ,   2.9,  15.9,  95.2])
# opt array([-0., -0., -0., -0.,  0.,  0.,  0.])

# 5
# init array([-62.9, -15.3,  -4.3,  -0. ,   4. ,  19.1,  76.4])
# opt array([-0., -0., -0., -0., -0.,  0.,  0.])

# 6
# init array([-56.4, -15. ,  -3.5,  -0. ,   4.1,  28.5,  86.9])
# opt array([-0., -0., -0., -0.,  0.,  0.,  0.])

#7
# init array([-54.3,  -9.6,  -2.2,   0. ,   6.3,  28.7,  96.3])
# opt array([-0., -0., -0.,  0.,  0.,  0.,  0.])

# 8
# init array([-0., -0., -0., -0.,  0.,  0.,  0.])
# opt array([-0., -0., -0., -0.,  0.,  0.,  0.])

#9
# init array([-52.2, -11.9,  -2.7,  -0. ,   7.7,  22.3,  87.7])
# opt array([-0.1, -0. , -0. , -0. ,  0. ,  0. ,  0. ])

# 10
# init array([-87.4, -14.8,  -2. ,   5.8,  29.8,  64.7, 169.6])
# opt array([-8.8, -0.6, -0. , -0. , -0. ,  0.2, 26.4])
# opt array([-9.0700e+01, -2.5500e+01, -2.0000e-01,  0.0000e+00,  1.0000e-01,
#             1.1300e+01,  1.6734e+03])

n = xmat.shape[0]
m = targets.shape[0]

# treat first column of targets as if it is the population of each state
# we we can get each state's share of pop from this, and use that to start Q
# shares = targets[:, 0] / targets[:, 0].sum()  # each state's share of total
# # shares.sum()

# Q = np.tile(shares, n).reshape((n, m))
# n = xmat.shape[0]
# m = targets.shape[0]
# Q = np.tile(shares, n).reshape((n, m))
# Q.shape
# Q.sum(axis=1)

# create a better starting Q that bases shares on whether a person
# is single, MFJ, other
targvars  # 0, 1, 2 are total, single, mfj
targets[:, 0:3] / targets[:, 0:3].sum(axis=0)  # each state's share of return type
xmat[0:10, 0:4]
xmat[:, 0:3].sum(axis=0)  # sums of returns by

targ_mstat = np.ndarray((m, 3))
targ_mstat[:, 1:3] = targets[:, 1:3]
targ_mstat[:, 0] = targets[:, 0] - targets[:, 1:3].sum(axis=1)  # create other
targ_mstat.sum(axis=0)
# each state's share of total by marital status other, single, mfj
shares_mstat = targ_mstat[:, 0:3] / targ_mstat[:, 0:3].sum(axis=0)
shares_mstat.sum(axis=0)  # check

xmat_mstat = np.ndarray((n, 3))
xmat_mstat[:, 1:3] = xmat[:, 1:3]
xmat_mstat[:, 0] = xmat[:, 0] - xmat[:, 1:3].sum(axis=1)  # create other

# we need Q_mstat (n, m)
Q_mstat = np.dot(xmat_mstat, shares_mstat.T)
Q_mstat.shape
Q_mstat.sum(axis=1)

Q = Q_mstat.copy()

# check initial state weights
# iwhs = np.dot(Q.T, np.diag(wh)).T  # try to speed this up
iwhs = np.multiply(Q, wh.reshape((-1, 1)))  # much faster


iwhs.shape
# np.round((iwhs.sum(axis=1) - wh), 2)

# check initial targets
idiff = np.dot(iwhs.T, xmat) - targets
ipdiff = idiff / targets * 100
np.round(ipdiff, 2)
np.round(np.quantile(ipdiff, qtiles), 1)
# np.quantile(np.abs(ipdiff), qtiles)


# %% solve the problem with constrained linear least squares
import scipy  # needed for sparse matrices
from scipy.optimize import lsq_linear


# %% solve the problem with geoweight
# g = mw.Microweight(wh, xmat, geotargets=targets)
g = gw.Geoweight(wh, xmat, targets=targets)
g.geosolve()

g.result
g.result.cost
g.result.cost * 2
g.whs_opt
g.elapsed_minutes

np.quantile(g.whs_opt / iwhs, qtiles)

gwdiff = g.whs_opt.sum(axis=1) - wh
np.round(gwdiff, 2)

gctargs = np.dot(g.whs_opt.T, xmat)
gtdiff = gctargs - targets
np.round(gtdiff, 2)

# percent differences
gtpdiff = gtdiff / targets * 100
np.round(gtpdiff, 4)  # percent difference
np.square(gtpdiff).sum()
np.round(np.quantile(gtpdiff, qtiles), 1)

whs_opt.shape
g.whs_opt.shape

np.round(g.whs_opt[:6, :5], 2)
np.round(whs_opt[:6, :5], 2)


# %% drops dict info

# assume drops is a dict like this rather than a matrix
#  key is state index, value is tuple of indexes for bad columns
# drops = {30: (3, ),
#          31: (3, ),
#          45: (5, 6),
#          46: (1, 2, 3, 4, 5, 6)}


# define rows and columns of targets to drop, using lists (NOT tuples)
targets.shape
xmat.shape
np.round(ipdiff, 2)
np.max(np.abs(ipdiff), axis=0)
np.max(np.abs(ipdiff), axis=1)
ipdiff[24, ]
tpdiff[45, ]
np.quantile(np.abs(tpdiff[imask]), qtiles)
targstates[0]
ipdiff

drops= get_drops(targets, ddrops)

ddrops = {0: [1, 2],
         1: [1, 3],
         3: [2, 4]}

ddrops = {3: [5],
         7: [6],
         8: [5]}

ddrops = {50: [3, 6]}
ddrops = {50: (6, 6)}
ddrops = [(6, (6))]
ddrops = {6: (5, 6)}
ddrops = {6: (6, )}  # MUST use comma in 1-element tuple so it is a tuple!

ddrops = {30: (3, ),
         31: (3, ),
         45: (5, 6),
         46: (1, 2, 3, 4, 5, 6)}

ddrops = {30: (3, ),
         45: (5, 6),
         46: (1, 2, 3, 4, 5, 6)}

ddrops = {50: (1, 2, 3, 4, 5, 6)}

ddrops = {0: (2,),
         50: (1, 2, 3, 4, 5, 6)}

np.max(np.abs(ipdiff), axis=1)
ipdiff[37, ]

# drops stub 1, raking, 1000 iter
ddrops = {0: (6,),
         2: (5, 6),
         7: (5,),
         11: (5, 6),
         16: (5,),
         20: (3, 5, 6),
         24: (6,),
         29: (5,),
         31: (6,),
         37: (5,),
         39: (5,),
         40: (6,),
         45: (5,),
         48: (5, 6),
         50: (1, 2, 3, 4, 5, 6)}


# drops stub 10, entropy, 50 iter
ddrops10 = {0: (4, 5, 6),
         48: (5, 6),
         49: (4, 5, 6),
         50: (1, 2, 3, 4, 5, 6)}


    # create a mask for targets that defines good columns for each area
    # create a mask-like array with shape of targets:
        # where bad state-col combinations are False
        # bad value indices will be a list of tuples
        # each tuple as an integer as its first value (a state index)
        # and a tuple of integers as 2nd value (indexes of bad columns)

ddrops = ddrops10  # for stub 10
drops = get_drops(targets, ddrops)  # see the function


# %% solve the problem using qmatrix
# stubs 1 and 10 are still a problem; 10 has an error to be fixed;
# would lsq work for this? poisson? TPC weight adjustment? fewer targets?


#   File "C:\programs_python\qrake.py", line 237, in qrake
#     if np.isnan(g).any() or np.isinf(g).any() or g.any() == 0:

#   File "C:\Users\donbo\anaconda3\envs\analysis\lib\site-packages\autograd\tracer.py", line 48, in f_wrapped
#     return f_raw(*args, **kwargs)

# TypeError: ufunc 'isnan' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''

# For stub 1, OA seems to be the problem, especially agi and marital returns
# especially the combination of agi and wages
# Is there a good way to drop it?
# It works ok if we drop married and wages, but maybe better to drop OA if we can

# stub 10 is harder
# almost works if we only have return counts and agi
# OA 10 has this for return counts:
# 9.74939513e+03, 1.53749731e+03, 6.53765786e+03

# what if we dropped OA from HT2 and calc'd each state as share of the remaining
# sum???

targets.shape
xmat.shape

flist = targstates + ['XX']
for k in drops.keys(): print(flist[k])
targvars[3]

# djb here is the new way of masking and not masking
np.abs(ipdiff).max()
np.max(np.abs(ipdiff), axis=0)
np.max(np.abs(ipdiff), axis=1)
ipdiff[24, ]


# stub 1 drops
drops = np.where(np.abs(ipdiff) > 100, True, False)  # True means we drop

# stub 10 drops
drops = np.where(np.abs(ipdiff) > 80, True, False)  # True means we drop
# create complex secondary where conditions for stub 10
# condition = np.logical_not(comp3['variable'].isin(['nret_all', 'nret_mfjss', 'nret_single']))
i_oa = (targets.shape[0] - 1, range(1, targets.shape[1]))  # index of other areas, except first
ipdiff[i_oa]
drops[i_oa] = True
i_single = (range(0, targets.shape[0]), 1)
drops[i_single]
drops[i_single] = True
drops
# For stub 10 we seem to need to drop it to 80, drop most of OA, and use entropy
# or else use ddrops
# stub 10 can be solved without drops using raking-ec default

drops.sum()

# solve for optimal Q method can be raking or raking-ec
res_r = qrake(Q, wh, xmat, targets, method='raking', maxiter=100)
res_rd = qrake(Q, wh, xmat, targets, method='raking', maxiter=100, drops=drops)

res_ec = qrake(Q, wh, xmat, targets, method='raking-ec', maxiter=50)
res_ecd = qrake(Q, wh, xmat, targets, method='raking-ec', maxiter=100, drops=drops)  # GOOD
res_ecd = qrake(Q, wh, xmat, targets, method='raking-ec', maxiter=20, drops=drops, objective=QUADRATIC)
res_ec2 = qrake(Q_opt_r, wh, xmat, targets, method='raking-ec')


res = res_r
res = res_rd
res = res_ecd


list(res)
res['esecs']
res['Q_opt']
res['Q_opt'].sum()
res['targets_opt']
res['pctdiff']
res['iter_opt']

np.square(res['pctdiff']).sum()
np.abs(res['pctdiff']).max()
np.abs(res['pctdiff'][drops]).max()

# check weights
# whs_opt = np.multiply(Q_opt, wh.reshape((-1, 1)))  # faster
whs_opt = res['whs_opt']

wdiff = whs_opt.sum(axis=1) - wh
np.round(wdiff, 2)
np.quantile(wdiff, qtiles)
# np.quantile(iwhs, qtiles)
# np.quantile(whs_opt, qtiles)
np.quantile(whs_opt / iwhs, qtiles)

# look at targets
good_targets = np.logical_not(drops)

# check targets
targets
# ctargs = np.dot(whs_opt.T, xmat)
ctargs = res['targets_opt']

tdiff = ctargs - targets
np.round(tdiff, 2)
tpdiff = tdiff / targets * 100
np.round(tpdiff, 4)  # percent difference
np.square(tpdiff).sum()
# np.quantile(tpdiff, (.5))
np.round(np.quantile(tpdiff, qtiles), 1)
# np.sort(tpdiff.flatten())

imask= get_mask(targets, drops)
np.round(np.quantile(tpdiff[imask], qtiles), 1)

# see how the targets re if we remove problematic states
alt = np.delete(tpdiff, (48, 49), axis=0)
np.round(np.quantile(alt, qtiles), 1)

# which states and targets are worst?
tpdiff.shape
np.max(np.abs(tpdiff), axis=0)
np.max(np.abs(tpdiff), axis=1)
bads = (32, 42, 49)
flist = targstates + ['XX']
flist
[flist[i] for i in bads]


# %% commented-out functions


# MatrixCalib <- function(Q,w,Xs){
# 	ver=1
# 	k=1
# 	while(ver>10^(-5) & k <=500)
# 	{
# 		cat(" n.iter = ", k,"\n")
# 		for(i in 1:m)
# 		{
# 			cat("Domain ",nom[i],": calibration ")
# 			g = calib((Xs*w),Q[,i],TTT[i,],method="raking")
# 			if (is.null(g) | any(is.na(g)) | any(g == 0) | any(is.infinite(g)) ) {g = rep(1,length(Q[,i]));cat("non done","\n")}
# 			else {cat("done","\n")}
# 			Q[,i]=Q[,i]*g
# 		}
# 	ver = sum(abs(rowSums(Q)-1))
# 	if (any(is.infinite(abs(rowSums(Q)-1)))) {ver = 10^(-5);cat("Existence of infinite coefficient(s) : non convergence\n")}
# 	cat("Stop condition :\n ")
# 	print(ver)
# 	Q=Q/rowSums(Q)
# 	k=k+1
# 	if (k > 500) cat("Maximal number of iterations not achieved : non convergence \n")
# 	}
# 	Q
# }

