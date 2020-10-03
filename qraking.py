# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 07:26:49 2020

https://github.com/google/empirical_calibration/blob/master/notebooks/survey_calibration_cvxr.ipynb

@author: donbo
"""
# %% imports
import os
import sys
import warnings
import requests
import pandas as pd
import numpy as np
import src.microweight as mw
import src.make_test_problems as mtp

from timeit import default_timer as timer

# pip install -q git+https://github.com/google/empirical_calibration
import empirical_calibration as ec

# pip install -q rdata
import rdata

# %% functions
# rake(Xsw, Q[:, i], TTT[i, :]) # TTT[i, :]

# Xs <- matrix(2:19, nrow=6)
#      [,1] [,2] [,3]
# [1,]    2    8   14
# [2,]    3    9   15
# [3,]    4   10   16
# [4,]    5   11   17
# [5,]    6   12   18
# [6,]    7   13   19
# lam <- matrix(c(1, 2, 3), ncol=1)
# > Xs %*% lam
#      [,1]
# [1,]   60
# [2,]   66
# [3,]   72
# [4,]   78
# [5,]   84
# [6,]   90
# x = np.array(np.arange(2, 20).reshape((3, 6))).T
# x
# lam = np.array([[1],
#                 [2],
#                 [3]])
# np.dot(x, lam)

djb1=Xsw; djb2=Q[:, i]; djb3=TTT[i, :]
Xs=djb1; d=djb2; total=djb3
rake(djb1, djb2, djb3)

def rake(Xs, d, total, q=1):
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

    # DJB NEW check whether we need to set initial value for g
    g = np.ones(w1.size)
    # phi = np.dot(Xs.T, w1) - total  # phi is 1 col matrix
    # T1 = (Xs * w1).T # T1 has k(m) rows and h(n) columns
    # phiprim = np.dot(T1, Xs) # phiprim is k x k
    # lam1 = np.dot(np.linalg.pinv(phiprim, rcond = 1e-15), phi) # k x 1
    # if np.abs(lam1).max() < EPS:
    #     g = 1 + np.exp(np.dot(Xs, lam) * q)
    # END DJB NEW

    # operands could not be broadcast together with shapes (20,1) (100,1)
    for i in range(max_iter):
        phi = np.dot(Xs.T, w1) - total  # phi is 1 col matrix
        T1 = (Xs * w1).T # T1 has k(m) rows and h(n) columns
        phiprim = np.dot(T1, Xs) # phiprim is k x k
        lam = lam - np.dot(np.linalg.pinv(phiprim, rcond = 1e-15), phi) # k x 1
        w1 = d * np.exp(np.dot(Xs, lam) * q)  # h(n) x 1; in R this is a vector??
        if np.isnan(w1).any() or np.isinf(w1).any():
            warnings.warn("No convergence")
            g = None
            break
        tr = np.inner(Xs.T, w1.T) # k x 1
        if np.max(np.abs(tr - total) / total) < EPS1:
            break
        if i==max_iter:
            warnings.warn("No convergence")
            g = None
        else:
            g = w1 / d
    print(i)
    return g


# %% package example

# !wget -q https://github.com/anqif/CVXR/raw/master/data/dspop.rda
# !wget -q https://github.com/anqif/CVXR/raw/master/data/dssamp.rda
dspop = rdata.conversion.convert(rdata.parser.parse_file('dspop.rda'))['dspop']
dssamp = rdata.conversion.convert(rdata.parser.parse_file('dssamp.rda'))['dssamp']

type(dspop) # pandas
dssamp

cols = ['sex', 'age']
weights, l2_norm = ec.maybe_exact_calibrate(
    covariates=dssamp[cols], # 100 rows
    target_covariates=dspop[cols],  # 1000 rows
    objective=ec.Objective.ENTROPY
)
l2_norm
# weights is an array, length 100, sum is 1
weights.sum()
check = np.multiply(dssamp[cols], weights.reshape(weights.size, 1))
check.sum(axis=0) # ok, this hits the means
dspop[cols].mean()

# so this gets weights that ensure that weighted sample means = pop means

# therefore, for sums, we have:
dspop[cols].sum()
tmeans * np.size(dspop, 0)
dspop[cols].sum() / np.size(dspop, 0) # this is what we should use as target

# this works ----
# get population means to use as targets
tmeans = np.array(dspop[cols].mean()).reshape(1, 2)
tsums = np.array(dspop[cols].sum()).reshape(1, 2)

weights2, l2_norm = ec.maybe_exact_calibrate(
    covariates=dssamp[cols], # 100 rows
    target_covariates=tmeans,   # tmeans
    objective=ec.Objective.ENTROPY
)
l2_norm
tmeans
# data times weights
check = np.multiply(dssamp[cols], weights2.reshape(weights.size, 1))
check.sum(axis=0) # ok, this hits the means

# data times weights * pop
tsums
check2 = np.multiply(dssamp[cols], weights2.reshape(weights.size, 1) * 1000)
check2.sum(axis=0) # ok, this hits the means

# compare pop mean, unweighted sample mean, and weighted sample mean
dspop.mean()
dssamp.mean()
check = np.multiply(dssamp, weights2.reshape(weights.size, 1))
check.sum(axis=0) # ok, this hits the means


# can we adjust using initial weights?

weights3, l2_norm = ec.maybe_exact_calibrate(
    covariates=dssamp[cols], # 100 rows
    target_covariates=tmeans,   # tmeans
    target_weights=np.array([[.25, .75]]), # make one target more important than another?
    objective=ec.Objective.ENTROPY
)
l2_norm
tmeans
# data times weights
check = np.multiply(dssamp[cols], weights2.reshape(weights.size, 1))
check.sum(axis=0) # ok, this hits the means


covs = pd.DataFrame(dssamp[cols].sum(axis=0))
pd.DataFrame(dssamp[cols].sum(), columns=cols)
type(covs)
dssamp[cols].sum()
weights2, l2_norm = ec.maybe_exact_calibrate(
    covariates=covs,  # length 100
    target_covariates=dspop[cols],  # length 1000
    objective=ec.Objective.ENTROPY
)
baseline_weights
# weights is an array, sum is 1
weights.size # 100, the same size as the sample - so we get a weight for each
weights.sum()


# %% test weighting
p = mtp.Problem(h=20, s=3, k=2)
p = mtp.Problem(h=100, s=8, k=4)
p = mtp.Problem(h=1000, s=20, k=8)
p = mtp.Problem(h=10000, s=40, k=15)
p = mtp.Problem(h=40000, s=50, k=30)
p = mtp.Problem(h=200000, s=50, k=30)
p = mtp.Problem(h=int(1e6), s=50, k=100)

p.wh
p.xmat
p.xmat.shape
p.targets

# pick a state row (zero based) for targets
st = 7

natsums = p.targets.sum(axis=0)  # national sum for each variable
shares = np.multiply(p.targets, 1 / natsums)
shares.sum(axis=0)

npop = p.wh.sum()
nsamp = npop * shares[st, 0]


tsums = p.targets[st, ]  # .shape # (2, )
# nsamp = tsums[0]  # treat the first variable as if it is a count
tmeans = tsums / nsamp
tmeans

# tcovs = tmeans.reshape((1, tmeans.size))

# covs = np.multiply(p.xmat, p.wh.reshape(p.wh.size, 1))
# covs2 = covs * shares[st, 0]

# bw = p.wh.reshape((p.wh.size, 1)) / 3

# p.xmat.shape  # 20, 2
# covs = p.xmat
covs = np.multiply(p.xmat, p.wh.reshape(p.wh.size, 1))
covs.sum(axis=0)
covs.mean(axis=0)
covs2 = covs * tmeans[0] / covs.mean(axis=0)[0]
covs2.mean(axis=0)
tmeans

# covs.shape
# covs2 = covs * shares[1, 0] # np.multiply(covs, )
# covs2.shape
# tcovs.shape  # 1, 2
# bw.shape

a = timer()
weights4, l2_norm = ec.maybe_exact_calibrate(
    covariates=covs2, # 1 row per person
    target_covariates=tmeans.reshape((1, tmeans.size)),   # tmeans
    # baseline_weights=bw,
    # target_weights=np.array([[.25, .75]]), # make one target more important than another?
    autoscale=True,
    objective=ec.Objective.ENTROPY
)
b = timer()
b - a
l2_norm
# tmeans

# covs2.sum(axis=0)

# data times weights
check = np.multiply(covs2, weights4.reshape(weights4.size, 1))
calcmeans = check.sum(axis=0) # ok, this hits the means

# compare means
tmeans  # targets
initmeans = covs2.mean(axis=0)  # initial means
calcmeans  # after weighting
np.square(initmeans - tmeans).sum()
np.square(calcmeans - tmeans).sum()


# %% good agistub
# assume we already have targets, wh, and xmat

st = 0

natsums = targets.sum(axis=0)  # national sum for each variable

wh = wh.reshape(wh.size, 1)
wh_avg = wh.mean()
npop = wh.sum()
nsamp = targets[st, 0]  # treat first variable as a count for which we want shares

# tmeans, used for target covariates, must be 2d array, so make tsums 2d array
# tsums are the actual total targets, tmeans are the mean per return
tsums = targets[st, ].reshape((1, tsums.size))  # must be 2d array (1, ntargs)
tmeans = tsums / nsamp  # .reshape((1, tsums.size))  # must be 2d array (1, ntargs)

covs = xmat * (wh / wh_avg)
initmeans = covs.mean(axis=0)  # initial weighted means
initsums = initmeans * nsamp

# uniform baseline weights
bw = wh * nsamp / npop
bw = bw / bw.sum()
bw

# bw = np.ones(wh.size) * 1 / xmat.shape[0] # this works
# wgood = weights.reshape(weights.size, 1)
bw
wh.shape

tmp = xmat* wh / wh_avg * wgood
tmp.sum(axis=0)

a = timer()
weights, l2_norm = ec.maybe_exact_calibrate(
    covariates=xmat* wh / wh_avg, # covs, # 1 row per person
    target_covariates=tmeans,  # tcovs,
    # baseline_weights=wgood,
    # target_weights=np.array([[1, 2, 3, 4, 3, 2, 1]]), # priority weights
    autoscale=True,
    objective=ec.Objective.ENTROPY
)
b = timer()
b - a
l2_norm
np.quantile(weights, [0, .1, .25, .5, .75, .9, 1])
weights.sum()
weights[1:10]

iweights =(wh / wh_avg) / nsamp
weights, l2_norm = ec.maybe_exact_calibrate(
    covariates=xmat, # covs, # 1 row per person
    target_covariates=tmeans,  # tcovs,
    baseline_weights=iweights,
    # target_weights=np.array([[1, 2, 3, 4, 3, 2, 1]]), # priority weights
    autoscale=True,
    objective=ec.Objective.ENTROPY
)

# array([7.42767781e-06, 1.78542362e-05, 2.13985906e-05, 2.52127750e-05,
#        2.80850144e-05, 3.00345197e-05, 1.17109605e-04])

# data times weights
check = np.multiply(covs, weights.reshape(weights.size, 1))
calcmeans = check.sum(axis=0)  # ok, this hits the means
calcsums = calcmeans * nsamp

wh.shape
wh_avg.shape
iweights =(wh / wh_avg)
iweights.shape
weights.shape
newwtts = wh / wh_avg * weights.reshape((weights.size, 1))
iweights
newwtts
np.dot(xmat.T, newwtts) * nsamp
np.dot(xmat.T, iweights) * nsamp
tsums

# compare means
tmeans  # targets
initmeans  # initial means
calcmeans  # after weighting
np.square(initmeans - tmeans).sum()
np.square(calcmeans - tmeans).sum()

# compare sums
tsums
initsums
calcsums
np.square(initsums - tsums).sum()
np.square(calcsums - tsums).sum()

# pct differences in sums
np.square((initsums - tsums) / tsums * 100).sum()
np.square((calcsums - tsums) / tsums * 100).sum()


# %% R calib raking function

# get a problem from agi sutb
Xs = xmat
targets
# use state 0
total = targets[0, ].reshape(targets[0,].size, 1)  # targets for a single state
d = start_values.query('STATE =="AK"')['iwhs'].to_numpy()
d.shape
d = d.reshape((d.size, 1))

# create a problem
p = mtp.Problem(h=100, s=5, k=4)
Xs = p.xmat
total = p.targets[0, ].reshape(p.targets.shape[1], 1) # 1-column matrix (array)
ratio = p.targets[0, 0] / p.targets[:, 0].sum()
d = (p.wh * ratio * 1.0).reshape((p.wh.size, 1))

Xs.shape
p.xmat.shape
total.shape
d.shape

# grake
a = timer()
g = rake(Xs, d, total)
b = timer()
b - a

g * p.wh
g * d

itot = np.dot(Xs.T, d)
ctot = np.dot(Xs.T, g * d)

# compare targets
total
itot
ctot
idiff = itot - total
cdiff = ctot - total
ipdiff = idiff / total * 100
cpdiff = cdiff / total * 100
np.round(idiff / total * 100, 4)
np.round(cdiff / total * 100, 4)
np.square(ipdiff).sum()
np.square(cpdiff).sum()

# compare weights
wnew = g * d
wdiff = wnew - d
wpdiff = wdiff / d * 100
np.quantile(wdiff, [0, .25, .5, .75, 1])
np.quantile(wpdiff, [0, .25, .5, .75, 1])
np.square(wpdiff).sum()





    # else if (method == "raking") {
    #     lambda = as.matrix(rep(0, ncol(Xs)))
    #     w1 = as.vector(d * exp(Xs %*% lambda * q))
    #     for (l in 1:max_iter) {
    #         phi = t(Xs) %*% w1 - total
    #         T1 = t(Xs * w1)
    #         phiprim = T1 %*% Xs
    #         lambda = lambda - ginv(phiprim, tol = EPS) %*% phi
    #         w1 = as.vector(d * exp(Xs %*% lambda * q))
    #         if (any(is.na(w1)) | any(is.infinite(w1))) {
    #             warning("No convergence")
    #             g = NULL
    #             break
    #         }
    #         tr = crossprod(Xs, w1)
    #         if (max(abs(tr - total)/total) < EPS1)
    #             break
    #     }
    #     if (l == max_iter) {
    #         warning("No convergence")
    #         g = NULL
    #     }
    #     else g = w1/d
    # }


# %% qraking approach using the rake function
p = mtp.Problem(h=10, s=3, k=2)
p = mtp.Problem(h=100, s=7, k=3)
Xs = p.xmat
w = p.wh.reshape(p.wh.size, 1)
TTT = p.targets
m = TTT.shape[0]  # number of states
n = Xs.shape[0]  # number of households

# TTT = targets1

# total = p.targets[0, ].reshape(p.targets.shape[1], 1) # 1-column matrix (array)
# ratio = p.targets[0, 0] / p.targets[:, 0].sum()
# d = (p.wh * ratio * 1.0).reshape((p.wh.size, 1))

# form the Q matrix
# each state's share of total returns, treating target 0 as # of returns
shares = TTT[:, 0] / TTT[:, 0].sum()
shares.sum()
Q = np.tile(shares, n).reshape((n, m))

# Q[0, :].sum()
# Q.sum(axis=1).size

Qnew = qrake(Q, w, Xs, TTT)
np.abs(Q.sum(axis=1) - 1).sum()
np.abs(Qnew.sum(axis=1) - 1).sum()

# np.abs(Q.sum(axis=0) - 1).sum()


whs = w * Q
whs.sum(axis=1)
w
ctargs = np.dot(whs.T, Xs)
np.round(ctargs - TTT, 2)
np.square(ctargs - TTT).sum()

whsnew = w * Qnew
whsnew.sum(axis=1)
w
ctargsnew = np.dot(whsnew.T, Xs)
np.round(ctargsnew - TTT, 2)
np.square(ctargsnew - TTT).sum()

Q.shape
w.shape
Xs.shape
TTT.shape
whs.shape
# djb1=Xsw; djb2=Q[:, i]; djb3=TTT[i, :]

def qrake(Q, w, Xs, TTT):
    EPS = 1e-6  # acceptable error (tolerance)
    MAX_ITER = 500
    ver = 1  # initialize error in sum of weight shares across states
    k = 1  # initialize iteration count
    m = TTT.shape[0]  # number of states
    # compute before the loop to save a little time (calib calcs in the loop)
    Xsw = Xs * w  # Xsw.shape n x number of targets
    # i = 0
    Q = Q.copy()

    while (ver > EPS) & (k <= MAX_ITER):
        print("Iteration: ", k)
        for i in range(m):
            print("Area: {} of {}:{}, result: ".format(i, 0, m - 1), end='')
            g = rake(Xsw, Q[:, i], TTT[i, :])
            if np.isnan(g).any() or np.isinf(g).any() or g.any()==0:
                # g = rep(1,length(Q[,i]))
                g = np.ones(g.size)
                print("non done") # we'll need to do this one again
            else:
                # print("done")
                Q[:, i] = Q[:, i] * g.reshape(g.size, )
                print(np.abs(Q.sum(axis=1) - 1).sum())

        # diff to be compared to epsilon
        absdiff = np.abs(Q.sum(axis=1) - 1)
        ver = absdiff.sum()
        if np.isinf(absdiff).any():
            ver = 1e-5
            print("Existence of infinite coefficients --> non-convergence.")

        print("Stop condition: {}".format(ver))
        Q = Q / Q.sum(axis=1)[:,None]  # so that we have proper broadcasting
        print(np.abs(Q.sum(axis=1) - 1).sum())
        p
        k = k + 1
        if k > MAX_ITER:
            print("Maximal number of iterations: non convergence .")

    return Q


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

######################################################

# here is the r sampling code for calib with method = raking
# where Xs is the matrix of covariates (matrix of calibration variables)
# d is the vector of initial weights
# q is vector of positive values accounting for heteroscedasticity; the variation of the g-weights is reduced for small values of q
# EPS = .Machine$double.eps
# EPS1 = 1e-06
# total is vector of population totals



    else if (method == "raking") {
        lambda = as.matrix(rep(0, ncol(Xs)))
        w1 = as.vector(d * exp(Xs %*% lambda * q))
        for (l in 1:max_iter) {
            phi = t(Xs) %*% w1 - total
            T1 = t(Xs * w1)
            phiprim = T1 %*% Xs
            lambda = lambda - ginv(phiprim, tol = EPS) %*% phi
            w1 = as.vector(d * exp(Xs %*% lambda * q))
            if (any(is.na(w1)) | any(is.infinite(w1))) {
                warning("No convergence")
                g = NULL
                break
            }
            tr = crossprod(Xs, w1)
            if (max(abs(tr - total)/total) < EPS1)
                break
        }
        if (l == max_iter) {
            warning("No convergence")
            g = NULL
        }
        else g = w1/d
    }

