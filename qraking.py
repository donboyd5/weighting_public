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


# %% utility functions


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
check = np.multiply(dssamp[cols], weights3.reshape(weights3.size, 1))
check.sum(axis=0) # ok, this hits the means

# make up some fake baseline weights that are a bit like the "true" weights
np.random.seed(123)
noise = np.random.normal(0, weights.std() / 10, weights3.shape)
bw = weights3 + noise
bw.min()  # make sure no negative weights

weights3b, l2_norm = ec.maybe_exact_calibrate(
    covariates=dssamp[cols], # 100 rows
    target_covariates=tmeans,   # tmeans
    target_weights=np.array([[.25, .75]]), # make one target more important than another?
    baseline_weights=bw,
    objective=ec.Objective.ENTROPY
)
l2_norm
tmeans
np.dot(dssamp[cols].T, weights3b)
weights3[0:10]
weights3b[0:10]

# are these weights closer to baseline than the actual weights? yes
np.square(weights3 - bw).sum()
np.square(weights3b - bw).sum()
np.round(np.abs(weights3 - bw)[0:9], 4)
np.round(np.abs(weights3b - bw)[0:9], 4)



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


# %% test weighting with empirical calibration
p = mtp.Problem(h=20, s=3, k=2, xsd=.1, ssd=.5)
p = mtp.Problem(h=100, s=8, k=4, xsd=.1, ssd=.5)
p = mtp.Problem(h=1000, s=20, k=8, xsd=.1, ssd=.5)
p = mtp.Problem(h=10000, s=40, k=15, xsd=.1, ssd=.5)
p = mtp.Problem(h=30000, s=40, k=30, xsd=.1, ssd=.5)  # typical stub size?
p = mtp.Problem(h=40000, s=50, k=30, xsd=.1, ssd=.5)
p = mtp.Problem(h=200000, s=50, k=30, xsd=.1, ssd=.5)
p = mtp.Problem(h=300000, s=1, k=100, xsd=.1, ssd=.5)  # our problem size?
p = mtp.Problem(h=int(1e6), s=50, k=100, xsd=.1, ssd=.5)

# p.wh
# p.wh.shape
# p.xmat
# p.xmat.shape
# p.targets

natsums = np.dot(p.xmat.T, p.wh)
np.round(p.targets.sum(axis=0) - natsums, 3)  # check
pop = p.wh.sum()
natmeans = natsums / pop

# "mean-producing" weights
mpw = p.wh / pop
np.dot(p.xmat.T, mpw)
natmeans

# pick a state row (zero based) for targets
st = 0
tsums = p.targets[st, ]
stpop = pop * tsums[0] / natsums[0]  # get the state pop sum from somewhere
tmeans = tsums / stpop

tmeans / natmeans
tsums / natsums

impw = mpw
ispw = mpw * stpop
# check these initial weights
np.dot(p.xmat.T, impw)
tmeans

tsums
np.dot(p.xmat.T, ispw)

# solve for optimal mean-producing weights
a = timer()
ompw, l2_norm = ec.maybe_exact_calibrate(
    covariates=p.xmat, # 1 row per person
    target_covariates=tmeans.reshape((1, -1)),   # tmeans
    baseline_weights=impw,
    # target_weights=np.array([[.25, .75]]), # make one target more important than another?
    autoscale=True,
    objective=ec.Objective.ENTROPY
)
b = timer()
b - a
l2_norm

tmeans
np.dot(p.xmat.T, ompw)

ospw = ompw * stpop  # optimal sum-producing weights
tsums
np.dot(p.xmat.T, ospw)
pdiff = (np.dot(p.xmat.T, ospw) - tsums) / tsums * 100
np.square(pdiff).sum()

wdiffs = ospw- ispw
wpdiffs = wdiffs / ispw * 100
np.square(wdiffs).sum()
np.square(wpdiffs).sum()

qtiles = [0, .1, .25, .5, .75, .9, 1]
np.quantile(ispw, qtiles)
np.quantile(ospw, qtiles)
np.round(np.quantile(wpdiffs, qtiles), 2)


# %% target a single state in an AGI stub
# presumes we have certain items
xmat
wh
targets
# get rid of 3rd target (number 2), single returns
# targets_bak = targets
# xmat_bak = xmat
targets.shape
targets = np.delete(targets, obj=2, axis=1)
targets.shape

xmat.shape
xmat = np.delete(xmat, obj=2, axis=1)
xmat.shape

targets = targets_bak
xmat = xmat_bak

natsums = np.dot(xmat.T, wh)
np.round(targets.sum(axis=0) - natsums, 3)  # check
pop = wh.sum()
natmeans = natsums / pop

# "mean-producing" weights
mpw = wh / pop
np.dot(xmat.T, mpw)
natmeans

# pick a state row (zero based) for targets
st = 0
tsums = targets[st, ]
stpop = pop * tsums[0] / natsums[0]  # get the state pop sum from somewhere
tmeans = tsums / stpop

tmeans / natmeans
tsums / natsums

# initial mean-producing and sum-producing weights
impw = mpw
ispw = mpw * stpop
# check these initial weights
np.dot(xmat.T, impw)
tmeans

tsums
np.dot(xmat.T, ispw)

# solve for optimal mean-producing weights
a = timer()
ompw, l2_norm = ec.maybe_exact_calibrate(
    covariates=xmat, # 1 row per person
    target_covariates=tmeans.reshape((1, -1)),   # tmeans
    baseline_weights=impw,
    # make some targets more important than others
    # target_weights=np.array([[1, 1, 1, 1, 1, 1]]),
    # autoscale=True,  # does not seem to work well
    objective=ec.Objective.ENTROPY
)
b = timer()
b - a
l2_norm
ompw.sum()  # should sum to 1

qtiles = [0, .1, .25, .5, .75, .9, 1]

tmeans
np.dot(xmat.T, ompw)

# compare targets
ospw = ompw * stpop  # optimal sum-producing weights
tsums
np.dot(xmat.T, ospw)
pdiffs = (np.dot(xmat.T, ospw) - tsums) / tsums * 100
np.round(pdiffs, 2)
np.square(pdiffs).sum()
np.round(np.quantile(pdiffs, qtiles), 2)

# compare weights
wdiffs = ospw - ispw
wpdiffs = wdiffs / ispw * 100
np.square(wdiffs).sum()
np.square(wpdiffs).sum()
# 9693566.633685842 with baseline weights
# 1767558944166.2852 without baseline weights

np.quantile(ispw, qtiles)
np.quantile(ospw, qtiles)
np.round(np.quantile(wpdiffs, qtiles), 2)


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
# use an agi stub
Xs = xmat
Xs.shape
w = wh.reshape((-1, 1))
w.shape
TTT = targets
TTT.shape # 21, 7


# create a problem
p = mtp.Problem(h=10, s=3, k=2)
p = mtp.Problem(h=1000, s=10, k=5)
p = mtp.Problem(h=10000, s=20, k=10)
p = mtp.Problem(h=40000, s=50, k=30)

Xs = p.xmat
w = p.wh.reshape(p.wh.size, 1)
TTT = p.targets

# start setting up
m = TTT.shape[0]  # number of states
n = Xs.shape[0]  # number of households

# form the Q matrix
# each state's share of total returns, treating target 0 as # of returns
shares = TTT[:, 0] / TTT[:, 0].sum()
shares.sum()
Q = np.tile(shares, n).reshape((n, m))
np.abs(Q.sum(axis=1) - 1).sum()
# Q[0, :].sum()
# Q.sum(axis=1).size

a = timer()
Qnew = qrake(Q, w, Xs, TTT)
b = timer()
b - a

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

pdiff = (ctargsnew - TTT) / TTT * 100
pdiff
np.round(np.quantile(pdiff, [0, .1, .25, .5, .75, .9, 1]), 4)
np.square(pdiff).sum()

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
    w = w.reshape((-1, 1))  # ensure the proper shape
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

