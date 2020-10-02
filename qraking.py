# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 07:26:49 2020

https://github.com/google/empirical_calibration/blob/master/notebooks/survey_calibration_cvxr.ipynb

@author: donbo
"""
# %% imports
import os
import sys
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

p.wh
p.xmat
p.xmat.shape
p.targets

natsums = p.targets.sum(axis=0)  # national sum for each variable
shares = np.multiply(p.targets, 1 / natsums)
shares.sum(axis=0)
shares[1, 0]

npop = p.wh.sum()
nsamp = npop / shares[1, 0]

# use the state 1 (2nd row) as targets
st = 1
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

covs.shape
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
calcmeans
tmeans
np.square(calcmeans - tmeans).sum()


# %% try it out on a column of an agi stub

# assume we already have targets, wh, and xmat



natsums = targets.sum(axis=0)  # national sum for each variable
shares = np.multiply(targets, 1 / natsums)
shares.sum(axis=0)
shares[1, 0]

npop = wh.sum()
nsamp = npop / shares[1, 0]

# use the state 1 (2nd row) as targets
st = 1
tsums = targets[st, ]  # .shape # (2, )
# nsamp = tsums[0]  # treat the first variable as if it is a count
tmeans = tsums / nsamp
tmeans

# tcovs = tmeans.reshape((1, tmeans.size))

# covs = np.multiply(p.xmat, p.wh.reshape(p.wh.size, 1))
# covs2 = covs * shares[st, 0]

# bw = p.wh.reshape((p.wh.size, 1)) / 3

# p.xmat.shape  # 20, 2
# covs = p.xmat
covs = np.multiply(xmat, wh.reshape(wh.size, 1))
covs.sum(axis=0)
covs.mean(axis=0)
covs2 = covs * tmeans[0] / covs.mean(axis=0)[0]
covs2.mean(axis=0)
tmeans

# covs2 = covs * shares[1, 0] # np.multiply(covs, )
# covs2.shape
# tcovs.shape  # 1, 2
# bw.shape

a = timer()
weights, l2_norm = ec.maybe_exact_calibrate(
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
check = np.multiply(covs2, weights.reshape(weights.size, 1))
calcmeans = check.sum(axis=0) # ok, this hits the means
calcmeans
tmeans
np.square(calcmeans - tmeans).sum()


