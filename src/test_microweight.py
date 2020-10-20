# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:22:13 2020

@author: donbo
"""

# %% imports
import numpy as np
import src.make_test_problems as mtp
import src.microweight as mw


# %% constants
qtiles = (0, .01, .1, .25, .5, .75, .9, .99, 1)


# %% functions
def targs(x, wh, xmat):
    return np.dot(xmat.T, wh * x)


def sspd(x, wh, xmat, targets):
    #  sum of squared percentage differences
    diffs = np.dot(xmat.T, wh * x) - targets
    pdiffs = diffs / targets * 100
    return np.square(pdiffs).sum()


# %% make problem
# p = mtp.Problem(h=1000, s=10, k=5, xsd=.1, ssd=.5)
p = mtp.Problem(h=40, s=2, k=3)
p = mtp.Problem(h=1000, s=10, k=5)
p = mtp.Problem(h=10000, s=10, k=10)
p = mtp.Problem(h=40000, s=10, k=30)

np.random.seed(1)
noise = np.random.normal(0, .0125, p.k)
noise
ntargets = p.targets * (1 + noise)

# ntargets = p.targets

prob = mw.Microweight(wh=p.wh, xmat=p.xmat, targets=ntargets, geotargets=p.geotargets)


# %% reweight the problem
x1, info1 = prob.reweight(method='ipopt', crange=0.001, quiet=False)
x1, info1 = prob.reweight(method='ipopt', crange=0.001, xlb=0, xub=1e5, quiet=False)
# type(info1)
# dir(info1)
list(info1)
info1['status']
info1['status_msg']
info1['obj_val']
info1['x']

x2, info2 = prob.reweight(method='empcal', increment=.00001)
info2
x2

x3, info3 = prob.reweight(method='rake', max_iter=20)
info3
x3

ntargets
targs(x1, p.wh, p.xmat)
targs(x2, p.wh, p.xmat)
targs(x3, p.wh, p.xmat)


# sum of squared percentage differences
sspd(x1, p.wh, p.xmat, ntargets)
sspd(x2, p.wh, p.xmat, ntargets)
sspd(x3, p.wh, p.xmat, ntargets)

# distribution of x values
np.quantile(x1, qtiles)
np.quantile(x2, qtiles)
np.quantile(x3, qtiles)

x1.sum()
x2.sum()
x3.sum()



# %% geoweight the problem
gw1 = prob.geoweight(method='qmatrix')
# dir(gw1)
gw1.geotargets_opt

gw2 = prob.geoweight(method='qmatrix-ec')
gw2.geotargets_opt

gw3 = prob.geoweight(method='poisson')
gw3.geotargets_opt

# sum of squared percentage differences
np.square((gw1.geotargets_opt - p.geotargets) / p.geotargets * 100).sum()
np.square((gw2.geotargets_opt - p.geotargets) / p.geotargets * 100).sum()
np.square((gw3.geotargets_opt - p.geotargets) / p.geotargets * 100).sum()


# %% check
import src.poisson as ps
dir(gw3)
gw1.whs_opt
gw2.whs_opt
gw3.whs_opt



