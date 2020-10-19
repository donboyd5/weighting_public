# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:22:13 2020

@author: donbo
"""

# %% imports
import numpy as np
import src.make_test_problems as mtp
import src.microweight as mw


# %% make problem
# p = mtp.Problem(h=1000, s=10, k=5, xsd=.1, ssd=.5)
p = mtp.Problem(h=1000, s=10, k=5)

prob = mw.Microweight(wh=p.wh, xmat=p.xmat, geotargets=p.targets)


# %% solve problem
ans1 = prob.geoweight(method='qmatrix')
dir(ans1)
ans1.geotargets_opt.shape
dir(prob)
prob.result.iter_opt

ans2 = prob.geoweight(method='qmatrix-ec')
ans2.geotargets_opt.shape

ans3 = prob.geoweight(method='poisson')
ans3.geotargets_opt.shape

np.square((ans1.geotargets_opt - p.targets) / p.targets * 100).sum()
np.square((ans2.geotargets_opt - p.targets) / p.targets * 100).sum()
np.square((ans3.geotargets_opt - p.targets) / p.targets * 100).sum()


# %% check
import src.poisson as ps
dir(ans3)
ans1.whs_opt
ans2.whs_opt
ans3.whs_opt



