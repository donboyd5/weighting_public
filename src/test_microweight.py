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

prob = mw.Microweight(wh=p.wh, xmat=p.xmat, targets=p.targets, geotargets=p.geotargets)


# %% reweight the problem
x, info = prob.reweight(method='ipopt')




# %% geoweight the problem
gw1 = prob.geoweight(method='qmatrix')
# dir(gw1)
gw1.geotargets_opt

gw2 = prob.geoweight(method='qmatrix-ec')
gw2.geotargets_opt

gw3 = prob.geoweight(method='poisson')
gw3.geotargets_opt

np.square((gw1.geotargets_opt - p.geotargets) / p.geotargets * 100).sum()
np.square((gw2.geotargets_opt - p.geotargets) / p.geotargets * 100).sum()
np.square((gw3.geotargets_opt - p.geotargets) / p.geotargets * 100).sum()


# %% check
import src.poisson as ps
dir(gw3)
gw1.whs_opt
gw2.whs_opt
gw3.whs_opt



