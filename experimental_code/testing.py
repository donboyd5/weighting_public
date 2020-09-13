# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 05:16:40 2020

@author: donbo
"""

# %% imports
import numpy as np
import src.geoweight as gw
import src.make_test_problems as mtp


# %% test problems of arbitrary size
mtp.Problem.help()

p = mtp.Problem(h=100, s=5, k=2)
# p = mtp.Problem(h=20000, s=30, k=10)  # moderate-sized problem, < 1 min

# I don't think our problems for a single AGI range will get bigger
# than the one below:
#   30k tax records, 50 states, 30 characteristics (targets) per state
# but problems will be harder to solve with real data
# p = mtp.Problem(h=30000, s=50, k=30) # took 31 mins on my computer

gw.Geoweight.help()

g1 = gw.Geoweight(p.wh, p.xmat, p.targets)

# look at the inputs
g1.wh
g1.xmat
g1.geotargets

# solve for state weights
g1.geoweight()

# examine results
g1.elapsed_minutes
dir(g1)
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
# with default diff weights g1.result.fun also will show percent differences
g1.result.fun
np.square(pdiff).sum()  # would like it to be approx zero

# do state weights sum to national weights for every household?
g1.whs_opt  # optimal state weights
wh_opt = g1.whs_opt.sum(axis=1)
wh_opt  # sum of optimal state weights for each household
np.square(wh_opt - g1.wh).sum()  # should be approx zero

