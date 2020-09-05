# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:13:46 2020

@author: donbo
"""

# https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/

import nlopt
from numpy import *

from nlopt.test import test_nlopt

test_nlopt()

# nlopt.version_major()
# nlopt.version_minor()
# nlopt.version_bugfix()

def myfunc(x, grad):
    if grad.size > 0:
        grad[0] = 0.0
        grad[1] = 0.5 / sqrt(x[1])
    return sqrt(x[1])


def myconstraint(x, grad, a, b):
    if grad.size > 0:
        grad[0] = 3 * a * (a*x[0] + b)**2
        grad[1] = -1.0
    return (a*x[0] + b)**3 - x[1]


opt = nlopt.opt('NLOPT_LN_COBYLA', 10)

opt.get_algorithm()
opt.set_lower_bounds(lb)
opt.set_upper_bounds(ub)




opt = nlopt.opt(nlopt.LD_MMA, 2)
opt.set_lower_bounds([-float('inf'), 0])
opt.set_min_objective(myfunc)
opt.add_inequality_constraint(lambda x,grad: myconstraint(x,grad,2,0), 1e-8)
opt.add_inequality_constraint(lambda x,grad: myconstraint(x,grad,-1,1), 1e-8)
opt.set_xtol_rel(1e-4)
x = opt.optimize([1.234, 5.678])
minf = opt.last_optimum_value()
print("optimum at ", x[0], x[1])
print("minimum value = ", minf)
print("result code = ", opt.last_optimize_result())
