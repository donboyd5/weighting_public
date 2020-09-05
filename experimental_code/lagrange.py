# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:47:46 2020

@author: donbo
"""

# http://kitchingroup.cheme.cmu.edu/blog/2018/11/03/Constrained-optimization-with-Lagrange-multipliers-and-autograd/

import numpy as np
from scipy.optimize import minimize


def objective(X):
    x, y, z = X
    return x**2 + y**2 + z**2


def eq(X):
    x, y, z = X
    return 2 * x - y + z - 3


sol = minimize(objective, [1, -0.5, 0.5],
               constraints={'type': 'eq', 'fun': eq})
sol


import autograd.numpy as np
from autograd import grad

def F(L):
    'Augmented Lagrange function'
    x, y, z, _lambda = L
    return objective([x, y, z]) - _lambda * eq([x, y, z])

# Gradients of the Lagrange function
dfdL = grad(F, 0)

x1 = [1.0, 2.0, 3.0, 4.0]
x1 = [1.0, -0.5, 0.5, 0.0]
dfdL(x1)

# Find L that returns all zeros in this function.
def obj(L):
    x, y, z, _lambda = L
    dFdx, dFdy, dFdz, dFdlam = dfdL(L)
    return [dFdx, dFdy, dFdz, eq([x, y, z])]

from scipy.optimize import fsolve
x, y, z, _lam = fsolve(obj, [0.0, 0.0, 0.0, 1.0])
print(f'The answer is at {x, y, z}')

from autograd import hessian
h = hessian(objective, 0)
h(np.array([x, y, z]))

np.linalg.eig(h(np.array([x, y, z])))[0]


