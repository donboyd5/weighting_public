# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:58:12 2020

@author: donbo
"""

# %% notes
# https://www.w3schools.com/python

# %% imports
# import operator
import numpy as np
# from scipy import linalg, optimize
from scipy.optimize import least_squares
# from scipy.optimize import minimize, rosen, rosen_der


# %% functions
def fun_rosenbrock(x):
    return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])


# %% runopt
x0_rosenbrock = np.array([2, 2])
x0_rosenbrock = np.array([7, 13])

fun_rosenbrock(x0_rosenbrock)

res1 = least_squares(fun_rosenbrock, x0_rosenbrock, method='trf')  # default
dir(res1)
res1.active_mask
res1.cost
res1.fun
res1.grad
res1.jac
res1.message
res1.nfev
# res1.jfev
res1.optimality
res1.status
res1.success
res1.x


# %% data types etc.
# matrices look like they can have column names but not row names

letters = np.array([1, 3, 5, 7, 9, 7, 5])

# 3rd to 5th elements
print(letters[2:5])        # Output: [5, 7, 9]

# 1st to 4th elements
letters[:-5]
print(letters[:-5])       # Output: [1, 3]

# 6th to last elements
print(letters[5:])         # Output:[7, 5]

x = np.array([1, 5, 2])
y = np.array([7, 4, 1])
x + y
# np.array([8, 9, 3])
x * y
x - y
x / y
x % y
operator.mod(x, y)  # modulus: remainder when first operand divided by second
7 // 3
7 % 3
(7 // 3) + (7 % 3)

[1, 1, 0] + [2, 3, 4]  # concatenate 2 lists
np.array([1, 1, 0]) + np.array([2, 3, 4])  # add 2 arrays

a = np.array([[1., 2.], [3., 4.]])
ainv = np.linalg.inv(a)

a
ainv
np.dot(a, ainv)
np.dot(ainv, a)
np.eye(2)
np.allclose(np.dot(a, ainv), np.eye(2))
np.allclose(np.dot(ainv, a), np.eye(2))



thisdict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
thisdict
print(thisdict)
