# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 07:26:18 2020

@author: donbo
"""

import empirical_calibration as ec

# can we speed up the conjugate gradient iterations

# get the condition number of the sparse matrix

norm_A = scipy.sparse.linalg.norm(A)
norm_invA = scipy.sparse.linalg.norm(scipy.sparse.linalg.inv(A))
cond = norm_A*norm_invA

# the larger the number, the slower the convergence
# can we precondition?
# http://pages.stat.wisc.edu/~wahba/stat860public/pdf1/cj.pdf
