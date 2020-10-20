# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 06:20:10 2020

@author: donbo
"""

import warnings
import numpy as np


def rake(Xs, d, total, q=1, objective=None, max_iter=10):
    # this is a direct translation of the raking code of the calib function
    # in the R sampling package, as of 10/3/2020
    # Xs the matrix of covariates
    # d vector of initial weights
    # total vector of targets
    # q vector or scalar related to heteroskedasticity
    # returns g, which when multiplied by the initial d gives the new weight
    EPS = 1e-15  # machine double precision used in R
    EPS1 = 1e-8  # R calib uses 1e-6
    # max_iter = 10

    # make sure inputs all have the right shape
    d = d.reshape((-1, 1))
    total = total.reshape((-1, 1))

    lam = np.zeros((Xs.shape[1], 1))  # lam is k x 1
    w1 = d * np.exp(np.dot(Xs, lam) * q) # h(n) x 1

    # set initial value for g (djb addition to program)
    g = np.ones(w1.size)

    for i in range(max_iter):
        phi = np.dot(Xs.T, w1) - total  # phi is 1 col matrix
        T1 = (Xs * w1).T # T1 has k(m) rows and h(n) columns
        phiprim = np.dot(T1, Xs) # phiprim is k x k
        lam = lam - np.dot(np.linalg.pinv(phiprim, rcond = 1e-15), phi) # k x 1
        w1 = d * np.exp(np.dot(Xs, lam) * q)  # h(n) x 1; in R this is a vector??
        if np.isnan(w1).any() or np.isinf(w1).any():
            warnings.warn("No convergence bad w1")
            g = None
            break
        tr = np.inner(Xs.T, w1.T) # k x 1
        if np.max(np.abs(tr - total) / total) < EPS1:
            break
        if i == max_iter:
            warnings.warn("No convergence after max iterations")
            g = None
        else:
            g = w1 / d  # djb: what if d has zeros?
        # djb temporary solution: force g to be float
        # TODO: explore where we have numerical problems and
        # fix them
        g = np.array(g, dtype=float)  # djb
        g = g.reshape((-1, ))
        # end of the for loop

    return g

