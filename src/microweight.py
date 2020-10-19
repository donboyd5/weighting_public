# -*- coding: utf-8 -*-
"""
microweight module

Classes:

    Microweight

Functions:

    abc

@author: donboyd5@gmail.com
"""

# %% imports
# needed for ipopt:
from __future__ import print_function, unicode_literals

from timeit import default_timer as timer

import src.utilities as ut
import src.qmatrix as qm
import src.poisson as ps

import numpy as np
import pandas as pd

import scipy.optimize as spo
# from scipy.optimize import least_squares

import ipopt


# %% Microweight class
class Microweight:
    """Class with data and methods for microdata weighting.

        Common terms and definitions:
        h: number of households (tax records, etc.)
        k: number of characteristics each household has (wages, pensions, etc.)
        s: number of states or geographic areas

        xmat: h x k matrix of characteristics for each household
        wh: 1 x h vector of national weights for households
        whs: h x s matrix of state weights for households (to be solved for)
            for each household, the sum of state weights must equal the
            total household weight

        beta: s x k matrix of poisson model coefficients
            (same for all households)
        delta: 1 x h vector of poisson model constants, 1 per household
            these values are uniquely determined by a given set of beta
            coefficients and the wh values


    """

    def __init__(self, wh, xmat, targets=None, geotargets=None):
        self.wh = wh
        self.xmat = xmat
        self.targets = targets
        self.geotargets = geotargets

    def geoweight(self,
                  method='qmatrix', Q=None, drops=None,
                  maxiter=100):

        # start = timer()
        # methods = ('qmatrix', 'qmatrix-ec', 'poisson')
        # h = self.xmat.shape[0]
        # k = self.xmat.shape[1]
        # s = self.geotargets.shape[0]

        # input checks:
            # geotargets must by s x k

        if method == 'qmatrix':
            result = qm.qmatrix(self.wh, self.xmat, self.geotargets,
                                Q=None,
                                method='raking', drops=drops,
                                maxiter=100)
        elif method == 'qmatrix-ec':
            pass
        elif method == 'poisson':
            pass

        # print(result.targets_opt)
        self.result = result
        return result # self.result
        # print(self.result.iter_opt)
        # end = timer()
        # self.elapsed_minutes = (end - start) / 60
        # self.retrieve_geovalues()

    def retrieve_geovalues(self):
        self.beta_opt = self.result.x.reshape(self.geotargets.shape)
        self.delta_opt = get_delta(self.wh, self.beta_opt, self.xmat)
        self.whs_opt = get_geoweights(self.beta_opt,
                                   self.delta_opt, self.xmat)
        self.geotargets_opt = get_geotargets(self.beta_opt,
                                             self.wh, self.xmat)
        self.targets_diff = self.geotargets_opt - self.geotargets

    def help():
        print("\nThe microweight class requires the following arguments:",
              "\twh:\t\t\th-length vector of national weights for households",
              "\txmat:\t\th x k matrix of characteristices (data) for households",
              "\tgeotargets:\ts x k matrix of targets", sep='\n')
        print("\nThe goal of the method geoweight is to find state weights" +
              " that will",
              "hit the targets while ensuring that each household's state",
              "weights sum to its national weight.\n", sep='\n')


# %% helper functions
def get_diff_weights(geotargets, goal=100):
    """
    difference weights - a weight to be applied to each target in the
      difference function so that it hits its goal
      set the weight to 1 if the target value is zero

    do this in a vectorized way
    """

    # avoid divide by zero or other problems

    # numerator = np.full(geotargets.shape, goal)
    # with np.errstate(divide='ignore'):
    #     dw = numerator / geotargets
    #     dw[geotargets == 0] = 1

    goalmat = np.full(geotargets.shape, goal)
    with np.errstate(divide='ignore'):  # turn off divide-by-zero warning
        diff_weights = np.where(geotargets != 0, goalmat / geotargets, 1)

    return diff_weights


def poisson():
    # betavec0 = np.zeros(self.geotargets.size)
    betavec0 = np.full(self.geotargets.size, 1e-12) # 1e-13 seems best
    dw = get_diff_weights(self.geotargets)
    result = spo.least_squares(
        targets_diff, betavec0,
        method='trf', jac='2-point', verbose=2,
        ftol=1e-10, xtol=1e-10,
        args=(self.wh, self.xmat, self.geotargets, dw))
    return result
