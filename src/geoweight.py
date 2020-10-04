# -*- coding: utf-8 -*-
"""
geoweight class.
Created on Sat Aug 29 04:55:45 2020

@author: donbo
"""

import numpy as np
import scipy.optimize as spo
import src.poisson as ps

from timeit import default_timer as timer


class Geoweight:
    """Class with data and methods for geographic weighting.

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

    def __init__(self, wh, xmat, geotargets=None):

        self.wh = wh
        self.xmat = xmat
        self.geotargets = geotargets

    def geoweight(self, method='qrake'):
        start = timer()
        h = self.xmat.shape[0]
        k = self.xmat.shape[1]
        s = self.geotargets.shape[0]

        # input checks:
            # geotargets must by s x k

        betavec0 = np.zeros(self.geotargets.size)
        dw = get_diff_weights(self.geotargets)

        result = spo.least_squares(targets_diff, betavec0,
                                   method='trf', jac='2-point', verbose=2,
                                   args=(self.wh, self.xmat, self.geotargets, dw))
        self.result = result
        end = timer()
        self.elapsed_minutes = (end - start) / 60
        self.retrieve_values()

    def retrieve_values(self):
        self.beta_opt = self.result.x.reshape(self.geotargets.shape)
        self.delta_opt = ps.get_delta(self.wh, self.beta_opt, self.xmat)
        self.whs_opt = ps.get_weights(self.beta_opt,
                                   self.delta_opt, self.xmat)
        self.geotargets_opt = ps.get_targets(self.beta_opt,
                                       self.wh, self.xmat)
        self.targets_diff = self.geotargets_opt - self.geotargets

    def help():
        print("\nThe Geoweight class requires the following arguments:",
              "\twh:\t\t\th-length vector of national weights for households",
              "\txmat:\t\th x k matrix of characteristices (data) for households",
              "\tgeotargets:\ts x k matrix of targets", sep='\n')
        print("\nThe goal of the method geoweight is to find state weights" +
              " that will",
              "hit the targets while ensuring that each household's state",
              "weights sum to its national weight.\n", sep='\n')


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
