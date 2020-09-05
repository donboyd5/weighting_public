# -*- coding: utf-8 -*-
"""
microweight module

Classes:

    Microweight

Functions:

    abc

@author: donboyd5@gmail.com
"""
from __future__ import print_function, unicode_literals

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from timeit import default_timer as timer
import ipopt


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

    def __init__(self, wh, xmat, geotargets=None):

        self.wh = wh
        self.xmat = xmat
        self.geotargets = geotargets

    def geoweight(self):
        start = timer()
        h = self.xmat.shape[0]
        k = self.xmat.shape[1]
        s = self.geotargets.shape[0]

        # input checks:
            # geotargets must by s x k

        betavec0 = np.zeros(self.geotargets.size)
        dw = get_diff_weights(self.geotargets)

        result = least_squares(targets_diff, betavec0,
                     method='trf', jac='2-point', verbose=2,
                     args=(self.wh, self.xmat, self.geotargets, dw))
        self.result = result
        end = timer()
        self.elapsed_minutes = (end - start) / 60
        self.retrieve_geovalues()

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


def get_delta(wh, beta, xmat):
    """
    Get vector of constants, 1 per household.

    See (Khitatrakun, Mermin, Francis, 2016, p.5)

    Note: we cannot let beta %*% xmat get too large!! or exp will be Inf and
    problem will bomb. It will get large when a beta element times an
    xmat element is large, so either beta or xmat can be the problem.
    """
    beta_x = np.exp(np.dot(beta, xmat.T))

    delta = np.log(wh / beta_x.sum(axis=0))  # axis=0 gives colsums
    return delta


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


def get_geotargets(beta, wh, xmat):
    """
    Calculate matrix of target values by state and characteristic.

    Returns
    -------
    targets_mat : matrix
        s x k matrix of target values.

    """
    delta = get_delta(wh, beta, xmat)
    whs = get_geoweights(beta, delta, xmat)
    targets_mat = np.dot(whs.T, xmat)
    return targets_mat


def get_geoweights(beta, delta, xmat):
    """
    Calculate state-specific weights for each household.

    Definitions:
    h: number of households
    k: number of characteristics each household has
    s: number of states or geographic areas

    See (Khitatrakun, Mermin, Francis, 2016, p.4)

    Parameters
    ----------
    beta : matrix
        s x k matrix of coefficients for the poisson function that generates
        state weights.
    delta : vector
        h-length vector of constants (one per household) for the poisson
        function that generates state weights.
    xmat : matrix
        h x k matrix of characteristics (data) for households.

    Returns
    -------
    matrix of dimension h x s.

    """
    # begin by calculating beta_x, an s x h matrix:
    #   each row has the sum over k of beta[s_i, k] * x[h_j, k]
    #     for each household where s_i is the state in row i
    #   each column is a specific household
    beta_x = np.dot(beta, xmat.T)

    # add the delta vector of household constants to every row
    # of beta_x and transpose
    # beta_xd <- apply(beta_x, 1, function(mat) mat + delta)
    beta_xd = (beta_x + delta).T

    weights = np.exp(beta_xd)

    return weights


def targets_diff(beta_object, wh, xmat, geotargets, diff_weights):
    '''
    Calculate difference between calculated targets and desired targets.

    Parameters
    ----------
    beta_obj: vector or matrix
        if vector it will have length s x k and we will create s x k matrix
        if matrix it will be dimension s x k
        s x k matrix of coefficients for the poisson function that generates
        state weights.
    wh: array-like
        DESCRIPTION.
    xmat: TYPE
        DESCRIPTION.
    geotargets: TYPE
        DESCRIPTION.
    diff_weights: TYPE
        DESCRIPTION.

    Returns
    -------
    matrix of dimension s x k.

    '''
    # beta must be a matrix so if beta_object is a vector, reshape it
    if beta_object.ndim == 1:
        beta = beta_object.reshape(geotargets.shape)
    elif beta_object.ndim == 2:
        beta = beta_object

    geotargets_calc = get_geotargets(beta, wh, xmat)
    diffs = geotargets_calc - geotargets
    diffs = diffs * diff_weights

    # retirm a matrix or vector, depending on the shape of beta_object
    if beta_object.ndim == 1:
        diffs = diffs.flatten()

    return diffs
