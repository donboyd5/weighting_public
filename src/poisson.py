# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 05:11:04 2020

@author: donbo
"""

import numpy as np


def get_delta(wh, beta, xmat):
    """Get vector of constants, 1 per household.

    See (Khitatrakun, Mermin, Francis, 2016, p.5)

    Note: beta %*% xmat can get very large!! in which case or exp will be Inf.
    It will get large when a beta element times an xmat element is large,
    so either beta or xmat can be the problem.

    In R the problem will bomb but with numpy it appears to recover
    gracefully.

    According to https://stackoverflow.com/questions/40726490/overflow-error-in-pythons-numpy-exp-function
      For most practical purposes, you can probably approximate
        1 / (1 + <a large number>) to zero. That is to say, just ignore the
      warning and move on. Numpy takes care of the approximation for
      you (when using np.float64).

    This will generate runtime warnings of overflow or divide by zero.
    """
    beta_x = np.exp(np.dot(beta, xmat.T))

    # beta_x[beta_x == 0] = 0.1  # experimental
    # beta_x[np.isnan(beta_x)] = 0.1

    delta = np.log(wh / beta_x.sum(axis=0))  # axis=0 gives colsums
    # print(delta)
    # delta[delta == 0] = 0.1  # experimental
    # delta[np.isnan(delta)] = 0.1
    return delta


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

    # return a matrix or vector, depending on the shape of beta_object
    if beta_object.ndim == 1:
        diffs = diffs.flatten()

    return diffs
