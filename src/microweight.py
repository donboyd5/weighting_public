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

import numpy as np
import pandas as pd
from collections import namedtuple

from timeit import default_timer as timer

import ipopt  # requires special installation

import src.utilities as ut
import src.geoweight_qmatrix as qm
import src.geoweight_poisson as ps
import src.reweight_ipopt as rwi

import scipy.optimize as spo
# from scipy.optimize import least_squares


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

    def reweight(self,
                  method='ipopt', Q=None, drops=None,
                  maxiter=100):
        if method == 'ipopt':
            x, info = rwi.rw_ipopt(self.wh, self.xmat, self.targets)
        elif method == 'empcal':
            pass
        return x, info

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
            result = qm.qmatrix(self.wh, self.xmat, self.geotargets,
                                Q=None,
                                method='raking-ec', drops=drops,
                                maxiter=100)
        elif method == 'poisson':
            result = ps.poisson(self.wh, self.xmat, self.geotargets)

        # print(result.targets_opt)
        # self.result = result
        return result # self.result
        # print(self.result.iter_opt)
        # end = timer()
        # self.elapsed_minutes = (end - start) / 60
        # self.retrieve_geovalues()

    def help():
        print("\nThe microweight class requires the following arguments:",
              "\twh:\t\t\th-length vector of national weights for households",
              "\txmat:\t\th x k matrix of characteristices (data) for households",
              "\tgeotargets:\ts x k matrix of targets", sep='\n')
        print("\nThe goal of the method geoweight is to find state weights" +
              " that will",
              "hit the targets while ensuring that each household's state",
              "weights sum to its national weight.\n", sep='\n')
