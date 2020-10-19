# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 06:08:32 2020

@author: donbo
"""
import numpy as np
from numpy.random import seed


class Problem:
    """Problem elements."""

    def __init__(self, h, s, k, xsd=.02, ssd=.02):

        self.h = h
        self.s = s
        self.k = k

        # prepare xmat
        seed(1)
        r = np.random.normal(0, xsd, (h, k))
        # r = np.random.randn(h, k) / 100  # random normal)
        xmean = 100 + 20 * np.arange(0, k)
        self.xmat = xmean * (1 + r)

        r = np.random.normal(0, ssd, (h, s))
        r[r < -.9] = -.9  # so that whs cannot be close to zero
        self.whs = 10 + 10 * (1 + r)
        self.wh = self.whs.sum(axis=1)
        self.ws = self.whs.sum(axis=0)
        self.geotargets = np.dot(self.whs.T, self.xmat)
        self.targets = self.geotargets.sum(axis=0)

    def help():
        print("The Problem class creates random problems of arbitrary size",
              "for purposes of testing geosolve.\n")
        print("It requires 3 integer arguments:",
              "\th:\t\tnumber of households (tax returns, etc.)",
              "\ts:\t\tnumber of states or other geographic areas",
              "\tk:\t\tnumber of characteristics each household has, where",
              "\t\t\t\tcharacteristics might be wages, dividends, etc.",
              sep='\n')
        print("\nIt creates an object with the following attributes:",
              "\twh:\t\t\th-length vector of national weights for households",
              "\txmat:\t\th x k matrix of characteristices (data) for households",
              "\ttargets:\ts x k matrix of targets", sep='\n')
        print("\nThe goal of geosolve is to find state weights that will",
              "hit the targets while ensuring that each household's state",
              "weights sum to its national weight.\n", sep='\n')


class rProblem:
    """
    Problem I solved in R, along with the optimal results obtained there.
    """

    def __init__(self):
        self.wh = [43.45278, 51.24605, 39.08130, 47.52817, 44.98483,
                   43.90340, 37.35561, 35.01735, 45.55096, 47.91773]

        x1 = [0.113703411, 0.609274733, 0.860915384, 0.009495756, 0.666083758,
              0.693591292, 0.282733584, 0.292315840, 0.286223285, 0.186722790]
        x2 = [0.6222994, 0.6233794, 0.6403106, 0.2325505, 0.5142511, 0.5449748,
              0.9234335, 0.8372956, 0.2668208, 0.2322259]
        self.xmat = np.array([x1, x2]).T
        self.targets = np.array(
                        [[55.50609, 73.20929],
                         [61.16143, 80.59494],
                         [56.79071, 75.41574]])


