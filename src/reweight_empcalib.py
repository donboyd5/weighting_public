# -*- coding: utf-8 -*-
"""
Empirical calibration

@author: donbo
"""


# %% imports

import warnings
import numpy as np
import pandas as pd
from collections import namedtuple
from timeit import default_timer as timer

# pip install -q git+https://github.com/google/empirical_calibration
import empirical_calibration as ec


# %% constants

SMALL_POSITIVE = np.nextafter(np.float64(0), np.float64(1))
# not sure if needed: a small nonzero number that can be used as a divisor
SMALL_DIV = SMALL_POSITIVE * 1e16
# 1 / SMALL_DIV  # does not generate warning

QUADRATIC = ec.Objective.QUADRATIC
ENTROPY = ec.Objective.ENTROPY


# %% gec primary function
def gec(wh, xmat, targets,
        target_weights: np.ndarray = None,
        objective: ec.Objective = ec.Objective.ENTROPY,
        increment: float = 0.001):

    # ec.Objective.ENTROPY ec.Objective.QUADRATIC

    # small_positive = np.nextafter(np.float64(0), np.float64(1))
    wh = np.where(wh == 0, SMALL_POSITIVE, wh)

    pop = wh.sum()
    tmeans = targets / pop

    # ompw:  optimal means-producing weights
    ompw, l2_norm = ec.maybe_exact_calibrate(
        covariates=xmat,
        target_covariates=tmeans.reshape((1, -1)),
        baseline_weights=wh,
        # target_weights=np.array([[.25, .75]]), # target priorities
        # target_weights=target_weights,
        autoscale=True,  # doesn't always seem to work well
        # note that QUADRATIC weights often can be zero
        objective=objective,  # ENTROPY or QUADRATIC
        increment=increment
    )
    # print(l2_norm)

    # wh, when multiplied by g, will yield the targets
    g = ompw * pop / wh
    g = np.array(g, dtype=float).reshape((-1, ))  # djb

    return l2_norm, g
