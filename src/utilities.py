# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 05:02:59 2020

@author: donbo
"""

# %% imports
import sys
import pandas as pd


# %% utility functions


def getmem(objects=dir()):
    """Memory used, not including objects starting with '_'.

        Example:  getmem().head(10)
    """
    mb = 1024**2
    mem = {}
    for i in objects:
        if not i.startswith('_'):
            mem[i] = sys.getsizeof(eval(i))
    mem = pd.Series(mem) / mb
    mem = mem.sort_values(ascending=False)
    return mem

