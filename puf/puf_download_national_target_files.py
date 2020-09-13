# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 09:51:44 2020

@author: donbo
"""
# %% imports
import requests
import pandas as pd
# import ipopt


# %% constants
WEBDIR = 'https://www.irs.gov/pub/irs-soi/'
DOWNDIR = 'C:/programs_python/weighting/puf/downloads/'
DATADIR = 'C:/programs_python/weighting/puf/data/'
PUFDIR = 'C:/programs_python/weighting/puf/'

INCSOURCES = '18in11si.xls'
MARSTAT = '18in12ms.xls'
INCDED = '18in14ar.xls'

files = [INCSOURCES, MARSTAT, INCDED]


# %% download and save files

for f in files:
    print(f)
    url = WEBDIR + f
    path = DOWNDIR + f
    r = requests.get(url)
    print(r.status_code)
    with open(path, "wb") as file:
        file.write(r.content)


# %% parse and save important file contents
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html

def xlrange(io, firstrow=1, lastrow=None, usecols=None, colnames=None):
    # firstrow and lastrow are 1-based
    if colnames is None:
        if usecols is None:
            colnames = None
        else:
            colnames = usecols.split(',')
    nrows = None
    if lastrow is not None:
        nrows = lastrow - firstrow + 1
    df = pd.read_excel(io,
                       header=None,
                       names=colnames,
                       usecols=usecols,
                       skiprows=firstrow - 1,
                       nrows=nrows)
    return df


files

path = DOWNDIR + files[0]
inccols = 'A, B, D, G, I'
# incols = 'A:D'
colnames = ['incrange', 'nagi', 'agi', 'nagi_taxret', 'agi_taxret']
df = xlrange(path, usecols=inccols, colnames=colnames,
        firstrow=10, lastrow=29)

# are reported totals close enough to sums of values that we can drop reported?
df.iloc[1:, 1:].sum()
df.iloc[0]
df.iloc[1:, 1:].sum() - df.iloc[0]  # yes

# create a mapping of incrange to stubs


# %% write results to file
df.to_csv(DATADIR + 'test.csv', index=False)


pd.read_csv(DATADIR + 'test.csv')



# %% create targets suitable for reweighting national puf



