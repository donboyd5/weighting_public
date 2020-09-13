# coding: utf-8
"""
Created on Sun Sep 13 06:33:21 2020

  # #!/usr/bin/env python
  See Peter's code here:
      https://github.com/Peter-Metz/state_taxdata/blob/master/state_taxdata/prepdata.py

  List of official puf files:
      https://docs.google.com/document/d/1tdo81DKSQVee13jzyJ52afd9oR68IwLpYZiXped_AbQ/edit?usp=sharing
      Per Peter latest file is here (8/20/2020 as of 9/13/2020)
      https://www.dropbox.com/s/hyhalpiczay98gz/puf.csv?dl=0
      C:\Users\donbo\Dropbox (Personal)\PUF files\files_based_on_puf2011\2020-08-20

@author: donbo
"""

# %% imports
import taxcalc as tc
import pandas as pd
import numpy as np
from bokeh.io import show, output_notebook

import src.reweight as rw

# setup
# recs = tc.Records() # get the puf, not the cps version

# %% constants
PUFDIR = 'C:/Users/donbo/Dropbox (Personal)/PUF files/files_based_on_puf2011/'
INDIR = PUFDIR + '2020-08-13_djb/'  # puf.csv that I created
# OUTDIR = PUFDIR + 'PUF 2017 Files/'
DATADIR = 'C:/programs_python/weighting/puf/data/'

# raw string allows Windows-styly slashes
# r'C:\Users\donbo\Downloads\taxdata_stuff\puf_2017_djb.csv'

# latest version of the puf that I created with taxdata
PUF_NAME = INDIR + 'puf.csv'
GF_NAME = INDIR + 'growfactors.csv'
WEIGHTS_NAME = INDIR + 'puf_weights.csv'

# latest official puf per peter:
PUF_NAME = r'C:\Users\donbo\Dropbox (Personal)\PUF files\files_based_on_puf2011\2020-08-20\puf.csv'



# agi stubs
# AGI groups to target separately
IRS_AGI_STUBS = [-9e99, 1.0, 5e3, 10e3, 15e3, 20e3, 25e3, 30e3, 40e3, 50e3,
                 75e3, 100e3, 200e3, 500e3, 1e6, 1.5e6, 2e6, 5e6, 10e6, 9e99]
HT2_AGI_STUBS = [-9e99, 1.0, 10e3, 25e3, 50e3, 75e3, 100e3,
                 200e3, 500e3, 1e6, 9e99]


# %% create objects
gfactor = tc.GrowFactors(GF_NAME)
dir(gfactor)

puf = pd.read_csv(PUF_NAME)

recs = tc.Records(data=puf,
                  start_year=2011,
                  gfactors=gfactor,
                  weights=WEIGHTS_NAME,
                  adjust_ratios=None)  # don't use puf_ratios

# recs = tc.Records(data=mypuf,
#                   start_year=2011,
#                   gfactors=gfactor,
#                   weights=WEIGHTS_NAME)  # apply built-in puf_ratios.csv

# %% advance the file
pol = tc.Policy()
calc = tc.Calculator(policy=pol, records=recs)
CYR = 2018
calc.advance_to_year(CYR)
calc.calc_all()


# %% create and examine data frame
puf_2018 = calc.dataframe(variable_list=[], all_vars=True)
puf_2018['pid'] = np.arange(len(puf_2018))

puf_2018.head(10)


# %% save advanced file
BASE_NAME = 'puf_adjusted'

# hdf5 is lightning fast
OUT_HDF = DATADIR + BASE_NAME + '.h5'
# %time puf_2018.to_hdf(OUT_HDF, key='puf_2018', mode='w')
%time puf_2018.to_hdf(OUT_HDF, 'data')  # 1 sec

# csv is slow, only use if need to share files
OUT_CSV = DATADIR + BASE_NAME + '.csv'
%time puf_2018.to_csv(OUT_CSV, index=False)  # 1+ minutes
# chunksize gives minimal speedup
# %time puf_2017.to_csv(OUT_NAME, index=False, chunksize=1e6)


# read back in
%time dfcsv = pd.read_csv(OUT_CSV)  # 8 secs
%time dfhdf = pd.read_hdf(OUT_HDF)  # 1 sec
dfcsv.tail()
dfhdf.tail()
puf_2018.tail()

del(dfcsv)
del(dfhdf)

# %% examine totals
IRSDAT = DATADIR + 'test.csv'
irstot = pd.read_csv(IRSDAT)
irstot

# %time dfhdf = pd.read_hdf(OUT_HDF)  # 1 sec

# filter out filers imputed from CPS
df = puf_2018.copy() # new data frame
df = df.loc[df["data_source"] == 1] # ~7k records dropped
df.info()
desc = df.describe()
cols = df.columns
df.s006.sum() / irstot.iloc[0,].nagi * 100 - 100  # -+1.4%

retcount = df.loc[df['c00100'] >= 10e6].s006.sum()
irscount = irstot.iloc[19].nagi
retcount / irscount * 100 - 100  # +7%

df["IRS_STUB"] = pd.cut(
    df["c00100"],
    IRS_AGI_STUBS,
    labels=list(range(1, len(IRS_AGI_STUBS))),
    right=False,
)

df['wagi'] = df['s006'] * df['c00100']
grouped = df.groupby('IRS_STUB')

comp = irstot.drop(0)[['incrange', 'nagi', 'agi']]
comp['nagi'] = comp['nagi'].astype(float)
comp['nsums'] = grouped.s006.sum()
comp['ndiff'] = comp['nsums'] - comp['nagi']
comp['npdiff'] = comp['ndiff'] / comp['nagi'] * 100
comp['wagi'] = grouped.wagi.sum() / 1000
comp['wdiff'] = comp['wagi'] - comp['agi']
comp['wpdiff'] = comp['wdiff'] / comp['agi'] * 100
comp['wdiff_pctagi'] = comp.wdiff / sum(comp.agi) * 100
comp
comp.round(1)


totals = comp.drop(columns=['incrange', 'npdiff', 'wpdiff', 'wdiff_pctagi']).sum()
totals.ndiff / totals.nagi * 100  # 1.4%
totals.wdiff / totals.agi * 100  # 3.1%


# %% reweight the 2018 puf
# pick an income range to reweight and hit the number of returns and the amount of AGI

def constraints(x, wh, xmat):
    return np.dot(x * wh, xmat)

stub = df.loc[df['IRS_STUB'] == 3].copy()
stub['ones'] = 1.0

cols = ['nagi', 'agi']
xcols = ['ones', 'c00100']
targets = irstot[cols].iloc[3]
targets.agi = targets.agi * 1000.
targets = np.asarray(targets, dtype=float)
type(targets)
targets

wh = np.asarray(stub.s006)
type(wh)

# xmat = stub[['c00100']]
# xmat = np.array()
xmat = np.asarray(stub[xcols], dtype=float)
xmat.shape

x0 = np.ones(wh.size)

t0 = constraints(x0, wh, xmat)
pdiff0 = t0 / targets * 100 - 100
pdiff0
comp[['npdiff', 'wpdiff']].iloc[2]

rwp = rw.Reweight(wh, xmat, targets)
x, info = rwp.reweight(xlb=0.1, xub=10,
                       crange=.0001,
                       ccgoal=10, objgoal=100,
                       max_iter=50)
info['status_msg']

np.quantile(x, [0, .1, .25, .5, .75, .9, 1])

t1 = constraints(x, wh, xmat)
pdiff1 = t1 / targets * 100 - 100
pdiff1


# %% notes
# Peter's mappings of puf to historical table 2
# "n1": "N1",  # Total population
# "mars1_n": "MARS1",  # Single returns number
# "mars2_n": "MARS2",  # Joint returns number
# "c00100": "A00100",  # AGI amount
# "e00200": "A00200",  # Salary and wage amount
# "e00200_n": "N00200",  # Salary and wage number
# "c01000": "A01000",  # Capital gains amount
# "c01000_n": "N01000",  # Capital gains number
# "c04470": "A04470",  # Itemized deduction amount (0 if standard deduction)
# "c04470_n": "N04470",  # Itemized deduction number (0 if standard deduction)
# "c17000": "A17000",  # Medical expenses deducted amount
# "c17000_n": "N17000",  # Medical expenses deducted number
# "c04800": "A04800",  # Taxable income amount
# "c04800_n": "N04800",  # Taxable income number
# "c05800": "A05800",  # Regular tax before credits amount
# "c05800_n": "N05800",  # Regular tax before credits amount
# "c09600": "A09600",  # AMT amount
# "c09600_n": "N09600",  # AMT number
# "e00700": "A00700",  # SALT amount
# "e00700_n": "N00700",  # SALT number

