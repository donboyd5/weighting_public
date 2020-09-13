# coding: utf-8
"""
Created on Sun Sep 13 06:33:21 2020

  # #!/usr/bin/env python
  See Peter's code here:
      https://github.com/Peter-Metz/state_taxdata/blob/master/state_taxdata/prepdata.py

@author: donbo
"""

# %% imports
import taxcalc as tc
import pandas as pd
import numpy as np
from bokeh.io import show, output_notebook

import src.reweight

# setup
# recs = tc.Records() # get the puf, not the cps version

# %% constants
PUFDIR = 'C:/Users/donbo/Dropbox (Personal)/PUF files/files_based_on_puf2011/'
INDIR = PUFDIR + '2020-08-13_djb/'
# OUTDIR = PUFDIR + 'PUF 2017 Files/'
DATADIR = 'C:/programs_python/weighting/puf/data/'

# raw string allows Windows-styly slashes
# r'C:\Users\donbo\Downloads\taxdata_stuff\puf_2017_djb.csv'


PUF_NAME = INDIR + 'puf.csv'
GF_NAME = INDIR + 'growfactors.csv'
WEIGHTS_NAME = INDIR + 'puf_weights.csv'


# %% create objects
gfactor = tc.GrowFactors(GF_NAME)
dir(gfactor)
mypuf = pd.read_csv(PUF_NAME)

recs = tc.Records(data=mypuf,
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
OUT_CSV = DATADIR + BASE_NAME + '.csv'
%time puf_2018.to_csv(OUT_CSV, index=False)  # 1+ minutes
# chunksize gives minimal speedup
# %time puf_2017.to_csv(OUT_NAME, index=False, chunksize=1e6)

# hdf5 is lightning fast
OUT_HDF = DATADIR + BASE_NAME + '.h5'
# %time puf_2018.to_hdf(OUT_HDF, key='puf_2018', mode='w')
%time puf_2018.to_hdf(OUT_HDF, 'data')  # 1 sec

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

# filter out filers imputed from CPS
df = puf_2017.loc[puf_2017["data_source"] == 1] # ~7k records dropped
cols = df.columns
df.s006.sum() / irstot.iloc[0,].nagi * 100 - 100  # -0.12%

retcount = df.loc[df['c00100'] >= 10e6].s006.sum()
irscount = irstot.iloc[19].nagi
retcount / irscount * 100 - 100  # -6.9%

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

