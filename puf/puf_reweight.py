# coding: utf-8
"""
  # #!/usr/bin/env python
  See Peter's code here:
      https://github.com/Peter-Metz/state_taxdata/blob/master/state_taxdata/prepdata.py

  List of official puf files:
      https://docs.google.com/document/d/1tdo81DKSQVee13jzyJ52afd9oR68IwLpYZiXped_AbQ/edit?usp=sharing
      Per Peter latest file is here (8/20/2020 as of 9/13/2020)
      https://www.dropbox.com/s/hyhalpiczay98gz/puf.csv?dl=0
      C:\Users\donbo\Dropbox (Personal)\PUF files\files_based_on_puf2011\2020-08-20
      # raw string allows Windows-style slashes
      # r'C:\Users\donbo\Downloads\taxdata_stuff\puf_2017_djb.csv'

https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

@author: donbo
"""

# %% imports
import taxcalc as tc
import pandas as pd
import numpy as np
from bokeh.io import show, output_notebook

import src.reweight as rw


# %% constants
DATADIR = 'C:/programs_python/weighting/puf/data/'
HDFDIR = 'C:/programs_python/weighting/puf/ignore/'
BASE_NAME = 'puf_adjusted'
PUF_HDF = HDFDIR + BASE_NAME + '.h5'  # hdf5 is lightning fast

# agi stubs
# AGI groups to target separately
IRS_AGI_STUBS = [-9e99, 1.0, 5e3, 10e3, 15e3, 20e3, 25e3, 30e3, 40e3, 50e3,
                 75e3, 100e3, 200e3, 500e3, 1e6, 1.5e6, 2e6, 5e6, 10e6, 9e99]
HT2_AGI_STUBS = [-9e99, 1.0, 10e3, 25e3, 50e3, 75e3, 100e3,
                 200e3, 500e3, 1e6, 9e99]


# %% get advanced file
%time puf_2018 = pd.read_hdf(PUF_HDF)  # 1 sec
puf_2018.tail()


# %% get and prepare targets
IRSDAT = DATADIR + 'targets2018.csv'
irstot = pd.read_csv(IRSDAT)
irstot
# drop targets for which I haven't yet set column descriptions
irstot = irstot.dropna(axis=0, subset=['column_description'])
irstot
irstot.columns

# check counts
irstot[['src', 'variable', 'value']].groupby(['src', 'variable']).agg(['count'])
irstot[['variable', 'value']].groupby(['variable']).agg(['count'])  # unique list

# quick check to make sure duplicate variables have same values
check = irstot[irstot.irsstub == 0][['src', 'variable']]
idups = check.duplicated(subset='variable', keep=False)
check[idups].sort_values(['variable', 'src'])
dupvars = check[idups]['variable'].unique()
dupvars

# now check values
keep = (irstot.variable.isin(dupvars)) & (irstot.irsstub==0)
dups = irstot[keep][['variable', 'src', 'column_description', 'value']]
dups.sort_values(['variable', 'src'])
# looks ok - we can select any of the duplicates we want


# %% prepare a puf summary for several target variables
# filter out filers imputed from CPS
df = puf_2018.copy()  # new data frame
df = df.loc[df["data_source"] == 1]  # ~7k records dropped
df['IRS_STUB'] = pd.cut(
    df['c00100'],
    IRS_AGI_STUBS,
    labels=list(range(1, len(IRS_AGI_STUBS))),
    right=False,
)
df.columns.sort_values().tolist()  # show all column names

vars = ['agi', 'wages', 'nret_all']
pufvars = ['pid', 'IRS_STUB', 'c00100', 'e00200', 's006']

df2 = df[pufvars].copy()
# df2.loc[:, ('nret')] = 1  # need to understand copy
df2['nret'] = 1
df2.head()


def wsum(grp, sumvars, wtvar):
    return grp[sumvars].multiply(grp[wtvar], axis=0).sum()


df3 = df2.groupby('IRS_STUB').apply(wsum,
                              sumvars=['nret', 'c00100', 'e00200'],
                              wtvar='s006')

df3 = df3.append(df3.sum().rename(0)).sort_values('IRS_STUB')
df3

# %% combine IRS totals and PUF totals and compare
keep = (irstot.src == '18in14ar.xls') & (irstot.variable.isin(['nret_all', 'agi', 'wages']))
irscomp = irstot[['irsstub', 'incrange', 'variable', 'value']][keep]
irscomp = irscomp.rename(columns={'value': 'irs'})
irscomp['irs'] = pd.Series.astype(irscomp['irs'], 'float')
irscomp
irscomp.info()

pufcomp = df3
pufcomp['irsstub'] = pufcomp.index
pufcomp = pufcomp.rename(columns={'nret': 'nret_all',
                                  'c00100': 'agi',
                                  'e00200': 'wages'})
pufcomp[['agi', 'wages']] = pufcomp[['agi', 'wages']] / 1e3
pufcomp = pd.melt(pufcomp, id_vars=['irsstub'], value_name='puf')
pufcomp

comp = pd.merge(irscomp, pufcomp, on=['irsstub', 'variable'])
comp['diff'] = comp['puf'] - comp['irs']
comp['pdiff'] = comp['diff'] / comp['irs'] * 100
format_mapping = {'irs': '{:,.0f}',
                  'puf': '{:,.0f}',
                  'diff': '{:,.0f}',
                  'pdiff': '{:,.1f}'}
for key, value in format_mapping.items():
    comp[key] = comp[key].apply(value.format)

# comp['diffpctagi'] = comp['diff'] / comp[[('irsstub'==0)]]['irs']
# comp.pdiff = comp.pdiff.round(decimals=1)
comp

comp[(comp['variable'] == 'nret_all')]
comp[(comp['variable'] == 'agi')]
comp[(comp['variable'] == 'wages')]

# compshow = comp[(comp['variable'] == 'nret_all')]
# compshow.style.format({"irs": "${:20,.0f}"})

# pd.options.display.float_format = '{:,.1f}'.format
# pd.reset_option('display.float_format')



# %% misc

df2.pivot_table(index='IRS_STUB',
               margins=True,
               margins_name='0',  # defaults to 'All'
               aggfunc=sum)

# map puf names to irstot variable values

df.info()
desc = df.describe()
cols = df.columns

tmp = irstot.loc[(irstot.src == '18in11si.xls') &
             (irstot.irsstub == 0) &
             (irstot.variable == 'nret_all')]
nret_all = tmp.iloc[0].value

df.s006.sum() / nret_all * 100 - 100  # -+1.4%
# djb pick up here ----

retcount = df.loc[df['c00100'] >= 10e6].s006.sum()
rec = irstot.loc[(irstot.src == '18in11si.xls') &
                 (irstot.irsstub == 19) &
                 (irstot.variable == 'nret_all')]
irscount = rec.iloc[0].value
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

    # Maps PUF variable names to HT2 variable names
VAR_CROSSWALK = {
    "n1": "N1",  # Total population
    "mars1_n": "MARS1",  # Single returns number
    "mars2_n": "MARS2",  # Joint returns number
    "c00100": "A00100",  # AGI amount
    "e00200": "A00200",  # Salary and wage amount
    "e00200_n": "N00200",  # Salary and wage number
    "c01000": "A01000",  # Capital gains amount
    "c01000_n": "N01000",  # Capital gains number
    "c04470": "A04470",  # Itemized deduction amount (0 if standard deduction)
    "c04470_n": "N04470",  # Itemized deduction number (0 if standard deduction)
    "c17000": "A17000",  # Medical expenses deducted amount
    "c17000_n": "N17000",  # Medical expenses deducted number
    "c04800": "A04800",  # Taxable income amount
    "c04800_n": "N04800",  # Taxable income number
    "c05800": "A05800",  # Regular tax before credits amount
    "c05800_n": "N05800",  # Regular tax before credits amount
    "c09600": "A09600",  # AMT amount
    "c09600_n": "N09600",  # AMT number
    "e00700": "A00700",  # SALT amount
    "e00700_n": "N00700",  # SALT number
}

