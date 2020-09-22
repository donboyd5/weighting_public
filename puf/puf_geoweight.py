# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 06:56:25 2020

Parallel group_by?
https://pandas.pydata.org/pandas-docs/stable/ecosystem.html?highlight=parallel

https://stackoverflow.com/questions/26187759/parallelize-apply-after-pandas-groupby

https://stackoverflow.com/questions/1704401/is-there-a-simple-process-based-parallel-map-for-python
https://towardsdatascience.com/10x-faster-parallel-python-without-python-multiprocessing-e5017c93cce1
https://github.com/ray-project/ray

@author: donbo
"""

# %% imports
import sys
import requests
import pandas as pd
import numpy as np
import src.microweight as mw
import src.make_test_problems as mtp


# %% constants
WEBDIR = 'https://www.irs.gov/pub/irs-soi/'
DOWNDIR = 'C:/programs_python/weighting/puf/downloads/'
DATADIR = 'C:/programs_python/weighting/puf/data/'
# PUFDIR = 'C:/programs_python/weighting/puf/'
HDFDIR = 'C:/programs_python/weighting/puf/ignore/'

HT2_2018 = "18in55cmagi.csv"

# agi stubs
# AGI groups to target separately
IRS_AGI_STUBS = [-9e99, 1.0, 5e3, 10e3, 15e3, 20e3, 25e3, 30e3, 40e3, 50e3,
                 75e3, 100e3, 200e3, 500e3, 1e6, 1.5e6, 2e6, 5e6, 10e6, 9e99]
HT2_AGI_STUBS = [-9e99, 1.0, 10e3, 25e3, 50e3, 75e3, 100e3,
                 200e3, 500e3, 1e6, 9e99]


# %% crosswalks
# This is Peter's xwalk plus mine -- it includes more than we will use
PUFHT2_XWALK = {
    'nret_all': 'N1',  # Total population
    'nret_mars1': 'MARS1',  # Single returns number
    'nret_mars2': 'MARS2',  # Joint returns number
    'c00100': 'A00100',  # AGI amount
    'e00200': 'A00200',  # Salary and wage amount
    'e00200_n': 'N00200',  # Salary and wage number
    'e00300': 'A00300',  # Taxable interest amount
    'e00600': 'A00600',  # Ordinary dividends amount
    'c01000': 'A01000',  # Capital gains amount
    'c01000_n': 'N01000',  # Capital gains number
    # check Social Security
    # e02400 is Total Social Security
    # A02500 is Taxable Social Security benefits amount
    'e02400': 'A02500',  # Social Security total (2400)
    'c04470': 'A04470',  # Itemized deduction amount (0 if standard deduction)
    'c04470_n': 'N04470',  # Itemized deduction number (0 if standard deduction)
    'c17000': 'A17000',  # Medical expenses deducted amount
    'c17000_n': 'N17000',  # Medical expenses deducted number
    'c04800': 'A04800',  # Taxable income amount
    'c04800_n': 'N04800',  # Taxable income number
    'c05800': 'A05800',  # Regular tax before credits amount
    'c05800_n': 'N05800',  # Regular tax before credits amount
    'c09600': 'A09600',  # AMT amount
    'c09600_n': 'N09600',  # AMT number
    'e00700': 'A00700',  # SALT amount
    'e00700_n': 'N00700',  # SALT number
    # check pensions
    # irapentot: IRAs and pensions total e01400 + e01500
    # A01750: Taxable IRA, pensions and annuities amount
    'irapentot': 'A01750',
}
PUFHT2_XWALK
# CAUTION: reverse xwalk relies on having only one keyword per value
HT2PUF_XWALK = {val: kw for kw, val in PUFHT2_XWALK.items()}
HT2PUF_XWALK
list(HT2PUF_XWALK.keys())


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


# %% program functions

def wsum(grp, sumvars, wtvar):
    """ Returns data frame row with weighted sums of selected variables.

        grp: a dataframe (typically a dataframe group)
        sumvars: the variables for which we want weighted sums
        wtvar:  the weighting variable
    """
    return grp[sumvars].multiply(grp[wtvar], axis=0).sum()


def constraints(x, wh, xmat):
    return np.dot(x * wh, xmat)



# %% ONETIME download Historical Table 2
files = [HT2_2018]

for f in files:
    print(f)
    url = WEBDIR + f
    path = DOWNDIR + f
    r = requests.get(url)
    print(r.status_code)
    with open(path, "wb") as file:
        file.write(r.content)


# %% read and adjust Historical Table 2

ht2 = pd.read_csv(DOWNDIR + HT2_2018, thousands=',')
ht2
ht2.info()
ht2.STATE.describe()  # 54 -- states, DC, US, PR, OA
ht2.STATE.value_counts().sort_values()
ht2.groupby('STATE').STATE.count()  # alpha order
ht2.head()
ht2.columns.to_list()
# # convert all strings to numeric
# stn = ht2raw.columns.to_list()
# stn.remove('STATE')
# ht2[stn] = ht2raw[stn].apply(pd.to_numeric, errors='coerce', axis=1)
ht2

h2stubs = pd.DataFrame([
    [0, 'All income ranges'],
    [1, 'Under $1'],
    [2, '$1 under $10,000'],
    [3, '$10,000 under $25,000'],
    [4, '$25,000 under $50,000'],
    [5, '$50,000 under $75,000'],
    [6, '$75,000 under $100,000'],
    [7, '$100,000 under $200,000'],
    [8, '$200,000 under $500,000'],
    [9, '$500,000 under $1,000,000'],
    [10, '$1,000,000 or more']],
    columns=['h2stub', 'h2range'])
h2stubs
h2stubs.info()

# agi_stub	aginame
# 0	All income ranges
# 1	Under $1
# 2	$1 under $10,000
# 3	$10,000 under $25,000
# 4	$25,000 under $50,000
# 5	$50,000 under $75,000
# 6	$75,000 under $100,000
# 7	$100,000 under $200,000
# 8	$200,000 under $500,000
# 9	$500,000 under $1,000,000
# 10	$1,000,000 or more


# %% get reweighted national puf
PUF_RWTD = HDFDIR + 'puf2018_reweighted' + '.h5'
pufrw = pd.read_hdf(PUF_RWTD)  # 1 sec
pufrw.columns.sort_values()


# %% prepare puf subset and weighted sums
# create a subset with ht2stub variable
pufsub = pufrw.copy()
pufsub['HT2_STUB'] = pd.cut(
    pufsub['c00100'],
    HT2_AGI_STUBS,
    labels=list(range(1, len(HT2_AGI_STUBS))),
    right=False)
pufsub.columns.sort_values().tolist()  # show all column names

pufsub[['pid', 'c00100', 'HT2_STUB', 'IRS_STUB']].sort_values(by='c00100')

# create list of target vars
alltargvars = ['nret_all', 'nret_mars1', 'nret_mars2',
            'c00100', 'e00200', 'e00300', 'e00600',
            'c01000', 'e02400', 'c04800', 'irapentot']
alltargvars

pufsums = pufsub.groupby('HT2_STUB').apply(wsum,
                                            sumvars=alltargvars,
                                            wtvar='wtnew')
pufsums = pufsums.append(pufsums.sum().rename(0)).sort_values('HT2_STUB')
pufsums['HT2_STUB'] = pufsums.index
pufsums



# %% prepare compatible ht2 subset
# we'll rename HT2 columns to be like puf

# prepare a list of column names we want from ht2
ht2_all = list(HT2PUF_XWALK.keys())  # superset
ht2_all
prefixes = ('N0')  # must be tuple, not a list
ht2_use = [x for x in ht2_all if not x.startswith(prefixes)]
drops = ['N17000', 'A17000', 'A04470', 'A05800', 'A09600', 'A00700']
ht2_use = [x for x in ht2_use if x not in drops]
ht2_use
# ht2_use.remove('N17000')
ht2_use.append('STATE')
ht2_use.append('AGI_STUB')
ht2_use

# get those columns, rename as needed, and create new columns
ht2_sub = ht2[ht2_use].copy()
ht2_sub = ht2_sub.rename(columns=HT2PUF_XWALK)
ht2_sub = ht2_sub.rename(columns={'AGI_STUB': 'HT2_STUB'})
# multiply dollar values by 1000
ht2_sub.columns
dollcols = ['c00100', 'e00200', 'e00300',
            'e00600', 'c01000', 'e02400', 'c04800', 'irapentot']
dollcols
ht2_sub[dollcols] = ht2_sub[dollcols] * 1000
ht2_sub


# %% compare pufsums to HT2 for US
ht2sums = ht2_sub.query('STATE=="US"')
ht2sums = ht2sums.drop(columns=['STATE'])
ht2sums.columns

pufsums.columns

ht2sums
pufsums

round(pufsums.drop(columns='HT2_STUB') / ht2sums.drop(columns='HT2_STUB') * 100 - 100)
# e02400 is way off, c04800 has some trouble, and irapentot is way off, so don't use them
# the rest look good

targvars = ['nret_all', 'nret_mars1', 'nret_mars2', 'c00100', 'e00200',
            'e00300', 'e00600']
targvars + ['HT2_STUB']


# %% prepare to geoweight
pufsub.columns
pufsub[['HT2_STUB', 'pid']].groupby(['HT2_STUB']).agg(['count'])

stub = 6
pufstub = pufsub.query('HT2_STUB == @stub')[['pid', 'HT2_STUB', 'wtnew'] + targvars]
pufstub

ht2stub = ht2_sub.query('HT2_STUB == @stub & STATE != "US"')[['STATE', 'HT2_STUB'] + targvars]
ht2stub

wh = pufstub.wtnew.to_numpy()
xmat = np.asarray(pufstub[targvars], dtype=float)
xmat.shape
targets = ht2stub.drop(columns=['STATE', 'HT2_STUB'])
targets = np.asarray(targets, dtype=float)

g = mw.Microweight(wh, xmat, targets)

# look at the inputs
g.wh
g.xmat
g.geotargets

g.wh.shape
g.xmat.shape
g.geotargets.shape

type(g.wh)
type(g.xmat)
type(g.geotargets)

# solve for state weights
g.geoweight()

# examine results
g.elapsed_minutes
g.result  # this is the result returned by the solver
dir(g.result)
g.result.cost  # objective function value at optimum
g.result.message

# optimal values
g.beta_opt  # beta coefficients, s x k
g.delta_opt  # delta constants, 1 x h
g.whs_opt  # state weights
g.geotargets_opt

np.round(g.result.fun, 1)
np.round(g.result.fun.reshape(53, 7), 1)








# %% geoweight
mtp.Problem.help()

p = mtp.Problem(h=10000, s=50, k=10)
# p = mtp.Problem(h=20000, s=30, k=10)  # moderate-sized problem, < 1 min

# I don't think our problems for a single AGI range will get bigger
# than the one below:
#   30k tax records, 50 states, 30 characteristics (targets) per state
# but problems will be harder to solve with real data
# p = mtp.Problem(h=30000, s=50, k=30) # took 31 mins on my computer

mw.Microweight.help()

p.xmat.shape
p.targets.shape
g1 = mw.Microweight(p.wh, p.xmat, p.targets)

# look at the inputs
g1.wh
g1.xmat
g1.geotargets

# solve for state weights
g1.geoweight()

# examine results
g1.elapsed_minutes
g1.result  # this is the result returned by the solver
dir(g1.result)
g1.result.cost  # objective function value at optimum
g1.result.message

# optimal values
g1.beta_opt  # beta coefficients, s x k
g1.delta_opt  # delta constants, 1 x h
g1.whs_opt  # state weights
g1.geotargets_opt




# %% Peter's  crosswalks
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

