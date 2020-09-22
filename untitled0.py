# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 04:33:48 2020

@author: donbo
"""

dfmi = pd.DataFrame([list('abcd'),
                     list('efgh'),
                     list('ijkl'),
                     list('mnop')],
                    columns=pd.MultiIndex.from_product([['one', 'two'],
                                                        ['first', 'second']]))

dfmi
dfmi.info()
dfmi['one']['second']
dfmi.loc[:, ('one', 'second')]
dfmi.loc[slice(None), ('one', 'second')]


%timeit dfmi['one']['second']
%timeit dfmi.loc[:, ('one', 'second')]

dfmi.loc[1:2, ('one', 'second')]

# what is a slice?
slice(10)
slice(10, 15)
slice(10, 15, 1)

a = ("a", "b", "c", "d", "e", "f", "g", "h")
type(a)
x = slice(2)  # returns the (0, 1) positions
x = slice(3, 5)  # returns positions (3, 4) (starting from 0)
a[x]
a[2]
a[slice(1)]
a[slice(4)]  # positions 0, 1, 2, 3
a[slice(2, 7, 2)]  # positions 2, 4, 6? c, e, g
a[slice(0, 8, 3)]  # 0, 3, 6  a, d, g
a[slice(0, 20, 3)]  # 0, 3, 6, 9 etc (BAD) still returns 0, 3, 6 not error

dfc = pd.DataFrame({'a': ['one', 'one', 'two',
                          'three', 'two', 'one', 'six'],
                    'c': np.arange(7)})

# with copy -- does not change dfc
dfd = dfc.copy()
mask = dfd['a'].str.startswith('o')
dfd.loc[mask, 'c'] = 42
dfd
dfc

# no copy -- changes dfc!!
dfe = dfc
mask = dfe['a'].str.startswith('o')
dfe.loc[mask, 'c'] = 42
dfe
dfc

dfc.nbytes

import sys
sys.getsizeof(dfc)
sys.getsizeof(dfd)
sys.getsizeof(dfe)


for i in dir():
    if not i.startswith('_'):
        print(i, sys.getsizeof(eval(i)))

mem = {}
mem[i] = sys.getsizeof(eval(i))
mem['djb'] = 300
mem['abc'] =30
mem
sorted(mem)
sorted(mem.values())
list(sorted(mem.items(), key=lambda x: x[1]))


def getmem(dir=dir()):
    mem = {}
    for i in dir:
        if not i.startswith('_'):
            mem[i] = sys.getsizeof(eval(i))
    mem = list(sorted(mem.items(), key=lambda x: x[1], reverse=True))
    print(mem)
    return mem

mem = getmem()
mem.head(10)
for i in mem:
    print(i)

mem.items()
type(mem)


def getmem(objects=dir()):
    mem = {}
    for i in objects:
        if not i.startswith('_'):
            mem[i] = sys.getsizeof(eval(i))
    mem = pd.Series(mem)
    mem = mem.sort_values(ascending=False)
    return mem

dir[1]

len(dir)

for n in range(20000000, 20000100):
    for x in range(2, n):
        if n % x == 0:
            # print(n, 'equals', x, '*', n//x)
            break
    else:
        # loop fell through without finding a factor
        print(n, 'is a prime number')


dict(sape=4139, guido=4127, jack=4098)

works if VAR_CROSSWALK is a dict
puf_2017.rename(columns=self.VAR_CROSSWALK, inplace=True)


tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                     'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two',
                     'one', 'two', 'one', 'two']]))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
df2 = df[:4]
df2


                     A         B
first second
bar   one    -0.727965 -0.589346
      two     0.339969 -0.693205
baz   one    -0.339355  0.593616
      two     0.884345  1.591431

