# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 09:53:37 2020

# IMPORTANT:
https://stackoverflow.com/questions/48078722/no-multiprocessing-print-outputs-spyder?noredirect=1&lq=1

https://www.kth.se/blogs/pdc/2019/02/parallel-programming-in-python-multiprocessing-part-1/

https://pandas.pydata.org/pandas-docs/stable/ecosystem.html?highlight=parallel
https://homes.cs.washington.edu/~jmschr/lectures/Parallel_Processing_in_Python.html
https://joblib.readthedocs.io/en/latest/parallel.html

https://github.com/spyder-ide/spyder/issues/2937

https://stackoverflow.com/questions/20548628/how-to-do-parallel-programming-in-python

@author: donbo
"""

# %% functions


def f(t):
    name, a, b, c = t
    return (name, a + b + c)


def my_function(x):
    return x**2


# %% t1
from multiprocessing import Pool

def square(x):
    # calculate the square of the value of x
    return x*x

if __name__ == '__main__':

    # Define the dataset
    dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    # Output the dataset
    print ('Dataset: ' + str(dataset))

    # Run this with a pool of 5 agents having a chunksize of 3 until finished
    agents = 5
    chunksize = 3
    with Pool(processes=agents) as pool:
        result = pool.map(square, dataset, chunksize)

    # Output the result
    print ('Result:  ' + str(result))


# %% more
from math import sqrt
from joblib import Parallel, delayed

n = int(1e7)

# single-core code
%time sqroots_1 = [sqrt(i ** 2) for i in range(n)]

# parallel code
# WAY TOO SLOW
%time sqroots_2 = Parallel(n_jobs=6)(delayed(sqrt)(i ** 2) for i in range(n))


sum(sqroots_1)
sum(sqroots_2)


# %% looptest
import multiprocessing
import numpy as np



if __name__ == "__main__":
   #the previous line is necessary under windows to not execute
   # main module on each child under windows

   X = np.random.normal(size=(10, 3))
   F = np.zeros((10, ))

   pool = multiprocessing.Pool(processes=16)
   # if number of processes is not specified, it uses the number of core
   F[:] = pool.map(my_function, (X[i,:] for i in range(10)) )


# %% test
from multiprocessing import Pool

data = [('bla', 1, 3, 7), ('spam', 12, 4, 8), ('eggs', 17, 1, 3)]
data

p = Pool(4)
results = p.map(f, data)
print results


# %% test2
from multiprocessing import Pool, freeze_support, cpu_count
import os

all_args = [(parameters1, parameters2) for i in range(24)]

# call freeze_support() if in Windows
if os.name == "nt":
    freeze_support()

# you can use whatever, but your machine core count is usually a good choice (although maybe not the best)
pool = Pool(cpu_count())

def wrapped_some_function_call(args):
    """
    we need to wrap the call to unpack the parameters
    we build before as a tuple for being able to use pool.map
    """
    sume_function_call(*args)

results = pool.map(wrapped_some_function_call, all_args)
total_error = sum(results)


# %% dask0
data.income.apply(lambda x: x * 1000).head(5)
import dask.dataframe as dd

df = pd.DataFrame({'A': np.random.randint(1000, size=100000),
                   'B': np.random.randint(1000, size=100000)})
df

ddf = dd.from_pandas(df, npartitions=4)

def add_squares(a, b):
    return a**2 + b**2

df.A.apply(lambda x: x * 1000).head(5)
ddf.A.apply(lambda x: x * 1000).head(5)
ddf.A.apply(lambda x: x * 1000, meta=('A', 'float')).head(5)

ddf.head(5)

data.groupby('occupation').income.mean().compute()

# A memory efficient style is to create pipelines of operations and trigger
# a final compute at the end.
datapipe = data[data.age < 20]
datapipe = datapipe.groupby('income').mean()
datapipe.head(4)


# %timeit ddf ['z'] = ddf.map_partitions(add_squares, meta=(None, 'int64')).compute()




# %% swifter
import pandas as pd
import numpy as np

df = pd.DataFrame({'X': np.random.randint(1000, size=100000),
                   'Y': np.random.randint(1000, size=100000)})
df


def add_squares(a, b):
    return a**2 + b**2


add_squares(3, 4)


%timeit  df['add_squares'] = df.apply(add_squares, b='b', axis=1)
import dask.dataframe as dd
ddf = dd.from_pandas(df, npartitions=4)

%timeit ddf ['z'] = ddf.map_partitions(add_squares, meta=(None, 'int64')).compute()

import swifter
df['add_sq'] = df.swifter.apply(lambda row:add_squares(row.X,row.Y),axis=1)
df




# %% stuff1
from concurrent.futures import ProcessPoolExecutor, as_completed
ppe = ProcessPoolExecutor(4)

futures = []
results = []
for group in np.split(df, 4):
    p = ppe.submit(aggregate_fun, group)
    futures.append(p)

for future in as_completed(futures):
    r = future.result()
    results.append(r)

df_output = pd.concat(results)


# %% mp again
import concurrent.futures
import math

PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True

def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
            print('%d is prime: %s' % (number, prime))

# if __name__ == '__main__':
#     main()

main()



# %% mp properly
from multiprocessing import Pool


def f(x):
    return x*x


if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))


# %% using mp
import multiprocessing as mp


def sum_up_to(number):
    return sum(range(1, number + 1))


sum_up_to(10)

a_pool = mp.Pool()


result = a_pool.map(sum_up_to, range(10))



# %% stuff
import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed
ppe = ProcessPoolExecutor(4)

futures = []
results = []
for group in np.split(df, 4):
    p = ppe.submit(aggregate_fun, group)
    futures.append(p)

for future in as_completed(futures):
    r = future.result()
    results.append(r)

df_output = pd.concat(results)


def summarize(x):
    """Summarize the data set by returning the length and the sum."""
    return len(x), sum(x)


summarize(range(0, 1000))

x = np.random.randn(int(1e7)) + 8.342
x0, x1 = summarize(x)
print(x1 / x0)



# %% pandarallel
# needs linux window
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

# df.apply(func)
df.parallel_apply(func)



# %% ray
# https://docs.ray.io/en/latest/
import numpy as np
import psutil
import ray
import scipy.signal

num_cpus = psutil.cpu_count(logical=False)

ray.init(num_cpus=num_cpus)

@ray.remote
def f(image, random_filter):
    # Do some image processing.
    return scipy.signal.convolve2d(image, random_filter)[::5, ::5]

filters = [np.random.normal(size=(4, 4)) for _ in range(num_cpus)]

# Time the code below.

for _ in range(10):
    image = np.zeros((3000, 3000))
    image_id = ray.put(image)
    ray.get([f.remote(image_id, filters[i]) for i in range(num_cpus)])


# %% multiprocessing
from multiprocessing import Pool
import numpy as np
import psutil
import scipy.signal

num_cpus = psutil.cpu_count(logical=False)

def f(args):
    image, random_filter = args
    # Do some image processing.
    return scipy.signal.convolve2d(image, random_filter)[::5, ::5]

pool = Pool(num_cpus)

filters = [np.random.normal(size=(4, 4)) for _ in range(num_cpus)]

# Time the code below.

for _ in range(10):
    image = np.zeros((3000, 3000))
    pool.map(f, zip(num_cpus * [image], filters))


# %% pandas
# https://stackoverflow.com/questions/26187759/parallelize-apply-after-pandas-groupby
## make some example data
import pandas as pd

np.random.seed(1)
n=10000
df = pd.DataFrame({'mygroup' : np.random.randint(1000, size=n),
                   'data' : np.random.rand(n)})
grouped = df.groupby('mygroup')

dflist = []
for name, group in grouped:
    dflist.append(group)

from ipyparallel import Client
rc = Client()
lview = rc.load_balanced_view()
lview.block = True

def myFunc(inDf):
    inDf['newCol'] = inDf.data ** 10
    return inDf

%%time
serial_list = map(myFunc, dflist)


# %% dask
# https://stackoverflow.com/questions/45545110/make-pandas-dataframe-apply-use-all-cores
# http://python.omics.wiki/multiprocessing_map

