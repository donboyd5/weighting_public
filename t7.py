# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 19:27:56 2020

@author: donbo
https://docs.dask.org/en/latest/setup/single-distributed.html
http://localhost:8787/status

"""

# from dask.distributed import Client
# client = Client()

import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

cluster = LocalCluster()
client = Client(cluster)

df = pd.DataFrame({'A': np.random.randint(1000, size=100000),
                   'B': np.random.randint(1000, size=100000)})
df

ddf = dd.from_pandas(df, npartitions=4)




client.close()
cluster.close()


# cluster.run_on_scheduler(lambda dask_scheduler=None:
#     dask_scheduler.close() & sys.exit(0))
