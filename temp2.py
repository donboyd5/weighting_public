# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 18:00:03 2020
https://stackoverflow.com/questions/29629103/simple-python-multiprocessing-function-in-spyder-doesnt-output-results

@author: donbo
"""
# multiproc_test.py

import random
import multiprocessing
from timeit import default_timer as timer


def list_append(count, id, out_list):
    """
    Creates an empty list and then appends a
    random number to the list 'count' number
    of times. A CPU-heavy operation!
    """
    for i in range(count):
        out_list.append(random.random())

if __name__ == "__main__":
    size = 1000000000   # Number of random numbers to add
    procs = 6   # Number of processes to create
    t1 = timer()

    # Create a list of jobs and then iterate through
    # the number of processes appending each process to
    # the job list
    jobs = []
    for i in range(0, procs):
        out_list = list()
        process = multiprocessing.Process(target=list_append,
                                          args=(size, i, out_list))
        jobs.append(process)

    # Start the processes (i.e. calculate the random number lists)
    for j in jobs:
        j.start()

    # Ensure all of the processes have finished
    for j in jobs:
        j.join()

    t2 = timer()
    print(t2 - t1)
    print("List processing complete.")

# time python temp2.py