# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 05:13:47 2020

@author: donbo
"""
arr_m = np.arange(12).reshape(2, 2, 3)

arr_m[1:5:2, ::3]

m = np.arange(15).reshape(4, 3)
m
m[(1, 2)]
m[(1,)]

m[(1, (0, 2))]
m[(1, [0, 2])]
m[[1, [0, 2]]]  # NOT ok - MUST start with a tuple

t = (1, (0, 2))
m[t] = 99

tt = np.array([[1, [0,2]]])
tt = (t, [0,2])
m[tt]

type([0,2])
drops = [(6, (6))]
drops = [(1, (1, 2))]
for ij in drops: mask[ij] = False
m = np.arange(15).reshape(5, 3)
for ij in drops: m[ij] = 99

tt = [(1, (0, 2)), (3, (1, 2))]
for t in tt: m[t] = 99
m

# {6: (5, 6)}
dt ={1: (0, 2), 3: (1, 2)}

for key in dt:
    print(key, '->', dt[key])

dt ={1: (0, 2), 3: (1, 2)}
m = np.arange(15).reshape(5, 3)
for key in dt:
    print(key)
    m[key, dt[key]] = 99
m

dt ={1: (0, 2), 3: (1, 2)}
m = np.arange(15).reshape(5, 3)
for row, cols in dt.items():
    # print(row)
    m[row, cols] = 99
m
a_dict = {'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}
for key, value in a_dict.items():
    print(type(key))
    print(type(value))
    print(key, '->', value)

a_dict = {1: (0, 2), 3: (1, 2)}
for key, value in a_dict.items():
    print(type(key))
    print(type(value))
    print(key, '->', value)


