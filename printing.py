# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:45:46 2020

@author: donbo
"""
import sys
import time

# best so far
for i in range(300):
    time.sleep(0.1)  # if delay is too short it messes up
    #print('\r', end='\r')
    val = str(i).zfill(2)
    print('\rProgress: ', val, end='\r', flush=True)

#    print('Downloading File FooFile.txt [%d%%]'%i, end="\r")

def f():
     print('xy', end='')
     print('\bz')

f()

curr = 12
total = 21
frac = curr/total
print('[{:>7.2%}]'.format(frac))

# The formatting code is the "{:>7.2%}", where ">" means that the string is be right aligned with spaces added to the front, "7" is the fixed length of the string + padded spaces (3 whole number digits + a point + 2 decimal digits + a percentage sign) ".2" is the number of decimal places to round the percentage to, and "%" means that the fraction should be expressed as a percentage.

for i in range(300):
    time.sleep(0.1)  # if delay is too short it messes up
    #print('\r', end='\r')
    val = str(i).zfill(2)
    val = l(10/3)
    print('\rProgress: ', val, end='\r', flush=True)

def progbar(curr, total, full_progbar):
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', '#'*filled_progbar + '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')
    sys.stdout.flush()

n = 1000
for i in range(n+1):
    progbar(i, n, 20)
print()

numbers  = [1e9, 23.2300, 0.1233, 1.0000, 1.0, 4.2230, 9887.2000]
for x in numbers:
    print('{:<8.4g}'.format(round(x, 2)))


print('{:8.4e}'.format(1e6))
print('{:8.4e}'.format(123.456789))

print('{:8.4g}'.format(1e6))
print('{:8.4g}'.format(123.456789))

print('{:8.6g}'.format(1e6))
print('{:8.6g}'.format(123.456789))
print('{:<8.6g}'.format(123.456789))

print('{:12.6g}'.format(1e6))
print('{:12.6g}'.format(123.456789))

print('{:.6g}'.format(1e6))
print('{:4g}'.format(123.456789))

print('{:8.4f}'.format(1e6))
print('{:8.4f}'.format(123.456789))

print('{:8.4G}'.format(1e6))
print('{:8.4G}'.format(123.456789))


# this is good
for i in range(1000):
    sleep(0.035)  # appears to need about .03 seconds
    val = str(i).zfill(3)
    print('\r', end='')
    print('text etc, area:    ', end='')
    print('\b\b\b\b', val, end='', flush=True)
    # sys.stdout.flush()


# best yet
print('text etc, area:    ', end='')
for i in range(1000):
    sleep(0.04)  # appears to need about .04 seconds
    val = str(i).zfill(3)
    print('\b\b\b\b', val, end='', flush=True)


import sys
import time
for i in range(10):
    print("Loading" + "." * i)
    sys.stdout.write("\033[F") # Cursor up one line
    time.sleep(1)
Also sometimes useful (for example if you print something shorter than before):

sys.stdout.write("\033[K") # Clear to the end of line




def process(data):
    size_str = str(data)
    sys.stdout.write('Progress: %s\r' % size_str)
    sys.stdout.flush()

for i in range(30):
    time.sleep(0.1)
    process(i)


while True:
    print(time.ctime(), end="\r", flush=True)
    time.sleep(1)

import os
import time
while True:
    print(time.ctime(), end ='\r', flush=True)
    time.sleep(1)
    os.system('cls')


from tqdm import tqdm
for i in tqdm(range(100)):
    time.sleep(1)

from time import sleep
text = ""
for char in tqdm(["a", "b", "c", "d", "e", "f", "g"]):
    sleep(0.75)
    text = text + char

print("line1", end = "\r")
print("line2")

for i in range(100):
    time.sleep(0.1)
    print('Downloading File FooFile.txt [%d%%]\r'%i, end="")


for i in range(20):
    time.sleep(0.1)
    sys.stdout.write("Download progress: %d%%   \r" % (i) )
    sys.stdout.flush()

    print(os.path.getsize(file_name)/1024+'KB / '+size+' KB downloaded!', end='\r')