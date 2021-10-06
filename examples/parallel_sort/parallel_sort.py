#!/usr/bin/python3
import pymm

from parallelSort import numpyParallelSort
import numpy as np 

pymm.enable_transient_memory()

shelf = pymm.shelf('sortExample',size_mb=1000,backend='hstore-cc',pmem_path='/mnt/pmem0')

print(shelf.items)

if 'sortedList' in shelf.items:
    print('List already sorted ...')
    print(shelf.sortedList[:10])
    exit()
       
count = 100000000 # fit in 1GB

shelf.sortedList = np.random.randint(10**12,size=count)

numpyParallelSort(shelf.sortedList)
print(shelf.sortedList[:10])
