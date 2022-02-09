#!/bin/python3
import time
import os
import numpy as np

def loading(rand_int=False):
    t0 = time.time()
    if (rand_int): 
        array = np.load("./data/rand_int.npy")
        print ("[RANDOM INT] The time to load a random integer array of %u MB is %0.2f sec" % (int(array.nbytes/1024/1024), time.time() - t0))
    else:
        array = np.load("./data/rand_float.npy")
        print ("[RANDOM FLOAT] The time to load a random float array of %u MB is %0.2f sec" % (int(array.nbytes/1024/1024), time.time() - t0))
    return array    
    
    
 # Sorting not in-place
def sorting(rand_int=False):
    array = loading(rand_int)
    t1 = time.time()
    array_sorted = np.sort(array, kind="quicksort")
    print ("The time to sort NOT in-place: takes quicksort is %0.2f sec" % (time.time() - t1)) 
    t1 = time.time()
    array_sorted = np.sort(array, kind="mergesort")
    print ("The time to sort NOT in-place: takes mergesort is %0.2f sec" % (time.time() - t1))
'''    
    # Sorting in-place
    t1 = time.time()
    array.sort(kind="quicksort")
    print ("The time to sort in-place: takes quicksort is %0.2f sec" % (time.time() - t1))
    
    t1 = time.time()
    array = loading(rand_int)
    array.sort(kind="mergesort")
    print ("The time to sort in-place: takes mergesort is %0.2f sec" % (time.time() - t1))
'''




sorting(rand_int=False)
sorting(rand_int=True)
