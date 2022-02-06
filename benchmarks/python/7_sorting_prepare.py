#!/bin/python3
import time
import os
import numpy as np

MB = 1024*1024
GB = 1024*1024*1024
size = 1*GB

np.random.seed(10)
def prepare(rand_int=False):
    t0 = time.time()
    if (rand_int): 
        array = np.random.random_integers(2**62,size=(1,int(size/8)))
        print ("The time to create a random int of %u MB is %0.2f sec" % (int(array.nbytes/1024/1024), time.time() - t0))
        np.save("./data/rand_int.npy", array)
    else:
        array = np.random.rand(int(size/8)) 
        print ("The time to create a random float array of %u MB is %0.2f sec" % (int(array.nbytes/1024/1024), time.time() - t0))
        np.save("./data/rand_float.npy", array)
    

prepare(rand_int=False)
prepare(rand_int=True)
