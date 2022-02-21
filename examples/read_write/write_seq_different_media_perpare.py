import time
import os
import numpy as np
import pickle as pkl
import pymm
import pickle
import os.path
import time
'''
Author: Moshik Hershcovitch <moshikh@il.ibm.com> 2022
License: Apache, Version 2.0
'''

M = 1024*128
G = 1024*M

### Create the data ###
size = 1*G
#size = 100*M
np.random.seed(2)
t0 = time.time()
array = np.random.rand(size)
t = time.time() - t0
print ("Creating the data of size %uMB took: %0.2fsec" % (int(array.nbytes/1024/1024), time.time() - t0))


filename = 'data/ndarray.npy'
print ("write to " + filename)
np.save(filename, array)
