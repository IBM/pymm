import time
import os
import numpy as np
import pickle as pkl
import pymm
import pickle
import os.path

# Load the datan
filename = 'data/18_nparray.npy'
t0 = time.time()
array =  np.load(filename)
t = time.time() - t0
print ("Loading the data from the file "+ filename +  " took: %0.2fsec" % t)

pymm_size_mb = 4*int(array.nbytes/1024/1024)

##################################################################
# Deep copy from DRAM to DRAM                                    #
##################################################################
def dram_dram():
    t0 = time.time()
    array1 = np.copy(array)
    print ("[DRAM->DRAM] The time to copy a random float array of %u MB is %0.2f sec" % (int(array.nbytes/1024/1024), time.time() - t0))

##################################################################
# Copy from DRAM to a file located on NVMe with pickle library   #
##################################################################
def dram_nvme_pickle():
    t0 = time.time()
    path = "/mnt/nvme1/tmp/"
    filename = path + 'pickle_write'
    fileObject = open(filename, 'wb')
    pkl.dump(array, fileObject)
    fileObject.close()
    print ("[DRAM->NVMe (pickle)] The time to copy a random float array of %u MB is %0.2f sec" % (int(array.nbytes/1024/1024), time.time() - t0))


##################################################################
# Copy from DRAM to a file located on PM with pickle library     #
##################################################################
def dram_pm_pickle():
    t0 = time.time()
    path = "/mnt/pmem0/tmp/"
    filename = path + 'pickle_write'
    fileObject = open(filename, 'wb')
    pkl.dump(array, fileObject)
    fileObject.close()
    print ("[DRAM->PM (pickle)] The time to copy a random float array of %u MB is %0.2f sec" % (int(array.nbytes/1024/1024), time.time() - t0))


##################################################################
# Use PyMM library benchmark PM[FS-DAX]:                         #
# 1. from DRAM to PM[FS-DAX]                                     #
# 2. from PM[FS-DAX] to PM[FS-DAX]                               #
##################################################################
def pymm_fs_dax():
    s = pymm.shelf('myShelf',size_mb=pymm_size_mb,pmem_path='/mnt/pmem0',force_new=True)

    t0 = time.time()
    s.array = array
    print ("[DRAM->PM[FS-DAX](PyMM)] The time to create a random float array of %u MB is %0.2f sec" % (int(array.nbytes/1024/1024), time.time() - t0))

    t0 = time.time()
    s.array1 = s.array
    print ("[PM->PM[FS-DAX](PyMM)] The time to create a random float array of %u MB is %0.2f sec" % (int(array.nbytes/1024/1024), time.time() - t0))


##################################################################
# Use PyMM library benchmark PM[DEV-DAX]:                        #
# 1. from DRAM to PM[DEV-DAX]                                    #
# 2. from PM[DEV-DAX] to PM[DEV-DAX]                             #
##################################################################

def pymm_dev_dax():
    s1 = pymm.shelf('myShelf1',size_mb=1024*150,pmem_path='/dev/dax1.0',force_new=True)

    t0 = time.time()
    s1.array = array
    print ("[DRAM->PM[DEV-DAX](PyMM) The time to create a random float array of %u MB is %0.2f sec" % (int(array.nbytes/1024/1024), time.time() - t0))
    t0 = time.time()
    s1.array1 = s1.array
    print ("[PM->PM[DEV-DAX](PyMM) The time to create a random float array of %u MB is %0.2f sec" % (int(array.nbytes/1024/1024), time.time() - t0))

dram_dram()
dram_nvme_pickle()
dram_pm_pickle()
pymm_fs_dax()
pymm_dev_dax()
