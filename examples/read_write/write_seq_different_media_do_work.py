import time
import os
import numpy as np
import pickle as pkl
import pymm
import pickle
import os.path
import os

nvme_path = "/mnt/nvme0/tmp/"
fs_dax0_path = "/mnt/pmem0/tmp0"
fs_dax1_path = "/mnt/pmem1/tmp1"
dev_dax0_path = "/dev/dax0.1"
dev_dax1_path = "/dev/dax1.1"

test_all = True

dram = False
pickle_nvme = False
pickle_fs_dax0  = False
pickle_fs_dax1  = False
pymm_fs_dax0  = False
pymm_fs_dax1  = False
pymm_dev_dax0 = False
pymm_dev_dax1 = False
pymm_nvme = False
numpy_save_nvme = False
numpy_save_fs_dax0  = False
numpy_save_fs_dax1  = False

if (test_all):
    dram = True    
    pickle_nvme = True    
    pickle_fs_dax0  = True    
    pickle_fs_dax1  = True    
    pymm_fs_dax0  = True    
    pymm_fs_dax1  = True    
    pymm_dev_dax0 = True    
    pymm_dev_dax1 = True    
    pymm_nvme = True    
    numpy_save_nvme = True    
    numpy_save_fs_dax0  = True    
    numpy_save_fs_dax1  = True    
    


# Load the data
filename = 'data/ndarray.npy'
t0 = time.time()
array =  np.load(filename)
t = time.time() - t0
print ("Loading the data from the file "+ filename +  " took: %0.2fsec" % t)

array_size_mb = array.nbytes/1024/1024
pymm_size_mb = 4*int(array_size_mb)
print(pymm_size_mb)

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

def dram_pickle(path):
    if not os.path.exists(path):
            os.makedirs(path)
    filename = path + 'pickle_write'
    print(filename)
    fileObject = open(filename, 'wb')
    t0 = time.time()
    pkl.dump(array, fileObject, protocol=4)
    print ("[DRAM->pickle]  The time to copy a random float array of %u MB is %0.2f sec" %  (int(array.nbytes/1024/1024), time.time() - t0))
    fileObject.close()


def dram_pickle_4G(path):
    if not os.path.exists(path):
            os.makedirs(path)
    filename = path + 'pickle_write'
    file_path = "pkl.pkl"
    n_bytes = 2**31
    max_bytes = 2**31 - 1
    data = bytearray(n_bytes)

    t0 = time.time()
    fileObject = open(filename, 'wb')
     ## write
    bytes_out = pkl.dumps(data)
    with open(filename, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])

    print ("[DRAM->NVMe (pickle)] The time to copy a random float array of %u MB is %0.2f sec" % (int(array.nbytes/1024/1024), time.time() - t0))


    ## read
#    bytes_in = bytearray(0)
#    input_size = os.path.getsize(filename)
#    with open(filename, 'rb') as f_in:
#        for _ in range(0, input_size, max_bytes):
#            bytes_in += f_in.read(max_bytes)
#    data2 = pkl.loads(bytes_in)



##################################################################
# Copy from DRAM to a file located on PM with pickle library     #
##################################################################
def dram_pickle_fs_dax(path):
    if not os.path.exists(path):
            os.makedirs(path)
    filename = path + 'pickle_write'
    print(filename)
    fileObject = open(filename, 'wb')
    t0 = time.time()
    pkl.dump(array, fileObject)
    print ("[DRAM->PM (pickle)]  The time to copy a random float array of %u MB is %0.2f sec" %  (int(array.nbytes/1024/1024), time.time() - t0))
    fileObject.close()


##################################################################
# Use PyMM library benchmark PM[FS-DAX]:                         #
# 1. from DRAM to PM[FS-DAX]                                     #
# 2. from PM[FS-DAX] to PM[FS-DAX]                               #
##################################################################
def pymm_fs_dax(path):
    if not os.path.exists(path):
            os.makedirs(path)
    print (path)
    path_split = os.path.split(path)
    s = pymm.shelf(path_split[1],size_mb=pymm_size_mb,pmem_path=path,force_new=True)

    t0 = time.time()
    s.array = array
    print ("[DRAM->PM[FS-DAX](PyMM)]  The time to copy a random float array of %u MB is %0.2f sec" %  (int(array.nbytes/1024/1024), time.time() - t0))
#    s.persist()
#    print ("[PM->PM[DEV-DAX(PyMM)]  persist %0.2f sec" %  (time.time() - t0))

#    t0 = time.time()
#    s.array1 = s.array
#    print ("[PM->PM[FS-DAX](PyMM)]   the time to copy a random float array of %u MB is %0.2f sec" %  (int(array.nbytes/1024/1024), time.time() - t0))
#    t0 = time.time()
#    s.persist()
#    print ("[PM->PM[DEV-DAX(PyMM)]  persist %0.2f sec" %  (time.time() - t0))
    s.close 


##################################################################
# Use PyMM library benchmark PM[DEV-DAX]:                        #
# 1. from DRAM to PM[DEV-DAX]                                    #
# 2. from PM[DEV-DAX] to PM[DEV-DAX]                             #
##################################################################

def pymm_dev_dax(path):
    s1 = pymm.shelf('pymm_dev_dax',size_mb=pymm_size_mb,pmem_path=path,force_new=True)

    t0 = time.time()
    s1.array = array
    print ("[DRAM->PM[DEV-DAX(PyMM)]  The time to copy a random float array of %u MB is %0.2f sec" %  (int(array.nbytes/1024/1024), time.time() - t0))
    t0 = time.time()
#    s1.array1 = s1.array
#    print ("[PM->PM[DEV-DAX(PyMM)]  The time to copy a random float array of %u MB is %0.2f sec" %  (int(array.nbytes/1024/1024), time.time() - t0))
#    t0 = time.time()
#    s1.persist()
#    print ("[PM->PM[DEV-DAX(PyMM)]  persist %0.2f sec" %  (time.time() - t0))
    s1.close 

##################################################################
# Use PyMM library benchmark PM[FS-DAX]:                         #
# 1. from DRAM to PM[FS-DAX]                                     #
# 2. from PM[FS-DAX] to PM[FS-DAX]                               #
##################################################################
def pymm_nvme(path):
    if not os.path.exists(path):
            os.makedirs(path)
    if os.path.exists(nvme_path + '/pymm_nvme.data'):
        os.remove(nvme_path + '/pymm_nvme.data')
    if os.path.exists(nvme_path + '/pymm_nvme.map'):
        os.remove(nvme_path + '/pymm_nvme.map')
    s2 = pymm.shelf('pymm_nvme',size_mb=pymm_size_mb,pmem_path=path,force_new=True)

    t0 = time.time()
    s2.array = array
    print ("[DRAM->NMVE(PyMM)] The time to copy a random float array of %u MB is %0.2f sec" % (int(array.nbytes/1024/1024), time.time() - t0))

#    t0 = time.time()
#    s2.array1 = s2.array
#    print ("NVME->NVME(PyMM)] The time to copy a random float array of %u MB is %0.2f sec" % (int(array.nbytes/1024/1024), time.time() - t0))
    s2.close


def ndarry_save(path):
    filename = path + 'numpy_save'
    t0 = time.time()
    np.save(filename, array)
    print ("[ndarry->NVMe (numpy_save)] The time to copy a random float array of %u MB is %0.2f sec" % (int(array.nbytes/1024/1024), time.time() - t0))


if (dram):
    dram_dram()

if (pickle_nvme):
    dram_pickle(nvme_path)

if (pickle_fs_dax0): 
    dram_pickle(fs_dax0_path)

if (pickle_fs_dax1): 
    dram_pickle(fs_dax1_path)

if (pymm_fs_dax0): 
    pymm_fs_dax(fs_dax0_path)

if (pymm_fs_dax1): 
    pymm_fs_dax(fs_dax1_path)

if(pymm_dev_dax0):
    pymm_dev_dax(dev_dax0_path)

if(pymm_dev_dax1):
    pymm_dev_dax(dev_dax1_path)

if(pymm_nvme):
    pymm_nvme(nvme_path)

if(numpy_save_nvme):
    ndarry_save(nvme_path)

if(numpy_save_fs_dax0): 
    ndarry_save(fs_dax0_path)

if(numpy_save_fs_dax1): 
    ndarry_save(fs_dax1_path)

# save (func_name, path, is_input_DRAM, time_create, copy_time, time_to_persist, time_close)  
