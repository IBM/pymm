import time
import os
import numpy as np
import pickle as pkl
import pymm
import os.path
import os
import argparse
import json

filter=0.9999
M = 1024*128
G = 1024*M
size = 0
array = np.array([3])
np.random.seed(2)
array_size_mb = 0
pymm_size_mb = 340*1024
pymm_shelf_factor = 3
remove_write = False



def clear_caches ():
    os.system("free -m; sync; echo 3 > /proc/sys/vm/drop_caches; swapoff -a && swapon -a;")
    os.system("free -m;")


##################################################################
# Copy from DRAM to a file located on file system               #
##################################################################

def numpymemmap_func(path, size_GB):
    print (path)
    filename = path + '/numpy_save.npy'
    results = { "open_time": 0, "copy_time" : 0, "close_persist_time" : 0}
    print(filename)
    t = np.arange(256)    
    t0 = time.time()
    array = np.memmap(filename, dtype='float64', mode='r', shape=(int(size_GB) * 1024*1024*128,), offset=64)
    results["open_time"] = time.time() - t0
    t0 = time.time()
    time.sleep(0)
    a = array[array>filter]
    time.sleep(0)
    results["copy_time"] = time.time() - t0
    print (a)
    print ("[DRAM->numpymemmap]  The time to copy a random float array of %u MB is %0.2f sec" %  (int(array.nbytes/1024/1024), results['open_time']))
    print ("[DRAM->numpymemmap]  The time to copy a random float array of %u MB is %0.2f sec" %  (int(array.nbytes/1024/1024), results['copy_time']))
    t0 = time.time()
    results['close_persist_time'] = time.time() - t0
    if (remove_write):
        os.remove(filename)
    return results

##################################################################
# Use PyMM library benchmark PM[FS-DAX]:                         #
# from DRAM to PM[FS-DAX]                                        #
##################################################################
def pymm_fs_dax_func(path):
    if not os.path.exists(path):
            os.makedirs(path)
    print (path)
    results = { "open_time": 0, "copy_time" : 0, "close_persist_time" : 0}
    path_split = os.path.split(path)
    t0 = time.time()
    s = pymm.shelf("tmp0",size_mb=200000,pmem_path="/mnt/pmem0/tmp0/")
#    s = pymm.shelf(path_split[1],size_mb=pymm_size_mb,pmem_path=path)
    print(s.get_item_names())
    results["open_time"] = time.time() - t0
    t0 = time.time()
    a = s.array[s.array>filter]
    results["copy_time"] = time.time() - t0
    print(a)
    print(type(array))
    print ("[DRAM->PM[FS-DAX](PyMM)]  The time to copy a random float array of %u MB is %0.2f sec" %  (int(array.nbytes/1024/1024), results["copy_time"]))
    t0 = time.time()
#    s.array.persist()
#    results["close_persist_time"] = time.time() - t0
#    print ("[PM->PM[FS-DAX(PyMM)]  persist %0.2f sec" %  (results["close_persist_time"]))
#    t0 = time.time()
#    s.array1 = s.array
#    print ("[PM->PM[FS-DAX](PyMM)]   the time to copy a random float array of %u MB is %0.2f sec" %  (int(array.nbytes/1024/1024), time.time() - t0))
#    t0 = time.time()
#    s.persist()
#    print ("[PM->PM[DEV-DAX(PyMM)]  persist %0.2f sec" %  (time.time() - t0))
    return results


##################################################################
# Use PyMM library benchmark PM[DEV-DAX]:                        #
# from DRAM to PM[DEV-DAX]                                       #
##################################################################

def pymm_dev_dax_func(path):
    results = { "open_time": 0, "copy_time" : 0, "close_persist_time" : 0}
    t0 = time.time()
    s1 = pymm.shelf('pymm_dev_dax',size_mb=pymm_size_mb,pmem_path=path)
    results["open_time"] = time.time() - t0

    t0 = time.time()
    a = s1.array[s1.array>filter]
    results["copy_time"] = time.time() - t0
    print (a)
    print ("[DRAM->PM[DEV-DAX(PyMM)]  The time to copy a random float array of %u MB is %0.2f sec" %  (int(array.nbytes/1024/1024), results["copy_time"]))
    t0 = time.time()
    results["close_persist_time"] = time.time() - t0
    print ("[PM->PM[DEV-DAX(PyMM)]  persist %0.2f sec" %  (results["close_persist_time"]))

#    s1.array1 = s1.array
#    print ("[PM->PM[DEV-DAX(PyMM)]  The time to copy a random float array of %u MB is %0.2f sec" %  (int(array.nbytes/1024/1024), time.time() - t0))
#    t0 = time.time()
#    s1.persist()
#    print ("[PM->PM[DEV-DAX(PyMM)]  persist %0.2f sec" %  (time.time() - t0))
    return results

##################################################################
# Use PyMM library benchmark PM[FS-DAX]:                         #
# from DRAM to NVMe                                              #
##################################################################
def pymm_nvme_func(path):
    print(path)
    results = { "open_time": 0, "copy_time" : 0, "close_persist_time" : 0}
    t0 = time.time()
    s2 = pymm.shelf('pymm_nvme',size_mb=pymm_size_mb,pmem_path=path)
    results["open_time"] = time.time() - t0

    t0 = time.time()
#    a = s2.array[s2.array>filter]
    a = pymm.np.amax(s2.array)
    results["copy_time"] = time.time() - t0
    print (a)
    print ("[DRAM->NMVE(PyMM)] The time to copy a random float array of %u MB is %0.2f sec" % (int(array.nbytes/1024/1024), results["copy_time"]))
    t0 = time.time()
#    s2.array.persist()
    results["close_persist_time"] = time.time() - t0
#    t0 = time.time()
#    s2.array1 = s2.array
#    print ("NVME->NVME(PyMM)] The time to copy a random float array of %u MB is %0.2f sec" % (int(array.nbytes/1024/1024), time.time() - t0))
    if (remove_write):
        os.remove(path + '/pymm_nvme.data')
        os.remove(path + '/pymm_nvme.map')
    return results

def ndarry_save_func(path):
    print (path)
    filename = path + '/numpy_save'
    results = { "open_time": 0, "copy_time" : 0, "close_persist_time" : 0}
    t0 = time.time()
    array = np.load(filename + ".npy")
    results["open_time"] = time.time() - t0
    t0 = time.time()
    a = array[array>filter]
    results["copy_time"] = time.time() - t0
    print (a)
    print ("[ndarry->NVMe (numpy_save)] The time to copy a random float array of %u MB is copy %0.2f " % (int(array.nbytes/1024/1024), results["open_time"]))
    print ("[ndarry->NVMe (numpy_save)] The time to copy a random float array of %u MB is max %0.2f " % (int(array.nbytes/1024/1024), results["copy_time"]))
    if (remove_write):
        os.remove(filename + ".npy")
    return results

def run_tests_func(args):
    clear_caches()
    results = {}
    if (args.numpymemmap_nvme):
        results['numpymemmap_dram_nvme'] = numpymemmap_func(args.nvme_path, args.size_GB)
        clear_caches()
    
    if (args.numpymemmap_fs_dax0): 
        results['numpymemmap_dram_fs_dax0'] = numpymemmap_func(args.fs_dax0_path, args.size_GB )
        clear_caches()
    
    if (args.numpymemmap_fs_dax1): 
        results['numpymemmap_dram_fs_dax1'] = numpymemmap_func(args.fs_dax1_path, args.size_GB)
        clear_caches()
    
    if (args.pymm_fs_dax0): 
        results['pymm_dram_fs_dax0'] = pymm_fs_dax_func(args.fs_dax0_path)
        clear_caches()
    
    if (args.pymm_fs_dax1): 
        results['pymm_dram_fs_dax1'] = pymm_fs_dax_func(args.fs_dax1_path)
        clear_caches()
    
    if (args.pymm_dev_dax0):
        results['pymm_dram_dev_dax0'] = pymm_dev_dax_func(args.dev_dax0_path)
        clear_caches()
    
    if (args.pymm_dev_dax1):
        results['pymm_dram_dev_dax1'] = pymm_dev_dax_func(args.dev_dax1_path)
        clear_caches()
    
    if (args.pymm_nvme):
        results['pymm_dram_nvme'] = pymm_nvme_func(args.nvme_path)
        clear_caches()
    
    if (args.numpy_save_nvme):
        results['ndarray_save_dram_nvme'] = ndarry_save_func(args.nvme_path)
        clear_caches()
    
    if (args.numpy_save_fs_dax0): 
        results['ndarray_save_dram_fs_dax0'] = ndarry_save_func(args.fs_dax0_path)
        clear_caches()
    
    if (args.numpy_save_fs_dax1): 
        results['ndarray_save_dram_fs_dax1'] = ndarry_save_func(args.fs_dax1_path)
        clear_caches()
    
    return results


def set_args(args):
    if (args.test_all):
        args.numpymemmap_nvme = True    
        args.numpymemmap_fs_dax0  = True    
        args.numpymemmap_fs_dax1  = True    
        args.pymm_fs_dax0  = True    
        args.pymm_fs_dax1  = True    
        args.pymm_dev_dax0 = True    
        args.pymm_dev_dax1 = True    
        args.pymm_nvme = True    
        args.numpy_save_nvme = True    
        args.numpy_save_fs_dax0  = True    
        args.numpy_save_fs_dax1  = True 

    if (args.numa_local):
        args.numpymemmap_nvme = True    
        args.numpymemmap_fs_dax0  = True    
        args.pymm_fs_dax0  = True    
        args.pymm_dev_dax0 = True    
        args.pymm_nvme = True    
        args.numpy_save_nvme = True    
        args.numpy_save_fs_dax0  = True    
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("size_GB", type=str, help="The data size in GB")
    parser.add_argument("--remove_write", type=str, help="removing the data after write")
    parser.add_argument("output_dir", type=str, help="where to store the results")
    parser.add_argument("--test_all", action="store_true", default=False, help="run all the different options")
    parser.add_argument("--numa_local", action="store_true", default=False, help="run all the different options")
    parser.add_argument("--numpymemmap_nvme", action="store_true", default=False, help="save the array as numpymemmap to NVMe")    
    parser.add_argument("--numpymemmap_fs_dax0", action="store_true", default=False, help="save the array as numpymemmap to socket 0")    
    parser.add_argument("--numpymemmap_fs_dax1", action="store_true", default=False, help="save the array as numpymemmap to socket 1")    
    parser.add_argument("--pymm_fs_dax0", action="store_true", default=False, help="save the array to PyMM shelf on fs_daxi [socket 0]")    
    parser.add_argument("--pymm_fs_dax1", action="store_true", default=False, help="save the array to PyMM shelf on fs_dax [socket 1]")    
    parser.add_argument("--pymm_dev_dax0", action="store_true", default=False, help="save the array to PyMM shelf on dev_dax [socket 0]")    
    parser.add_argument("--pymm_dev_dax1", action="store_true", default=False, help="save the array to PyMM shelf on dev_dax [socket 1]")    
    parser.add_argument("--pymm_nvme", action="store_true", default=False, help="save the array to PyMM shelf on NVMe")    
    parser.add_argument("--numpy_save_nvme", action="store_true", default=False, help="save with numpy.save to NVMe")    
    parser.add_argument("--numpy_save_fs_dax0", action="store_true", default=False, help="save with numpy.save to fs_dax [socket 0]")    
    parser.add_argument("--numpy_save_fs_dax1", action="store_true", default=False, help="save with numpy.save to fs_dax [socket 1]")   
    parser.add_argument("--nvme_path", action = 'store', default="/mnt/nvme0/tmp", type = str, help ="The path to the directory on the NVMe --- the default is /mnt/nvme0/tmp")  
    parser.add_argument("--fs_dax0_path", action = 'store', default="/mnt/pmem0/tmp0", type = str, help ="The path to the directory on fs_dax [socket 0] --- the default is /mnt/pmem0/tmp0")  
    parser.add_argument("--fs_dax1_path", action = 'store', default="/mnt/pmem1/tmp1", type = str, help ="The path to the directory on fs_dax [socket 1] --- the default is /mnt/pmem1/tmp1")  
    parser.add_argument("--dev_dax0_path", action = 'store', default="/dev/dax0.1", type = str, help ="The path to dev_dax [socket 0] --- the default is /dev/dax0.1")  
    parser.add_argument("--dev_dax1_path", action = 'store', default="/dev/dax1.1", type = str, help ="The path to dev_dax [socket 1] --- the default is /dev/dax1.1")  

    args = parser.parse_args()
    set_args(args)
    results = run_tests_func(args)
    with open(args.output_dir + "_results." + str(args.size_GB) + "GB.json" ,'w') as fp:
            json.dump(results, fp)

if __name__ == "__main__":
            main()

