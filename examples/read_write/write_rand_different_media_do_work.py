import time
import os
import numpy as np
import pickle as pkl
import pymm
import pickle
import os.path
import os
import argparse
import json



M = 1024*128
G = 1024*M
size = 0
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
# https://vmascagn.web.cern.ch/vmascagn/LABO_2020/numpy-memmap_for_ghost_imaging.html
def numpymemmap_func(path, size_GB, array_data, array_pos, persist_at_the_end):
    print (path)
    filename = path + '/numpy_mmap.npy'
    results = { "open_time": 0, "copy_time" : 0, "close_persist_time" : 0}
    print(filename)
    array = np.zeros(int(size_GB) * 1024*1024*128)
    t0 = time.time()
    mmap_array = np.memmap(filename, dtype='float64', mode='w+', shape=(int(size_GB) * 1024*1024*128,))
    mmap_array[:] = array[:]
    mmap_array.flush()
    results["open_time"] = time.time() - t0
    print(np.amax(mmap_array))
    t0_sum = 0; t1_sum = 0
    t0 = time.time()
    for i in range(array_pos.shape[0]):
        mmap_array[array_pos[i]] = array_data[i]
        if (not persist_at_the_end):
            t0_sum += time.time() - t0
            t1 = time.time()
            mmap_array.flush()
            t1_sum += time.time() - t1
            t0 = time.time()
    if (persist_at_the_end):
        t0_sum = time.time() - t0
        t1 = time.time()
        mmap_array.flush()
        t1_sum = time.time() - t1
    results["copy_time"] = t0_sum
    results['close_persist_time'] = t1_sum
    print(np.amax(mmap_array))
    print ("[DRAM->numpymemmap]  open_time  %0.2f sec" %   results['open_time'])
    print ("[DRAM->numpymemmap]  copy_time %0.2f sec" %   results['copy_time'])
    print ("[DRAM->numpymemmap]  persist_time %0.2f sec" % results['close_persist_time'])
    if (remove_write):
        os.remove(filename)
    return results





##################################################################
# Deep copy from DRAM to DRAM                                    #
##################################################################
def dram_dram_func():
    results = { "open_time": 0, "copy_time" : 0, "close_persist_time" : 0}
    t0 = time.time()
    array1 = np.copy(array)
    results["copy_time"] = time.time() - t0
    print ("[DRAM->DRAM] The time to copy a random float array of %u MB is %0.2f sec" % (int(array.nbytes/1024/1024), (results["copy_time"])))
    return results        

##################################################################
# Copy from DRAM to a file located on file systemi               #
##################################################################

def dram_pickle_func(path):
    if not os.path.exists(path):
            os.makedirs(path)
    filename = path + '/pickle_write'
    results = { "open_time": 0, "copy_time" : 0, "close_persist_time" : 0}
    print(filename)
    t0 = time.time()
    fileObject = open(filename, 'wb')
    results["open_time"] = time.time() - t0
    t0 = time.time()
    pkl.dump(array, fileObject, protocol=4)
    results["copy_time"] = time.time() - t0
    print ("[DRAM->pickle]  The time to copy a random float array of %u MB is %0.2f sec" %  (int(array.nbytes/1024/1024), results['copy_time']))
    t0 = time.time()
    fileObject.close()
    results['close_persist_time'] = time.time() - t0
    if (remove_write):
        os.remove(filename)
    return results

##################################################################
# Use PyMM library benchmark PM[FS-DAX]:                         #
# from DRAM to PM[FS-DAX]                                        #
##################################################################
def pymm_fs_dax_func (path, size_GB, array_data, array_pos, persist_at_the_end):
    if not os.path.exists(path):
            os.makedirs(path)
    print (path)
    results = { "open_time": 0, "copy_time" : 0, "close_persist_time" : 0}
    path_split = os.path.split(path)
    t0 = time.time()
    s = pymm.shelf(path_split[1],size_mb=pymm_size_mb,pmem_path=path,force_new=True)
    s.array = np.zeros(int(int(size_GB)*1024*1024*1024/8))
    s.array.persist()
    results["open_time"] = time.time() - t0
    t0_sum = 0; t1_sum = 0
    print(np.amax(s.array))
    t0 = time.time()
    for i in range(array_pos.shape[0]):
        s.array[array_pos[i]] = array_data[i]
        if (not persist_at_the_end):
            t0_sum += time.time() - t0
            t1 = time.time()
            s.array.persist()
            t1_sum += time.time() - t1
            t0 = time.time()
    if (persist_at_the_end):
        t0_sum = time.time() - t0
        t1 = time.time()
        s.array.persist()
        t1_sum = time.time() - t1
    results["copy_time"] = t0_sum
    results['close_persist_time'] = t1_sum
    print(np.amax(s.array))
    print ("[DRAM->numpymemmap]  open_time  %0.2f sec" %   results['open_time'])
    print ("[DRAM->numpymemmap]  copy_time %0.2f sec" %   results['copy_time'])
    print ("[DRAM->numpymemmap]  persist_time %0.2f sec" % results['close_persist_time'])
    if (remove_write):
        os.remove(path + "/" + path_split[1] + ".data")
        os.remove(path + "/" + path_split[1] + ".map")
    return results


##################################################################
# Use PyMM library benchmark PM[DEV-DAX]:                        #
# from DRAM to PM[DEV-DAX]                                       #
##################################################################

def pymm_dev_dax_func(path, array_data, array_pos, persist_at_the_end):
    results = { "open_time": 0, "copy_time" : 0, "close_persist_time" : 0}
    t0 = time.time()
    s1 = pymm.shelf('pymm_dev_dax',size_mb=pymm_size_mb,pmem_path=path,force_new=True)
    results["open_time"] = time.time() - t0

    t0 = time.time()
    s1.array = array
    s1.array[0].persist()
    results["copy_time"] = time.time() - t0
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
    if not os.path.exists(path):
            os.makedirs(path)
    if os.path.exists(path + '/pymm_nvme.data'):
        os.remove(path + '/pymm_nvme.data')
    if os.path.exists(path + '/pymm_nvme.map'):
        os.remove(path + '/pymm_nvme.map')

    results = { "open_time": 0, "copy_time" : 0, "close_persist_time" : 0}
    t0 = time.time()
    s2 = pymm.shelf('pymm_nvme',size_mb=pymm_size_mb,pmem_path=path,force_new=True)
    results["open_time"] = time.time() - t0

    t0 = time.time()
    s2.array = array
    results["copy_time"] = time.time() - t0
    print ("[DRAM->NMVE(PyMM)] The time to copy a random float array of %u MB is %0.2f sec" % (int(array.nbytes/1024/1024), results["copy_time"]))
    t0 = time.time()
    s2.array.persist()
    results["close_persist_time"] = time.time() - t0
#    t0 = time.time()
#    s2.array1 = s2.array
#    print ("NVME->NVME(PyMM)] The time to copy a random float array of %u MB is %0.2f sec" % (int(array.nbytes/1024/1024), time.time() - t0))
    if (remove_write):
        os.remove(path + '/pymm_nvme.data')
        os.remove(path + '/pymm_nvme.map')
    return results

def ndarry_save_func(path):
    filename = path + '/numpy_save'
    results = { "open_time": 0, "copy_time" : 0, "close_persist_time" : 0}
    t0 = time.time()
    np.save(filename, array)
    results["copy_time"] = time.time() - t0
    print ("[ndarry->NVMe (numpy_save)] The time to copy a random float array of %u MB is %0.2f sec" % (int(array.nbytes/1024/1024), results["copy_time"]))
    if (remove_write):
        os.remove(filename + ".npy")
    return results

def run_tests_func(args, array_data, array_pos):
    results = {}
    if (args.dram):
        results['dram'] = dram_dram_func()

    if (args.numpymemmap_nvme):
        results['numpymmap_nvme'] = numpymemmap_func(args.nvme_path, args.size_GB, array_data, array_pos, args.persist_at_the_end)
    
    if (args.numpymemmap_fs_dax0):
        results['numpymmap_fs_dax0'] = numpymemmap_func(args.fs_dax0_path, args.size_GB, array_data, array_pos, args.persist_at_the_end)

    if (args.numpymemmap_fs_dax1):
        results['numpymmap_fs_dax1'] = numpymemmap_func(args.fs_dax1_path, args.size_GB, array_data, array_pos, args.persist_at_the_end)

    if (args.pickle_nvme):
        results['pickle_dram_nvme'] = dram_pickle_func(args.nvme_path)
    
    if (args.pickle_fs_dax0): 
        results['pickle_dram_fs_dax0'] = dram_pickle_func(args.fs_dax0_path)
    
    if (args.pickle_fs_dax1): 
        results['pickle_dram_fs_dax1'] = dram_pickle_func(args.fs_dax1_path)
    
    if (args.pymm_fs_dax0): 
        results['pymm_dram_fs_dax0'] = pymm_fs_dax_func(args.fs_dax0_path, args.size_GB, array_data, array_pos, args.persist_at_the_end)
    
    if (args.pymm_fs_dax1): 
        results['pymm_dram_fs_dax1'] = pymm_fs_dax_func(args.fs_dax1_path, args.size_GB, array_data, array_pos, args.persist_at_the_end)
    
    if (args.pymm_dev_dax0):
        results['pymm_dram_dev_dax0'] = pymm_fs_dax_func(args.nvme_path, args.size_GB, array_data, array_pos, args.persist_at_the_end)
    
    if (args.pymm_dev_dax1):
        results['pymm_dram_dev_dax1'] = pymm_fs_dax_func(args.dev_dax1_path,args.size_GB, array_data, array_pos, args.persist_at_the_end)
    
    if (args.pymm_nvme):
        results['pymm_dram_nvme'] = pymm_fs_dax_func(args.nvme_path,args.size_GB, array_data, array_pos, args.persist_at_the_end)
    
    if (args.numpy_save_nvme):
        results['ndarray_save_dram_nvme'] = ndarry_save_func(args.nvme_path)
    
    if (args.numpy_save_fs_dax0): 
        results['ndarray_save_dram_fs_dax0'] = ndarry_save_func(args.fs_dax0_path)
    
    if (args.numpy_save_fs_dax1): 
        results['ndarray_save_dram_fs_dax1'] = ndarry_save_func(args.fs_dax1_path)
    
    return results


def set_args(args):
    if (args.test_all):
        args.dram = True    
        args.numpymemmap_nvme = True    
        args.numpymemmap_fs_dax0 = True    
        args.numpymemmap_fs_dax1 = True    
        args.pickle_fs_dax0  = True    
        args.pickle_fs_dax1  = True    
        args.pymm_fs_dax0  = True    
        args.pymm_fs_dax1  = True    
        args.pymm_dev_dax0 = True    
        args.pymm_dev_dax1 = True    
        args.pymm_nvme = True    
        args.numpy_save_nvme = True    
        args.numpy_save_fs_dax0  = True    
        args.numpy_save_fs_dax1  = True 

    if (args.numa_local):
        args.dram = True    
        args.pickle_nvme = True    
        args.numpymemmap_nvme = True    
        args.numpymemmap_fs_dax0 = True    
        args.pickle_fs_dax0  = True    
        args.pymm_fs_dax0  = True    
        args.pymm_dev_dax0 = True    
        args.pymm_nvme = True    
        args.numpy_save_nvme = True    
        args.numpy_save_fs_dax0  = True    
    

def create_rand_data(num):
    ### Create the data ###
    t0 = time.time()
    array = np.random.rand(num)
    print ("Creating the data of shape[0]=%u took: %0.2fsec" % (array.shape[0], time.time() - t0))
    print (array)
    return array

def create_rand_pos(num, end_pos):
    ### Create the data ###
    t0 = time.time()
    array = np.random.randint(end_pos, size=(int(num),))
    print ("Creating the data of size shape[0]=%u took: %0.2fsec" % (array.shape[0], time.time() - t0))
    print (array)
    return array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("size_GB", type=str, help="The data size in GB")
    parser.add_argument("output_dir", type=str, help="where to store the results")
    parser.add_argument("--persist_at_the_end", action="store_true", default=False, help="Flush just at the end of the write")
    parser.add_argument("--remove_write", action="store_true", default=False, help="not remove the data after write")
    parser.add_argument("--test_all", action="store_true", default=False, help="run all the different options")
    parser.add_argument("--numa_local", action="store_true", default=False, help="run all the different options")
    parser.add_argument("--dram", action="store_true", default=False, help="deep copy the array to the DRAM")    
    parser.add_argument("--numpymemmap_nvme", action="store_true", default=False, help="save the array as numpy_mmap")    
    parser.add_argument("--numpymemmap_fs_dax0", action="store_true", default=False, help="save the array as numpy_mmap")    
    parser.add_argument("--numpymemmap_fs_dax1", action="store_true", default=False, help="save the array as numpy_mmap")    
    parser.add_argument("--pickle_nvme", action="store_true", default=False, help="save the array as pickle to NVMe")    
    parser.add_argument("--pickle_fs_dax0", action="store_true", default=False, help="save the array as pickle to socket 0")    
    parser.add_argument("--pickle_fs_dax1", action="store_true", default=False, help="save the array as pickle to socket 1")    
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
    
    
    num_write_items_8B = 1000*1 
    num_write_items_64B = 1000000
    num_write_items_4K = 100
    num_write_items_2M = 10

    array_data = create_rand_data (num_write_items_8B)
    array_pos = create_rand_pos(num_write_items_8B, int(args.size_GB)*1024*1024*1024/8)

#    clear_caches()
    set_args(args)
    results = run_tests_func(args, array_data, array_pos)
    with open(args.output_dir + "_results." + str(args.size_GB) + "GB.json" ,'w') as fp:
            json.dump(results, fp)

if __name__ == "__main__":
            main()

