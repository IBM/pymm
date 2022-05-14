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
pymm_size_mb = 320*1000
pymm_shelf_factor = 3
num_reads = 1000000
read_size = 8
is_adding_number = 1


def clear_caches ():
    os.system("free -m; sync; echo 3 > /proc/sys/vm/drop_caches; swapoff -a && swapon -a;")
    os.system("free -m;")


### DRAM ##
def dram_dram_func(path, size_GB, array_pos):
    results = { "open_time": 0, "copy_time" : 0, "close_persist_time" : 0}
    print (path)
    t0 = time.time()
    filename = path + '/numpy_save.npy'
    array_data = np.load(filename)
    results["open_time"] = time.time() - t0
    itemsize = int(read_size/8)
    t0 = time.time()
    for i in range(array_pos.shape[0]):
        array_idx = array_pos[i]
        sum_ = array_data[array_idx:array_idx+itemsize] + 1
    results["copy_time"] = time.time() - t0


    print(sum_)
    print ("[DRAM]-> DRAM open_time  %0.2f sec" %   results['open_time'])
    print ("[DRAM]-> DRAM copy_time %0.2f sec" %   results['copy_time'])
    print ("[DRAM]-> DRAM persist_time %0.2f sec" % results['close_persist_time'])
    return results




##################################################################
# Copy from DRAM to a file located on file system               #
##################################################################
# https://vmascagn.web.cern.ch/vmascagn/LABO_2020/numpy-memmap_for_ghost_imaging.html
def numpymemmap_func(path, size_GB, array_pos):
    # Not implementent 
    print (path)
    filename = path + '/numpy_save.npy'
    results = { "open_time": 0, "copy_time" : 0, "close_persist_time" : 0}
    print(filename)
    t0 = time.time()
    array = np.memmap(filename, dtype='float64', mode='r', shape=(int(size_GB) * 1024*1024*128,), offset=64)
    results["open_time"] = time.time() - t0
    itemsize = int(read_size/8)
    t0 = time.time()
    for i in range(array_pos.shape[0]):
        array_idx = array_pos[i]
        sum_ = array[array_idx:array_idx+itemsize] + 1
    
    results["copy_time"] = time.time() - t0 
    print ("[DRAM->pymm]  open_time  %0.2f sec" %   results['open_time'])
    print ("[DRAM->pymm]  copy_time %0.2f sec" %   results['copy_time'])
    print ("[DRAM->pymm]  persist_time %0.2f sec" % results['close_persist_time'])
    return results


    t0 = time.time()
    return results


##################################################################
# Use PyMM library benchmark PM[FS-DAX]:                         #
# from DRAM to PM[FS-DAX]                                        #
##################################################################
def pymm_func (path, size_GB, array_pos, is_dev_dax):
    print (path)
    results = { "open_time": 0, "copy_time" : 0, "close_persist_time" : 0}
    name = "pymm_dev_dax"
    if (is_dev_dax == 0): 
        path_split = os.path.split(path)
        name = path_split[1]
    
    t0 = time.time()
    print (name)
    s = pymm.shelf(name,size_mb=pymm_size_mb,pmem_path=path)
    results["open_time"] = time.time() - t0
    itemsize = int(read_size/8)
    t0 = time.time()
    for i in range(array_pos.shape[0]):
        array_idx = array_pos[i]
        sum_ = s.array[array_idx:array_idx+itemsize] + 1

    results["copy_time"] = time.time() - t0
    print ("[DRAM->pymm]  open_time  %0.2f sec" %   results['open_time'])
    print ("[DRAM->pymm]  copy_time %0.2f sec" %   results['copy_time'])
    print ("[DRAM->pymm]  persist_time %0.2f sec" % results['close_persist_time'])
    return results


def ndarry_save_func(path, size_GB, array_pos):
    print (path)
    filename = path + '/numpy_save.npy'
    results = { "open_time": 0, "copy_time" : 0, "close_persist_time" : 0}
    print(filename)
    t0 = time.time()
    array = np.load(filename)
    results["open_time"] = time.time() - t0
    itemsize = int(read_size/8)
    t0 = time.time()
    for i in range(array_pos.shape[0]):
        array_idx = array_pos[i]
        sum_ = array[array_idx:array_idx+itemsize] + 1
    
    results["copy_time"] = time.time() - t0 
    print ("[DRAM->pymm]  open_time  %0.2f sec" %   results['open_time'])
    print ("[DRAM->pymm]  copy_time %0.2f sec" %   results['copy_time'])
    print ("[DRAM->pymm]  persist_time %0.2f sec" % results['close_persist_time'])
    return results




def run_tests_func(args, array_pos):
    results = {}

    if (args.dram):
        clear_caches()
        results['dram'] = dram_dram_func(args.nvme_path, args.size_GB,  array_pos)

    if (args.numpymemmap_nvme):
        clear_caches()
        results['numpymmap_nvme'] = numpymemmap_func(args.nvme_path, args.size_GB,  array_pos)
    
    if (args.numpymemmap_fs_dax0):
        clear_caches()
        results['numpymmap_fs_dax0'] = numpymemmap_func(args.fs_dax0_path, args.size_GB, array_pos)

    if (args.numpymemmap_fs_dax1):
        clear_caches()
        results['numpymmap_fs_dax1'] = numpymemmap_func(args.fs_dax1_path, args.size_GB, array_pos)

    if (args.pymm_fs_dax0): 
        clear_caches()
        results['pymm_dram_fs_dax0'] = pymm_func(args.fs_dax0_path, args.size_GB,  array_pos, is_dev_dax=0)
    
    if (args.pymm_fs_dax1): 
        clear_caches()
        results['pymm_dram_fs_dax1'] = pymm_func(args.fs_dax1_path, args.size_GB,  array_pos, is_dev_dax=0)
    
    if (args.pymm_dev_dax0):
        clear_caches()
        results['pymm_dram_dev_dax0'] = pymm_func(args.dev_dax0_path, args.size_GB,  array_pos, is_dev_dax=1)
    
    if (args.pymm_dev_dax1):
        clear_caches()
        results['pymm_dram_dev_dax1'] = pymm_func(args.dev_dax1_path,args.size_GB,  array_pos, is_dev_dax=1)
    
    if (args.pymm_nvme):
        clear_caches()
        results['pymm_dram_nvme'] = pymm_func(args.nvme_path,args.size_GB,  array_pos)
    
    if (args.numpy_save_nvme):
        clear_caches()
        results['ndarray_save_dram_nvme'] = ndarry_save_func(args.nvme_path, args.size_GB,  array_pos)
    
    if (args.numpy_save_fs_dax0): 
        clear_caches()
        results['ndarray_save_dram_fs_dax0'] = ndarry_save_func(args.fs_dax0_path, args.size_GB,  array_pos)
    
    if (args.numpy_save_fs_dax1): 
        clear_caches()
        results['ndarray_save_dram_fs_dax1'] = ndarry_save_func(args.fs_dax1_path, args.size_GB,  array_pos)
    
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
    
def create_rand_data(size_GB):
    ### Create the data ###
    size = (int(size_GB)*G)
    t0 = time.time()
    print ("start Creating the data of size %dGB " % (int(size_GB)))
    array = np.random.rand(size)
#    pymm_size_mb = pymm_shelf_factor*int(array_size_mb)
    print ("Creating the data of size %uMB took: %0.2fsec" % (int(array.nbytes/1024/1024), time.time() - t0))
    return array

def create_rand_pos(num, end_pos):
    ### Create the data ###
    t0 = time.time()
    array = np.random.randint(end_pos, size=(int(num),))
    print ("Creating the data of size shape[0]=%u took: %0.2fsec" % (array.shape[0], time.time() - t0))
#    print (array)
    return array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("size_GB", type=str, help="The data size in GB")
    parser.add_argument("output_dir", type=str, help="where to store the results")
    parser.add_argument("--num_reads", type=int, default=1000*1000, help="The number of reads to the array")
    parser.add_argument("--read_size", type=int, default=8, help="The read size in Bytes")
    parser.add_argument("--test_all", action="store_true", default=False, help="run all the different options")
    parser.add_argument("--numa_local", action="store_true", default=False, help="run all the different options")
    parser.add_argument("--dram", action="store_true", default=False, help="deep copy the array to the DRAM")    
    parser.add_argument("--numpymemmap_nvme", action="store_true", default=False, help="save the array as numpy_mmap")    
    parser.add_argument("--numpymemmap_fs_dax0", action="store_true", default=False, help="save the array as numpy_mmap")    
    parser.add_argument("--numpymemmap_fs_dax1", action="store_true", default=False, help="save the array as numpy_mmap")    
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
    
    global num_reads
    global read_size
    num_reads =  args.num_reads
    read_size =  args.read_size
    num_read_items_8B = 1000*1000 
    num_read_items_64B = 1000000
    num_read_items_4K = 100
    num_read_items_2M = 10

    array_pos = create_rand_pos(num_reads, int(args.size_GB)*1024*1024*1024/8 - read_size/8)

    results = run_tests_func(args, array_pos)
    with open(args.output_dir + "_results." + str(args.size_GB) + "GB.read_size_" + str(args.read_size) + "B.num_read_" + str(args.num_reads) + ".json" ,'w') as fp:
            json.dump(results, fp)

if __name__ == "__main__":
            main()

