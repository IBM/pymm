# Basic benchmarks for different media

In these benchmarks we are testsing basic operations on different media type.

# Sequance Write 

In this benchmark we are testing sequance write operation on the following media:
- DRAM - deep copy
- FS_DAX - in the local numa and in the far numa with PyMM, numpy.save, pickle
- Dev_DAX  - in the local numa and in the far numa with PyMM
- NVMe -  PyMM, numpy.save, pickle


We create a numpy array with a predifne size with the following command:
--help
The defualt size is 1GB
Two optiones -g for GB and -m for MG
Example: 5GB
```
python3 write_seq_different_media_do_prepare.py -g 5
```
Example: 100MB
```
python3 write_seq_different_media_do_prepare.py -m 100
```
Two headers before running the python code and a few ....:
- DAX_REASET=1 - to reseat dev_dax
- numactl --cpunodebind=0 - to run the python program on specific socket 

```
$ DAX_REASET=1 numactl --cpunodebind=0 python3 write_seq_different_media_do_work.py
```

See all the options:
```
--help
```

12 options for testing
``
--test_all # Activated all the options
--dram # Deep copy an array from DRAM to DRAM
--pickle_nvme # Copy an array with pickle to NVMe
--pickle_fs_dax0  # Copy an array with pickle to fs_dax in socket0
--pickle_fs_dax1  # Copy an array with pickle to fs_dax in socket1 (different numa node)
--pymm_fs_dax0  # Copy an array to a PyMM shelf on fs_dax in socket0
--pymm_fs_dax1  # Copy an array to a PyMM shelf on fs_dax in socket1  (different numa node)
--pymm_dev_dax0 # Copy an array to a PyMM shelf on dev_dax in socket0
--pymm_dev_dax1 # Copy an array to a PyMM shelf on dev_dax in socket1 (different numa node)
--pymm_nvme # Copy an array to a PyMM shelf on NVMe
--numpy_save_nvme # Copy an array to with numpy.save to NVMe
--numpy_save_fs_dax0  # Copy an array to with numpy.save to fs_dax in socket0
--numpy_save_fs_dax1  # Copy an array to with numpy.save to fs_dax in socket1 (different numa node)
```
These are the locations on the different media: 
nvme_path = "/mnt/nvme0/tmp/"
fs_dax0_path = "/mnt/pmem0/tmp0"
fs_dax1_path = "/mnt/pmem1/tmp1"
dev_dax0_path = "/dev/dax0.1"
dev_dax1_path = "/dev/dax1.1"

You can change the default locations
```
--nvme_path 
--fs_dax0_path 
--fs_dax1_path 
--dev_dax0_path 
--dev_dax1_path 
```




