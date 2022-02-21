# Basic benchmarks for different media

In these benchmarks we are testsing basic operations on different media type.

# Sequance Write 

In this benchmark we are testing sequance write operation on th efollowing media:
We create a numpy array with a predifne size and bemchmark the write time of: 
- DRAM - deep copy
- FS_DAX - in the local numa and in the far numa with PyMM, numpy.save, pickle
- Dev_DAX  - in the local numa and in the far numa with PyMM
- NVMe -  PyMM, numpy.save, pickle


Two headers before running the python code and a few ....:
- DAX_REASET=1 - to reseat dev_dax
- numactl --cpunodebind=0 - to run the python program on specific socket 

```
$ DAX_REASET=1 numactl --cpunodebind=0 python3 write_seq_different_media_do_work.py
```

--help 

