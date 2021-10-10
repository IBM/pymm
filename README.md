# PyMM

PyMM is a python library that allows the storing and manipulation of existing heavily used types such as Numpy ndarray and PyTorch on Persistent Memory (e.g., Intel Optane Non-Volatile DIMMs). The key benefit of PyMM is its ability to persistently store program variables without the need to serialize or de-serialize (e.g. pickle).  Using PyMM, program variables are stored in logical groupings known as "shelves", which inherently map to different memory resources in the system.  Variables that exist on a shelf can be readily used with commonly used existing libraries including NumPy and PyTorch.

With PyMM a data type can be easily saved in Persistent Memory and directly transferred to the GPU/CPU without copying the data to DRAM.  Use of this memory technology means that: 
1.) after program or machine restarts, data is immediately accessible to the program without the need to reload from a slower storage system or database.
2.) save data-types that are too large to fit in DRAM can be supported (up to 6TB)

PyMM is implemented as a Python 3 library and CPython-based extension. The low-level index and memory management are taken from [MCAS](https://github.com/IBM/mcas) (Memory Centric Active Storage), which is a high-performance key-value store designed specifically for persistent memory and DRAM.


## How to Build

## Update submodules
```bash
cd mcas
git submodule update --init --recursive
```


## Run dependencies for your OS 

``` bash
cd deps
./install-python-deps.sh
cd ../
``` 

## Configure as debug build:

```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/dist .. 
```

Build bootstrap libraries:

```bash
make bootstrap
```

Build everything else (-j optional for parallel build):

```
make -j
```



## Verify 
```
$ python3
Python 3.6.8 (default, Mar 18 2021, 08:58:41)
[GCC 8.4.1 20200928 (Red Hat 8.4.1-1)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import pymm
[--(PyMM)--] Version <current version> (CC=env)
[LOG]: Pymm extension
```	

