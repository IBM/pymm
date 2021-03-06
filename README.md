

<p align="center">
  <img width="160" height="120" src="https://raw.githubusercontent.com/IBM/pymm/main/pymm_logo.png">
</p>

# Python Memory Management (PyMM)

Python Memory Management is a python library that makes it easy for 
a Python developer to get started using Persistent Memory
 (e.g., Intel Optane Non-Volatile DIMMs). The approach is to bring
 persistent memory to existing data types that are understood by the broad
 array of libraries in the Python ecosystem.

To begin with, our focus is on the NumPy arrays and PyTorch tensors and
 the data analytics application domain. That allows storing and 
manipulating NumPy arrays and PyTorch tensors on Persistent Memory.
 The key benefit of PyMM is its ability to persistently store program 
variables without the need to serialize or de-serialize (e.g. pickle).
 Using PyMM, program variables are stored in logical groupings known as
 “shelves”, which inherently map to different memory resources in the system.
 Variables that exist on a shelf can be readily used with commonly used
 existing libraries including NumPy and PyTorch. In future releases, 
we hope to extend this to Apache Arrow, and panda dataframes-like data types.

# Install PyMM

### Dependencies
```
./deps/install-<Your-OS-Version>.sh
./deps/install-python-deps.sh
```

### Build PyMM
Build default is Optimized. Use --debug for debug mode.
```
python setup.py
```

Once everything has been correctly installed, the pymm module can be loaded.

```python
>>> import pymm
>>>
```

For more installation detailes: [BUILD_PyMM](./info/build_PyMM.md)


### Docker hub container image
A precompile pymm with the latest version:
https://hub.docker.com/repository/docker/moshik1/pymm

#### Docker run command
In the docker run command, you should add a volume mount point that the shelf will run on.
In this example, we are using "/mnt/pmem0" for FS-DAX, but you can also use any other mount point. 
```
docker run -it -v /mnt/pmem0:/mnt/pmem0 moshik1/pymm:tag
```

## Persistent Shelf Abstraction

PyMM allows the programmer to easily define what type of memory (i.e.,
volatile or persistent) a variable should be assigned to.  This is
achieved with the **shelf** abstraction.  Each shelf is associated
with an MCAS memory pool.  For ease of management, we recommend setting
up an fsdax persistent memory mount (e.g., /mnt/pmem0) - see [Notes on
DAX](./MCAS_notes_on_dax.md).  This is where shelves will be created.

Shelves are given a name and capacity (currently they cannot expand or
contract). A shelf can be created as follows:

```python
>>> s = pymm.shelf('myShelf',size_mb=1024,pmem_path='/mnt/pmem0',force_new=True)
```

The last parameter, 'force_new', if set to True the call clears any existing
pool and creates an entirely new allocation.  The 'pmem_path' parameter
defines where the shelves are created.

Normal re-opening of an existing shelf is performed with:

```python
>>> s = pymm.shelf('myShelf',size_mb=1024,pmem_path='/mnt/pmem0')
```

```bash
$ ls -sh /mnt/pmem0
total 1.1G
1.0G myShelf.data  4.0K myShelf.map
```

Note, that each shelf/pool results in a data file and metadata file (.map).  To
erase a shelf, the corresponding files can be deleted through 'rm'.  If
a shelf already exists, then the pymm.shelf construction will re-open 
the existing shelf.

Once a shelf has been created, variables can be "placed" on it.
Shelved data is **durable** across process and machine restarts (as
are flushed files on traditional storage).  Of course, if the machine
homing the persistent memory resources fails, your data cannot be
easily recovered.  This is a topic of future work.

If you are using a devdax persistent memory partition, you can use the DAX_RESET=1
environment variable to reset and clear the pool.

## Shelving Variables

When a shelf is opened, any variables on it are immediately available.  There is no need for loading or de-serialization (e.g., unpickling).

To create a Numpy ndarray on the shelf, a pymm.ndarray data type is used.  The
constructor parameters for this are identical to numpy.ndarray.  To create a variable
*x* on the shelf, the following assignment is used:

```python
>>> import numpy as np
>>> s.x = pymm.ndarray((1000,1000),dtype=np.int64)
```

This creates the ndarray in persistent memory and under-the-hood, stores the
metadata so that the variable type is known on recovery.  Once created the
array can be populated in place.  For example:

```python
>>> s.x[1][1] = 9 # explicit slice assignment
>>> s.x.fill(3)   # in-place operation

# load data from file
>>> with open("array.dat", "rb") as source:
        source.readinto(memoryview(s.x))
```

Shelved variables can be assigned as copies of non-shelved variables. In this 
case, the right-hand-side is evaluated in DRAM before copying over to persistent
memory.

```python
# create random 2D-array
>>> s.x = np.random.randint(1,100, size=(1000,1000), dtype=int64)
```

Shelved variables can be read directly:

```python
>>> s.y
shelved_ndarray([[75, 87, 59, ..., 27,  2, 33],
                 [69, 56, 62, ..., 36, 35,  3],
                 [91,  3, 89, ...,  4, 89, 64],
                 ...,
                 [42, 87, 15, ..., 67,  4, 77],
                 [26, 74, 55, ..., 87, 83, 46],
                 [59, 94, 87, ..., 29, 13, 62]])
>>> sum(s.y.diagonal())
50160
```

Variables are erased as follows:
```python
>>> s.erase('y')
```

To show the variables on a shelf, the *items* property is used, which gives a list
of variable names (strings) as follows:

```python
>>> s.items
['x', 'y']
```

Alternatively:
```python
>>> dir(s)
['__class__', '__del__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattr__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'erase', 'get_item_names', 'mr', 'name', 'x', 'y']
```

## Usage

You can get the current memory usage (as a percent of the memory resources) as follows:

```python
>>> s.used
6
```

## References 

Volatile references can be used to access shelved data.  References cannot themselves be
put on a shelf:

```python
>>> Yref = s.y
>>> Yref
shelved_ndarray([[75, 87, 59, ..., 27,  2, 33],
                 [69, 56, 62, ..., 36, 35,  3],
                 [91,  3, 89, ...,  4, 89, 64],
                 ...,
                 [42, 87, 15, ..., 67,  4, 77],
                 [26, 74, 55, ..., 87, 83, 46],
                 [59, 94, 87, ..., 29, 13, 62]])
>>> Yref.sum()
50015284
```

## Demo

A simple demo is available (src/python/pymm/demo.py).  You will need matplotlib
and scikit-image working for the demo (it will open a GUI dialog box).

```python
$ python3
Python 3.6.8 (default, Aug 18 2020, 08:33:21) 
[GCC 8.3.1 20191121 (Red Hat 8.3.1-5)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import pymm
>>> pymm.demo(True)
```

## Supported Types

Currently, only NumPy ndarray types can be put on a shelf.  Future work plans to look at
supporting other types.

```python
>>> s.y = 0
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/mcas/build/src/python/pymm/build/lib.linux-x86_64-3.6/pymm/shelf.py", line 128, in __setattr__
    raise RuntimeError('non-pymm type ({}, {}) cannot be put on the shelf'.format(name,type(value)))
RuntimeError: non-pymm type (y, <class 'int'>) cannot be put on the shelf
```

## Future Plans
* Extend beyond Numpy ndarray type
* Support other types of memory (e.g., CXL attached)
* Shelf expansion
* Support for distributed durability.

## Appendix

* Example creating a large array:

```python
Python 3.6.8 (default, Aug 18 2020, 08:33:21) 
[GCC 8.3.1 20191121 (Red Hat 8.3.1-5)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import pymm
>>> import numpy as np
>>> s = pymm.shelf('myShelf', size_mb=1024*512, pmem_path='/mnt/pmem0', force_new=True)
>>> s.x = pymm.ndarray((1000,1000,1000,500),dtype=np.uint8)
>>> s.used
92
>>int(s.x.nbytes/1024/1024/1024)
465 
```

* Example loading a large file (beyond DRAM size)

```python
>>> import pymm
>>> import numpy as np
>>> s = pymm.shelf('myShelf', size_mb=1024*512, pmem_path='/mnt/pmem0', force_new=True)
>>> s.z = np.memmap("/mnt/nvme0/file_300GB.out", dtype="float32", mode="r")
>>> s.used
60
>>int(s.z.nbytes/1024/1024/1024)
300
```

Debugging:

```bash
PYMM_DEBUG=3 python3
```


