# Python Micro-MCAS (PyMM)

NOTE: Experimental feature

Micro-MCAS is the term we use to represent a local, network-less
instantiation of MCAS.  It can be somewhat viewed as an alternative to
PMDK in that it provides software to manage the persistent memory
space with name-based (key) regions of memory (value).  Micro-MCAS uses
the persistent memory storage components (hstore, hstore-cc) from the 
fuller MCAS solution.

PyMM is Python software for using the Micro-MCAS capability.  The aim
of PyMM is to make it very easy for a Python developer to get started
with using persistent memory.  The approach is to bring persistent memory
to *existing* data types that are understood by the broad array of libraries
in the Python ecosystem.

To begin with, our focus is on the Numpy ndarray type and the data analytics application
domain.  In future releases, we hope to extend this to Apache Arrow,
and panda dataframes-like data types.  We aim to make using persistent memory with existing
libraries (e.g., scikit-learn) really easy.

## Getting Started

PyMM is based on Python 3.6 or later.  We use site-local package installs to avoid
conflicts with existing system-wide installed libraries.  At minimum,
for the demo, the following should be installed:

```bash
pip3 install numpy --user -I
pip3 install matplotlib --user -I
pip3 install scikit-image --user -I
```
Currently, PyMM requires a full build and install of MCAS.  See [MCAS_BUILD](../README.md).  This should be performed first.

Once everything has been correctly installed, the pymm module can be loaded.

```python
Python 3.6.8 (default, Aug 18 2020, 08:33:21) 
[GCC 8.3.1 20191121 (Red Hat 8.3.1-5)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import pymm
>>>
```


## Persistent Shelf Abstraction

PyMM allows the programmer to easily define what type of memory (i.e.,
volatile or persistent) a variable should be assigned to.  This is
achieved with the **shelf** abstraction.  Each shelf is associated
with a MCAS memory pool.  For ease of management, we recommend setting
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

Note, each shelf/pool results in a data file and metadata file (.map).  To
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

When a shelf is opened, any variables on it are immediately available.  There
is no need for loading or de-serialization (e.g., unpickling).

To create a Numpy ndarray on the shelf, a pymm.ndarray data type is used.  The
constructor parameters for this are identical to numpy.ndarray.  To create a variable
*x* on the shelf, the following assignment is used:

```python
>>> import numpy as np
>>> s.x = pymm.ndarray((1000,1000),dtype=np.int64)
```

This creates the ndarray in persistent memory and under-the-hood, stores the
meta data so that the variable type is known on recovery.  Once created the
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

Shelved variables, once created, cannot be implicitly overwritten.  Instead, they
must be explicitly *erased* from the shelf.

```python
>>> s.y =  np.random.randint(1,100, size=(1000,1000), dtype=np.int64)
setattr y = <class 'numpy.ndarray'>
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/mcas/build/src/python/pymm/build/lib.linux-x86_64-3.6/pymm/shelf.py", line 122, in __setattr__
    self.__dict__[name] = pymm.ndarray.build_from_copy(self.mr, name, value)
  File "/mcas/build/src/python/pymm/build/lib.linux-x86_64-3.6/pymm/ndarray.py", line 74, in build_from_copy
    strides = array.strides)
  File "/mcas/build/src/python/pymm/build/lib.linux-x86_64-3.6/pymm/ndarray.py", line 106, in __new__
    value_named_memory = memory_resource.open_named_memory(name)
  File "/mcas/build/src/python/pymm/build/lib.linux-x86_64-3.6/pymm/check.py", line 25, in _f
    return f(*args, **kwargs)
  File "/mcas/build/src/python/pymm/build/lib.linux-x86_64-3.6/pymm/memoryresource.py", line 104, in open_named_memory
    (handle, mem) = super()._MemoryResource_open_named_memory(name)
RuntimeError: named memory already open
>>> 
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

### Usage

You can get the current memory usage (as a percent of the memory resources) as follows:

```python
>>> s.used
6
```

### References 

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

A simply demo is available (src/python/pymm/demo.py).  You will need matplotlib
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

* Example creating large array:

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

* Example loading large file (beyond DRAM size)

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
