# Title: Accelerating Fault tolerant for Python Users with Persistent memory

In this blog, I would like to introduce our new open-source python library called: PyMM (Python Memory Management).
This python library makes it easy for a Python developer to use Persistent Memory (e.g., Intel Optane Non-Volatile DIMMs). 
The approach brings persistent memory to existing data types, and we started with NumPy arrays and PyTorch tensors since
they are both popular data types among data science users.

## What is Persistent Memory:

Persistent Memory is a term for a device that resides on the memory bus and has high throughput, low latency(under one micro-sec), and non-volatility. The data stays on the device after a system crash in contrary to DRAM, which is volatile.  

The idea beyond this device is to close the gap between SSD Flash Drive and DRAM. In 2019, Intel launched its Intel 3D XPoint DIMMs (also known as Optane DC persistent memory modules), which is the current dominant solution, 
Optane DC's main attributes are:
- Data persists after power loss, similar to SSD Flash and contrary to DRAM, where the data is lost after power loss.  
- Provides access latency which is two orders of magnitude less than NVMe (SSD Flash drive) devices, but the latency of read/write is x3-5 times slower than DRAM.  
- The throughput is more than NVMe but less than DRAM.
- Suppose to be x2 cheaper than DRAM but order of magnitude expensive than NVMe.   
- Being on the memory bus allows caching in the CPU similar to DRAM with cache coherence for multi-process. 
- The maximum capacity of one Optane DC DIMM is 512GB which is x8 larger than the current DDR4 DRAM and in one server, we can reach up to 6TB.

### Optane DC modes
Optane can configurate in two modes:
#### Memory Mode

#### Persistent Mode 

##### Device DAX (DEVDAX)

##### FS-DAX 

## Motivation 

Persistent Memory is in its early age and didn't reach its full potential. Most of the research and the adoptions for Optane are
in two dimensions, databases and storage systems use the persistent mode for fault tolerance, fast recovery, and reduced battery cost.
For the memory mode, the usage is by data science that needs enormous memory with static data.

|  | Memory Mode  | Persistent Mode  |
| ------- | --- | --- |
| Data science | v | This Blog |
 Database/ Storage Systems | few | v |



We see great potential for data science users in the sense of Fault tolerance.

### Fault tollerent

Machine Learning models are expensive to train: they require expensive high compute and long training times. Therefore, models are extra
sensitive to program faults or unexpected system crashes, which can erase hours if not days worth of work. Since PM is faster than native storage there is huge potintial to use it instead of NVMe. 

We will see one use case:
#### Checkpointing 
The most popular strategy for falut tollerent in machine learning is checkpointing, where periodically the model state is saved to persistent
storage and if the system crashed, the algorithm can return to its last checkpointing. 

In our syntetic benchmark of checkpoiting 10GB of NumPy array, we see that checkpoint using PyMM in DevDAX mode, you can save almost half the time to checkpoint to NVMe, which is 3.5sec. Assume that in your machine learning algorithm you have 10000 epochs, you save 10 hours just by saving to PM instead of NVMe. 

Two pic:
2b
3b




### Other fault tollerent use cases

We see in our benchmarking that persistent random write to NumPy array (which means that we persist after each write) is much faster with PyMM on PM than writing to FLASH SSD. We see here a potinial for on-line learning when you wish to persist your model after each change. 

We see that writing directly from GPU to PM is just 20% slower than writing from GPU to CPU and your model can be persist.

 

Machine Learning models are expensive to train: they require expensive high-1
compute hardware and have long training times. Therefore, models are extra2
sensitive to program faults or unexpected system crashes, which can erase hours if3
not days worth of work. While there are plenty of strategies designed to mitigate the4
risk of unexpected system downtime, the most popular strategy in machine learning5
is called checkpointing: periodically saving the state of the model to persistent6
storage. Checkpointing is an effective strategy, however, it requires carefully7
balancing two operations: how often a checkpoint is made (the checkpointing8
schedule), and the cost of creating a checkpoint itself.
  
  

# Install PyMM

### Dpendencies
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
There is also a pre-complie pymm with the latest version:
https://hub.docker.com/repository/docker/moshik1/pymm

#### Docker run command
In the docker run command, you should add a volume mount point that the shelf will run on.
In this example, we are using "/mnt/pmem0" for FS-DAX, but you can also use any other mount point. 
```
docker run -it -v /mnt/pmem0:/mnt/pmem0 moshik1/pymm:tag
```

# simple Usage

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



## Flush  

When using persistent memory the writes that are still in the cache and didn't flush to the PM can be vanished after a crash. 
To make sure that the data is secure in the PM you need to flush the data. 
We have persist function to FLUSH the data. for example: 
```
.persist()
```

In NumPy array you can use a persist_offset()

We have a more advance method that can speed up the persistent when persist that are larger than 30MB. 
For this use you need a sudo permission, you will have 


## An How to create a Persistent Memory 

You can create Persistet Memory in two modes fs_dax and dev_dax. The advantage of dev_dax is the higher performance then fs_dax in .... . On the other hand, it is mush easier to with debug with fs_dax since there is a file system and it is easy to control the data. 

### Persistent Mode
```
$ echo "create FS_DAX 0 in namespace0.0 "
$ sudo ndctl list
$ sudo ndctl destroy-namespace namespace0.0 -f
$ sudo mkdir /mnt/pmem0
$ sudo ndctl create-namespace -m fsdax -e namespace0.0 --align 2M --size 405874409472 --force
$ sudo mkfs -t ext4 /dev/pmem0
$ sudo mount -o dax /dev/pmem0 /mnt/pmem0
$ sudo chmod a+rwx /mnt/pmem0
```
#### DevDAX
```
echo "create DEV_DAX 0 in namespace0.1 "
sudo ndctl list
sudo ndctl destroy-namespace namespace0.1 -f
sudo ndctl create-namespace -m devdax -e namespace0.1 --size 405874409472 --align 2M --force
sudo chmod go+rw /dev/dax0.1
```
For more detailes: 
link:

script that create 4 PM disks: 2 FS-DAX and 2 DEVDEAX, each on each socket:
link: 


#### Memory Mode

— You might need to remove  existing namespace
sudo ndctl destroy-namespace namespace0.0 -f


sudo ipmctl create -goal MemoryMode=100
sudo ipmctl show -memoryresources
https://docs.pmem.io/ipmctl-user-guide/provisioning/create-memory-allocation-goal

You can conf a hybrid of Memory mode and Persistent mode


### Clear DEVDAX
To clear the dev_dax you should use the command DAX_RESET=1 before starting your program
```
$ DAX_RESET=1
```


## How to emulate a persistent Memory 
The emulation of persistent memory on DRAM is voltile but it allows you to emulate persistent memory. It uses a kernel ability to create a emulate persistent memory space on your DRAM. 

In linux distribution REDHAT or FEDORA, change the kernel using this task.
In this example we emulate 60GB start from offset 12GB.    
```
sudo grubby --args="memmap=60G\!12G" --update-kernel=ALL
```

To see the changes in the config file, please use this command.   
```
$ cat /etc/default/grub
GRUB_TIMEOUT=5
GRUB_DISTRIBUTOR="$(sed 's, release .*$,,g' /etc/system-release)"
GRUB_DEFAULT=saved
GRUB_DISABLE_SUBMENU=true
GRUB_TERMINAL_OUTPUT="console"
GRUB_CMDLINE_LINUX="rd.driver.blacklist=nouveau memmap=60G!12G"
GRUB_DISABLE_RECOVERY="true"
GRUB_ENABLE_BLSCFG=true
GRUB_CMDLINE_LINUX="rd.driver.blacklist=nouveau memmap=60G!12G"
GRUB_CMDLINE_LINUX="rd.driver.blacklist=nouveau memmap=60G!12G"
GRUB_CMDLINE_LINUX="rd.driver.blacklist=nouveau memmap=60G!12G"
GRUB_CMDLINE_LINUX="rd.driver.blacklist=nouveau memmap=60G!12G"
GRUB_CMDLINE_LINUX="rd.driver.blacklist=nouveau memmap=60G!12G"
```
Then reboot the machine to update the kernel
```
sudo reboot
```

On my machine since I have already two PM DIMMS it creates amn emulated disk, pmem2.
If you don't have any PM, the kernel will creates it as pmem0.
This command will show you that there is a new disk
```
[moshikh@sg-pmem2 ~]$ lsblk
NAME                    MAJ:MIN RM   SIZE RO TYPE MOUNTPOINT
...
pmem2                   259:4    0     4G  0 disk /mnt/mem
```

Create an emulated FS-DAX
```
sudo mkfs.ext4 /dev/pmem2
sudo mount -o dax /dev/pmem2 /mnt/mem/
sudo chmod 777 /mnt/mem/
```

Create an emulated DEVDAX
```
sudo ndctl create-namespace -e namespace2.0 -m devdax --align 2M  --force
```

Links for emulated PM: https://pmem.io/blog/2016/02/how-to-emulate-persistent-memory/
https://pmem.io/knowledgebase/howto/100000012-how-to-emulate-persistent-memory-using-the-linux-memmapkernel-option/

## More reading staff:
PyMM paper: https://dl.acm.org/doi/10.1145/3477113.3487266




