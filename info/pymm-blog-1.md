# PyMM Usage

In this blog, I would like to introduce our new open-source python library called: PyMM (Python Memory Management).
This python library makes it easy for a Python developer to use Persistent Memory (e.g., Intel Optane Non-Volatile DIMMs). 
The approach brings persistent memory to existing data types, and we started with NumPy arrays and PyTorch tensors since
they are both popular data types among data science.


## What is Persistent Memory:


## Motivation 
Persistent Memory in's early age and didn't reach its full potential. Most of the research and the adoptions for Optane are
in two dimensions, databases and storage systems use the persistent mode for fault tolerance, fast recovery, and reduced battery cost.
For the memory mode, the usage is by data science that needs enormous memory with static data.

|  | Memory Mode  | Persistent Mode  |
| ------- | --- | --- |
| Data science | v | This Blog |
 Database/ Storage Systems | few | v |




This has great potinitail for data science users since not all the data is static, there is also models and the training phase can last for hours.
We see potinital after benchmarking NumPy arrays of diffreant sizes: 
1. Checkpointing and logging -  With PyMM on dax mode we see a checkpoint acceleration of almost x2. 
2. Persist Random writes - persist random writes to Optane with PyMM is x3 times faster then writing to NVMe or writing to Optane with other options.
3. Huge data - when the data is huge it is better to use peristent memory then moving to numa node (?Maybe not in this blog?)
  

## Benchmarking 

We are showing somce instrsting results of running PyMM with : 
two grphs: 
1. 10GB 
2. 10GB


## How to start: 

Build: 


## Basic commands 

### Open a shelf


### Create a ... 



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

To clear the dev_dax you should use the command DAX_RESET=1 before starting your program
```
DAX_RESET=1 python3 
```
 


## How to emulate a persistent Memory 
The emulation of persistent memory on DRAM is voltile but it allows you to emulate persistent memory, it gives you way to access the same variables over and over and it is much faster then using tempfs. 





## More reading staff:
PyMM paper: 

ArXiv Ppaper:  


