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
  

## Flush issue 

wbinvd 

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




## An How to create a Persistent Memory 


## How to emulate a persistent Memory 



## More reading staff:
PyMM paper: 

ArXiv Ppaper:  


