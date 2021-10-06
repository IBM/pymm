import pymm
import numpy as np
import math
import torch
import gc

from inspect import currentframe, getframeinfo
line = lambda : currentframe().f_back.f_lineno

def fail(msg):
    print(colored(255,0,0,msg))
    raise RuntimeError(msg)

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def print_error(*args):
    print(colored(255,0,0,*args))

def log(*args):
    print(colored(0,255,255,*args))
    

print('[TEST]: enabling transient memory ...')
pymm.enable_transient_memory(pmem_file='/mnt/pmem0/swap',pmem_file_size_gb=2, backing_directory='/tmp')

s = pymm.shelf('myShelf',size_mb=2048,backend="hstore-cc",force_new=True)

# create a large right-hand side expression which will be evaluated in pmem transient memory
w = np.ndarray((1000000000*1),dtype=np.uint8) # 1GB

# create something even larger that will fail in pmem and drop into mmap filed
w2 = np.ndarray((1000000000*10),dtype=np.uint8) # 10GB

# copy to shelf
s.w = w

# force clean up of w
del w
del w2
gc.collect()
gc.get_objects()

print(s.w._value_named_memory.addr())

print('[TEST]: disabling transient memory ...')
pymm.disable_transient_memory()
