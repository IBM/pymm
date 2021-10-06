#!/usr/bin/python3
import pymm
import numpy as np
import gc

shelf = pymm.shelf('myShelf',size_mb=32,backend='hstore-cc',pmem_path='/mnt/pmem0')

shelf.m = np.arange(0,9).reshape(3,3)

print('shelf.m:\n {}'.format(shelf.m))
print('shelf.m.name: {}'.format(shelf.m.name))
print('shelf.m.addr: {}'.format(shelf.m.addr))

print('creating reference ref to shelf.m ...')
ref = shelf.m
print('shelf.ref.addr: {}'.format(ref.addr))

print('shelf-to-shelf copy ...')
shelf.n = shelf.m
shelf.n += 10

print('shelf.m:\n {}'.format(shelf.m))
print('shelf.m.addr: {}'.format(shelf.m.addr))
print('shelf.n:\n {}'.format(shelf.n))
print('shelf.n.addr: {}'.format(shelf.n.addr))

# need to delete the outstanding reference
del ref
gc.collect()

shelf.erase('n')
shelf.erase('m')

