#!/usr/bin/python3 -m unittest
#
# basic numpy ndarray
#
import unittest
import pymm
import numpy as np
import math
import gc
import tensorflow as tf # you will need a recent version e.g.,2.6.0
import torch

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def print_error(*args):
    print(colored(255,0,0,*args))

def log(*args):
    print(colored(0,255,255,*args))


class TestNdarray(unittest.TestCase):
    
    def test_dlpack_array(self):
        log("Testing: dlpack_array ...")
        shelf = pymm.shelf('myShelf',size_mb=1024,pmem_path='/mnt/pmem0',force_new=True)
        shelf.a = pymm.dlpack_array((5,),dtype=np.float64)
        print(shelf.a.addr)
        print(shelf.a)
        shelf.inspect()
        del shelf
        gc.collect()

    def test_dlpack_array_reopen(self):
        log("Testing: dlpack_array (reopen) ...")
        shelf = pymm.shelf('myShelf', pmem_path='/mnt/pmem0',force_new=False)
        print(shelf.a)
        shelf.inspect()

    def test_dlpack_array_from_dlpack_tf(self):
        log("Testing: dlpack_array pycapsule generation ...")
        shelf = pymm.shelf('myShelf', pmem_path='/mnt/pmem0',force_new=False)
        capsule = shelf.a.as_capsule()
        print(capsule)
        log("Testing: dlpack_array importing to tensorflow ...")
        tf_tensor = tf.experimental.dlpack.from_dlpack(capsule)
        print(tf_tensor)
        shelf.inspect()
        del capsule
#        del tf_tensor
        gc.collect()
        log("Testing: dlpack_array pycapsule tests done")

    def XXtest_dlpack_array_from_dlpack_torch(self):
        log("Testing: dlpack_array pycapsule generation ...")
        shelf = pymm.shelf('myShelf', pmem_path='/mnt/pmem0',force_new=False)
        capsule = shelf.a.as_capsule()
        print(capsule)
        log("Testing: dlpack_array importing to pytorch ...")
        torch_tensor = torch.utils.dlpack.from_dlpack(capsule)
        print(torch_tensor)
        shelf.inspect()
        del capsule
#        del tf_tensor
        gc.collect()
        log("Testing: dlpack_array pycapsule tests done")
        

if __name__ == '__main__':
    unittest.main()

    
