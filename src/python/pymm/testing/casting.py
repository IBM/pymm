#!/usr/bin/python3 -m unittest
#
# basic base type test
#
import unittest
import pymm
import numpy as np
import math
import torch

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def print_error(*args):
    print(colored(255,0,0,*args))

def log(*args):
    print(colored(0,255,255,*args))
    
force_new=True

class TestCasting(unittest.TestCase):
    def setUp(self):
        global force_new
        self.s = pymm.shelf('myShelf',size_mb=1024,pmem_path='/mnt/pmem0',force_new=force_new)
        force_new=False

    def tearDown(self):
        del self.s
        
    def test_ndarray(self):
    
        log("Testing: ndarray")
        self.s.x = pymm.ndarray((100,100))
        n = np.ndarray((100,100))
    
        # shelf type S
        self.assertTrue(str(type(self.s.x)) == "<class 'pymm.ndarray.shelved_ndarray'>")

        self.s.x.fill(1)
        # shelf type S after in-place operation
        self.assertTrue(str(type(self.s.x)) == "<class 'pymm.ndarray.shelved_ndarray'>")
    
        # shelf type S * NS (non-shelf type)
        self.assertTrue(str(type(self.s.x * n)) == "<class 'numpy.ndarray'>")

        # shelf type NS * S
        self.assertTrue(str(type(n * self.s.x)) == "<class 'numpy.ndarray'>")
    
        # shelf type S * shelf type S
        self.assertTrue(str(type(self.s.x * self.s.x)) == "<class 'numpy.ndarray'>")

        self.s.erase('x')
        log("Testing: ndarray OK!")


    def test_torch_tensor(self):
    
        log("Testing: torch_tensor")
        self.s.x = pymm.torch_tensor((10,))
        n = torch.Tensor([1,2,3,4,5,6,7,8,9,10])

        # shelf type S
        self.assertTrue(str(type(self.s.x)) == "<class 'pymm.torch_tensor.shelved_torch_tensor'>")
    
        self.s.x.fill(1)
        self.assertTrue(self.s.x[0] == 1)
        
        # shelf type S after in-place operation
        self.assertTrue(str(type(self.s.x)) == "<class 'pymm.torch_tensor.shelved_torch_tensor'>")

        # shelf type S * NS (non-shelf type)
        self.assertTrue(str(type(self.s.x * n)) == "<class 'torch.Tensor'>")

        # shelf type NS * S
        self.assertTrue(str(type(n * self.s.x)) == "<class 'torch.Tensor'>")
        
        # shelf type S * shelf type S
        self.assertTrue(str(type(self.s.x * self.s.x)) == "<class 'torch.Tensor'>")
    
        self.s.x += 1
        self.s.x *= 2
        self.s.x -= 0.4
        self.s.x /= 2
    
        self.s.erase('x')
        log("Testing: torch_tensor OK")

        
if __name__ == '__main__':
    unittest.main()
