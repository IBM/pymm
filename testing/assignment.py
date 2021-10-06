#!/usr/bin/python3 -m unittest
#
# basic shelf assignment tests
#
import unittest
import pymm
import numpy as np
import math
import torch

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def log(*args):
    print(colored(0,255,255,*args))
    
force_new=True
class TestAssignment(unittest.TestCase):
    def setUp(self):
        global force_new
        self.s = pymm.shelf('myShelf',size_mb=1024,pmem_path='/mnt/pmem0',force_new=force_new)
        force_new=False
    
    def tearDown(self):
        del self.s
        
    def test_ndarray(self):

        log("Testing: np.ndarray RHS ...")
        self.s.w = np.ndarray((100,100),dtype=np.uint8)
        self.s.w = np.ndarray([2,2,2],dtype=np.uint8)
        w = np.ndarray([2,2,2],dtype=np.uint8)
        self.s.w.fill(9)
        w.fill(9)    
        if np.array_equal(self.s.w, w) != True:
            raise RuntimeError('ndarray equality failed')

        log("Testing: pymm.ndarray RHS ...")
        self.s.x = pymm.ndarray((100,100),dtype=np.uint8)
        self.s.x = pymm.ndarray([2,2,2],dtype=np.uint8)
        x = np.ndarray([2,2,2],dtype=np.uint8)
        self.s.x.fill(8)
        x.fill(8)    
        self.assertTrue(np.array_equal(self.s.x, x))
        
        self.s.erase('x')
        self.s.erase('w')
        log("Testing: ndarray OK!")


    def test_torch_tensor(self):
        log("Testing: torch.tensor RHS ...")
        self.s.w = torch.tensor((100,100,100),dtype=torch.uint8)
        self.s.w = torch.tensor([2,2,2],dtype=torch.uint8)
        w = torch.tensor([2,2,2],dtype=torch.uint8)
        self.s.w.fill_(9)
        w.fill_(9)
        print("self.s.w = {}".format(self.s.w))
        print("w = {}".format(w))
        self.assertTrue(torch.equal(self.s.w, w))

        log("Testing: pymm.ndarray RHS ...")
        #self.s.x = pymm.torch_tensor((100,100,100),dtype=torch.uint8)

        self.s.x = pymm.torch_tensor([[1,1,1,1],[2,2,2,2]],dtype=torch.uint8)
        x = torch.tensor([[1,1,1,1],[2,2,2,2]],dtype=torch.uint8)
        
        print("self.s.x =\n {}".format(self.s.x))
        print("x =\n {}".format(x))

        self.assertTrue(torch.equal(self.s.x, x))
        
        log("Testing: ndarray OK!")


if __name__ == '__main__':
    unittest.main()
