#!/usr/bin/python3 -m unittest
#
# basic shelf assignment tests
#
import unittest
import pymm
import numpy as np
import math
import torch
import gc


def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def log(*args):
    print(colored(0,255,255,*args))

shelf = pymm.shelf('myShelf',size_mb=1024,pmem_path='/mnt/pmem0',force_new=True)

class TestAssignment(unittest.TestCase):

    def test_ndarray(self):

        log("Testing: np.ndarray RHS ...")
        shelf.w = np.ndarray((100,100),dtype=np.uint8)
        shelf.w = np.ndarray([2,2,2],dtype=np.uint8)
        w = np.ndarray([2,2,2],dtype=np.uint8)
        shelf.w.fill(9)
        w.fill(9)    
        if np.array_equal(shelf.w, w) != True:
            raise RuntimeError('ndarray equality failed')

        log("Testing: pymm.ndarray RHS ...")
        shelf.x = pymm.ndarray((100,100),dtype=np.uint8)
        shelf.x = pymm.ndarray([2,2,2],dtype=np.uint8)
        x = np.ndarray([2,2,2],dtype=np.uint8)
        shelf.x.fill(8)
        x.fill(8)    
        self.assertTrue(np.array_equal(shelf.x, x))
        
        shelf.erase('x')
        shelf.erase('w')
        log("Testing: ndarray OK!")


    def test_torch_tensor(self):
        log("Testing: torch.tensor RHS ...")
        shelf.w = torch.tensor((100,100,100),dtype=torch.uint8)
        shelf.w = torch.tensor([2,2,2],dtype=torch.uint8)
        w = torch.tensor([2,2,2],dtype=torch.uint8)
        shelf.w.fill_(9)
        w.fill_(9)
        print("shelf.w = {}".format(shelf.w))
        print("w = {}".format(w))
        self.assertTrue(torch.equal(shelf.w, w))

        log("Testing: pymm.ndarray RHS ...")
        #shelf.x = pymm.torch_tensor((100,100,100),dtype=torch.uint8)

        shelf.x = pymm.torch_tensor([[1,1,1,1],[2,2,2,2]],dtype=torch.uint8)
        x = torch.tensor([[1,1,1,1],[2,2,2,2]],dtype=torch.uint8)
        
        print("shelf.x =\n {}".format(shelf.x))
        print("x =\n {}".format(x))

        self.assertTrue(torch.equal(shelf.x, x))
        
        log("Testing: ndarray OK!")



if __name__ == '__main__':
    unittest.main()
