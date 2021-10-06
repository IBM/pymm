#!/usr/bin/python3 -m unittest
#
# basic numpy ndarray
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

class TestNdarray(unittest.TestCase):
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
        self.assertTrue(np.array_equal(self.s.w, w))

        log("Testing: pymm.ndarray RHS ...")
        self.s.x = pymm.ndarray((100,100),dtype=np.uint8)
        self.s.x = pymm.ndarray([2,2,2],dtype=np.uint8)
        x = np.ndarray([2,2,2],dtype=np.uint8)
        self.s.x.fill(8)
        x.fill(8)
        self.assertTrue(np.array_equal(self.s.x, x))

        self.s.erase('x')
        self.s.erase('w')

    def test_ndarray_shape(self):
        self.s.r = np.ndarray((3,4,))
        self.s.r.fill(1)
        print(id(self.s.r))
    
        self.assertTrue(self.s.r.shape[0] == 3 and self.s.r.shape[1] == 4)

        self.s.r2 = self.s.r.reshape((2,6,))
        print(id(self.s.r2))

        self.assertTrue(self.s.r2.shape[0] == 2 and self.s.r2.shape[1] == 6)

        self.s.r3 = self.s.r.reshape((1,-1,))
        print(id(self.s.r3))

    
        self.s.r2.fill(2)
        self.s.r3.fill(3)
        print(self.s.r)
        print(self.s.r2)
        print(self.s.r3)
    
        self.s.erase('r')
        self.s.erase('r2')
        self.s.erase('r3')

    def test_ndarray_slicing(self):
        log("Testing: np.ndarray slicing ...")
        d = np.ones((3,5,),dtype=np.uint8)
        self.s.w = d
        self.s.w_flat = self.s.w.reshape(-1)
        self.assertTrue(np.array_equal(self.s.w_flat, np.ones((3*5,),dtype=np.uint8)))
        self.s.x = np.arange(0,10)
        self.assertTrue(self.s.x[-1] == 9)
        self.assertTrue(self.s.x[-2] == 8)
        self.assertTrue(np.array_equal(self.s.x[-2:],[8,9]))
        self.assertTrue(np.array_equal(self.s.x[:-2],np.arange(0,8)))
        self.assertTrue(np.array_equal(self.s.x[::-1],[9,8,7,6,5,4,3,2,1,0]))
        self.assertTrue(np.array_equal(self.s.x[1::-1],[1,0]))
        self.assertTrue(np.array_equal(self.s.x[-3::-1],[7,6,5,4,3,2,1,0]))
        self.assertTrue(np.array_equal(self.s.x[:-3:-1],[9,8]))
        self.s.i = 1
        self.s.w[:,int(self.s.i)]
        self.s.w[:,self.s.i]

    def test_column_access(self):
        log("Testing: column access...")
        A = np.zeros((3, 4,),dtype=np.uint8)
        A[1] = 1
        A[2] = np.arange(0,4)
        A[:,0] = 2
        b = np.arange(3).reshape(3,1)
        q = (np.random.randn(3,1) - b/10).reshape(-1)
        # q is ndarray [x,y,z]
        A[:,0] = q

        self.s.B = np.zeros((3, 4,), dtype=np.uint8)
        self.s.B[1] = 1
        self.s.B[:,0] = 2
        self.s.B[:,0] = q
        print(self.s.B)
        self.s.erase('B')

    def test_column_access_2(self):
        log("Testing: column access variation...")
        self.s.A = np.zeros((3, 4,),dtype=np.uint8)
        self.s.b = np.arange(3).reshape(3,1)
        self.s.c = np.arange(3).reshape(3,1)
        self.s.A[:, 0] = (self.s.b - self.s.c).reshape(-1)

    def test_matrixop_and_reshape(self):
        self.s.theta = pymm.ndarray((3, 4,))
        self.s.theta.fill(2.0)
        self.s.moment = pymm.ndarray((3, 4,))
        self.s.moment.fill(1.2)
        self.s.theta_deviation = (self.s.theta - self.s.moment).reshape(-1)        
    
        


if __name__ == '__main__':
    unittest.main()

    
