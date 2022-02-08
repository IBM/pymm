#!/usr/bin/python3 -m unittest
#
# "after" part of recovery test 
#
import unittest
import pymm
import numpy as np
import math
import os

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def log(*args):
    print(colored(0,255,255,*args))

shelf = 0
shelf2 = 0
class TestRecovery(unittest.TestCase):

    def setUp(self):
        global shelf
        global shelf2

        if shelf == 0:
            log("Recovery: running test setup...")

            shelf = pymm.shelf('myShelf', load_addr='0x700000000',
                               backend='hstore-cc', pmem_path='/mnt/pmem0/1', force_new=False)

            shelf2 = pymm.shelf('myShelf-2', load_addr='0x800000000',
                                backend='hstore-cc', pmem_path='/mnt/pmem0/2', force_new=False)

            print("Shelf.items = {}".format(shelf.items))
            print("Shelf2.items = {}".format(shelf2.items))
            log("Recovery: shelf init OK")
        
    def test_check_A(self):
        global shelf
        global shelf2
        print(shelf.A[1][2])
        y = np.ndarray((3,8),dtype=np.uint8)
        y.fill(1)
        y[1] += 1
        self.assertTrue(shelf.A[1][2] == 2)
        self.assertTrue(np.array_equal(shelf.A, y))

    def test_check_B(self):
        global shelf
        global shelf2
        self.assertTrue(np.array_equal(shelf.B, np.zeros((3,8,))))
        print(shelf.B)

    def test_check_C(self):
        global shelf
        global shelf2
        self.assertTrue(shelf2.A == 99)
        print(shelf2.A)

        
        

if __name__ == '__main__':
    unittest.main()
