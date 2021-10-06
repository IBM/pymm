#!/usr/bin/python3 -m unittest
#
# "before" part of recovery test 
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
class TestBefore(unittest.TestCase):

    def setUp(self):
        global shelf
        global shelf2

        if shelf == 0:
            log("Recovery: running test setup...")
            os.system("rm -Rf /mnt/pmem0/1")
            os.system("rm -Rf /mnt/pmem0/2")
            os.system("mkdir -p /mnt/pmem0/1")
            os.system("mkdir -p /mnt/pmem0/2")

            shelf = pymm.shelf('myShelf', size_mb=256, load_addr='0x700000000',
                               backend='hstore-cc', pmem_path='/mnt/pmem0/1', force_new=True)

            shelf2 = pymm.shelf('myShelf-2', size_mb=256, load_addr='0x800000000',
                                backend='hstore-cc', pmem_path='/mnt/pmem0/2', force_new=True)

            log("Recovery: shelf init OK")
        
    def test_write_A(self):
        global shelf
        global shelf2

        log("Recovery: creating shelf.A")
        shelf.A = np.ndarray((3,8),dtype=np.uint8)
        shelf.A.fill(1)
        shelf.A[1] += 1
        print(shelf.A)

    def test_write_B(self):
        global shelf
        global shelf2

        log("Recovery: creating shelf.B")
        shelf.B = np.zeros((3,8,))
        print(shelf.B)

    def test_write_C(self):
        global shelf
        global shelf2

        log("Recovery: creating shelf2.A")
        shelf2.A = 99
        print(shelf2.A)

    def test_write_final(self):
        global shelf
        global shelf2

        print("shelf.items ", shelf.items)
        print("shelf.items ", shelf2.items)
        del shelf
        del shelf2

        
        

if __name__ == '__main__':
    unittest.main()
