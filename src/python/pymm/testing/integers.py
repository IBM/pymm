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


shelf = pymm.shelf('myShelf',size_mb=1024,pmem_path='/mnt/pmem0',force_new=True)

class TestNdarray(unittest.TestCase):
    def test_integer_assignment(self):
        log("Test: integer assignment and re-assignment")
        shelf.a = 2
        shelf.b = 30
        shelf.b = shelf.b / shelf.a
        print(shelf.b)
        shelf.erase('a')
        shelf.erase('b')

    def test_integer_with_ndarray(self):
        log("Test: array change with integer")
        shelf.count = 1
        shelf.count += 1
        log("Test: array change with ndarray")
        shelf.n = np.arange(0,9).reshape(3,3)
        print(shelf.n)
        shelf.p = shelf.n / 3
        print(shelf.p)
        shelf.p = shelf.n / int(shelf.count)
        print(shelf.p)
        shelf.p = shelf.n / shelf.count
        print(shelf.p)
        shelf.erase('p')
        shelf.erase('n')

    def test_rmod_operation(self):
        log("Test: rmod op")
        shelf.c = 10
        print('18%10={}'.format(18 % shelf.c))
        print('10%3={}'.format(shelf.c % 3))
        self.assertTrue(18 % shelf.c == 8)
        self.assertTrue(shelf.c % 3 == 1)

        
        
        
        

if __name__ == '__main__':
    unittest.main()

