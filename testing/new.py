#!/usr/bin/python3 -m unittest
#
# testing transient memory (needs modified Numpy)
#
import unittest
import pymm
import numpy as np
import torch
import gc

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def log(*args):
    print(colored(0,255,255,*args))

shelf = pymm.shelf('myShelf',size_mb=1024,pmem_path='/mnt/pmem0',force_new=True)

class TestNew(unittest.TestCase):
    
    def test_a(self):
        shelf.x = 'Hello world'
        shelf.y = 1234
        shelf.z = 1.34
        shelf.inspect(True)
        shelf.persist()
        print("Persistent shelf variables!")
        shelf.inspect(True)



if __name__ == '__main__':
    unittest.main()
