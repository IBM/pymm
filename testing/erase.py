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

    def test_erase(self):
        log("Testing: erase ...")
        shelf.x = 9
        shelf.y = 10
        shelf.n = pymm.ndarray((9,9))
        shelf.t = pymm.torch_tensor([[8,8],[9,9]])
        shelf.b = b'El ni\xc3\xb1o come camar\xc3\xb3n'
        shelf.l = pymm.linked_list()
        shelf.l.append(1)

        print(shelf.t)

        print(shelf.items)
        shelf.erase(shelf.n)
        shelf.erase(shelf.b)
        shelf.erase(shelf.t)
        shelf.erase(shelf.l)
        
        print(shelf.items)
        shelf.erase('x')
        self.assertTrue(shelf.items == ['y'])

        gc.collect()
        shelf.erase(shelf.y)
        print(shelf.items)



if __name__ == '__main__':
    unittest.main()
