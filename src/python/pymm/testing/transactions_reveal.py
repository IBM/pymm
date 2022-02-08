#!/usr/bin/python3 -m unittest
#
# basic inspection
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

shelf = pymm.shelf('myTransactionsShelf',pmem_path='/mnt/pmem0')

class TestReveal(unittest.TestCase):

    def test_inspect(self):
        shelf.inspect(verbose=False)
        
if __name__ == '__main__':
    unittest.main()
