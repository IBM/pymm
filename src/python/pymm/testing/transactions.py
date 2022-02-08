#!/usr/bin/python3 -m unittest
#
# basic transactions
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

#
# shelf = pymm.shelf('myShelf',pmem_path='/mnt/pmem0')

class TestTransactions(unittest.TestCase):
    
    def Xtest_transactions(self):

        shelf = pymm.shelf('myTransactionsShelf',size_mb=1024,pmem_path='/mnt/pmem0',force_new=True)
        log("Testing: transaction on matrix fill ...")
        shelf.n = pymm.ndarray((100,100),dtype=np.uint8)
        shelf.n += 1
        shelf.n += 1
        # shelf.s = 'This is a string!'
        # shelf.f = 1.123
        # shelf.i = 645338
        # shelf.b = b'Hello'

        # shelf.w = np.ndarray((100,100),dtype=np.uint8)
        # shelf.w.fill(ord('a'))
        # shelf.t = pymm.torch_tensor(np.arange(0,10))
        
        print(shelf.items)
        shelf.inspect(verbose=False)
        shelf.persist()
        shelf.inspect(verbose=False)

    def test_shelf_transaction(self):
        shelf = pymm.shelf('myTransactionsShelf',size_mb=1024,pmem_path='/mnt/pmem0',force_new=True)

        shelf.n = pymm.ndarray((100,100),dtype=np.uint8)
        shelf.m = pymm.ndarray((100,100),dtype=np.uint8)

        print(shelf.tx_begin([shelf.m,shelf.n]))

        for i in np.arange(0,10):
            shelf.n += 1
            shelf.m += 3
        
        print(shelf.items)
        shelf.inspect(verbose=False)
        shelf.tx_end()
        shelf.inspect(verbose=False)
        #shelf.erase('n')
        
if __name__ == '__main__':
    unittest.main()
