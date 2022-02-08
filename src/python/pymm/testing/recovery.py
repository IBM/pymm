#!/usr/bin/python3 -m unittest
#
# testing recovery
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

def fail(*args):
    print(colored(255,0,0,*args))

def check(expr, msg):
    if expr == True:
        msg = "Check: " + msg + " OK"
        log(msg)
    else:
        msg = "Check: " + msg + " FAILED"
        fail(msg)


class TestRecovery(unittest.TestCase):

    def test_establish(self):
        log("Testing: establishing shelf and values")
        shelf = pymm.shelf('myShelfRecovery',size_mb=1024,pmem_path='/mnt/pmem0',force_new=True)

        # different types
        shelf.s = 'Hello'
        shelf.s += ' world!'
        print(list('Hello'))
        shelf.f = 1.123
        shelf.fm = 2.2
        shelf.fm += 1.1
        shelf.i = 911
        shelf.im = 900
        shelf.im += 99
        shelf.nd = np.ones((3,))
        shelf.nd2 = np.identity(10)
        shelf.b = b'This is a bytes type'
        shelf.bm = b'This is a '
        shelf.bm += b'modified bytes type'
        shelf.t = torch.tensor([[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3]])
        shelf.l = pymm.linked_list()
        shelf.l.append(1)
        shelf.l.append(2)
        shelf.l.append('z')
        
        del shelf
        gc.collect()

    def test_recovery(self):
        log("Testing: recovering shelf and values")
        shelf = pymm.shelf('myShelfRecovery',pmem_path='/mnt/pmem0',force_new=False)

        print(">{}<".format(shelf.s))
        print(shelf.f)
        print(shelf.fm)
        print(shelf.i)
        print(shelf.im)
        print(shelf.nd)
        print(shelf.nd2)
        print(shelf.b)
        print(shelf.bm)
        print(list(shelf.s))
        print(round(shelf.fm,2))
        print(shelf.t)
        print(shelf.l)
                
        check(shelf.s == 'Hello world!', 'string recovery')
        check(shelf.f == 1.123, 'float recovery')
        check(round(shelf.fm,2) == 3.30, 'float modified recovery')
        check(shelf.i == 911, 'integer recovery')
        check(shelf.im == 999, 'integer modified recovery')
        check(np.array_equal(shelf.nd, np.ones((3,))),'1D ndarray')
        check(np.array_equal(shelf.nd2, np.identity(10)),'2D ndarray')
        check(shelf.b == b'This is a bytes type', 'bytes')
        check(shelf.bm == b'This is a modified bytes type', 'modified bytes')
        check(shelf.t.equal(torch.tensor([[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3]])), 'torch tensor')
        check(shelf.l[0] == 1, 'linked list')
        check(shelf.l[1] == 2, 'linked list')
        check(shelf.l[2] == 'z', 'linked list')

        print(shelf.items)
        shelf.inspect(verbose=True)
        
if __name__ == '__main__':
    unittest.main()
