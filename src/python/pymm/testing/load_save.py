#!/usr/bin/python3 -m unittest
#
# testing matrix load/save from/to shelf
#
import unittest
import pymm
import numpy as np
import gc

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def print_error(*args):
    print(colored(255,0,0,*args))

def log(*args):
    print(colored(0,255,255,*args))


force_new=True
class TestLoadSave(unittest.TestCase):
    def setUp(self):
        global force_new
        self.s = pymm.shelf('myShelf',size_mb=1024,backend="hstore-cc",force_new=force_new)
        force_new=False

    def tearDown(self):
        del self.s
    
    def test_A_save(self):
        # create a large right-hand side expression which will be evaluated in pmem transient memory
        self.s.w = np.ndarray((10000),dtype=np.uint8) # 1GB
        self.s.w.fill(1)
        np.save("savedSnapshot", self.s.w)

    def test_B_modify(self):
        # do something that needs undoing
        self.s.w.fill(3)

    def test_C_load(self):
        # undo
        self.s.w = np.load("savedSnapshot.npy")
        self.assertTrue(np.all((self.s.w == 1)))

        
if __name__ == '__main__':
    unittest.main()
