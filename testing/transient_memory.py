#!/usr/bin/python3 -m unittest
#
# testing transient memory (needs modified Numpy)
#
import unittest
import pymm
import numpy as np

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def log(*args):
    print(colored(0,255,255,*args))
    

force_new=True
class TestTransientMemory(unittest.TestCase):
    def setUp(self):
        global force_new
        self.s = pymm.shelf('myShelf',size_mb=1024,pmem_path='/mnt/pmem0',force_new=force_new)
        pymm.pymmcore.enable_transient_memory(backing_directory='/tmp', pmem_file='/mnt/pmem0/swap', pmem_file_size_gb=1)

    def test_tm(self):
        self.s.x = np.ndarray((1000000,), dtype=np.uint8) # large RHS eval
        print(self.s.x)
        print(colored(255,255,255,"OK!"))
    
    def tearDown(self):
        del self.s
        pymm.pymmcore.disable_transient_memory()
    

if __name__ == '__main__':
    unittest.main()
