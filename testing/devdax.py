#!/usr/bin/python3 -m unittest
#
# test for devdax and hstore
#
# first run ...
# DAX_RESET=1 python3 ~/mcas/src/python/pymm/testing/devdax.py
#
# then re-run ..(as many times as you want, vars will increment)
#
# python3 ~/mcas/src/python/pymm/testing/devdax.py
#
import os
import unittest
import pymm
import numpy as np

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def log(*args):
    print(colored(0,255,255,*args))

class TestDevDaxSupport(unittest.TestCase):

    def setUp(self):
        self.s = pymm.shelf('myShelf',size_mb=1024,backend='hstore-cc',pmem_path='/dev/dax1.0',force_new=True)
        print(self.s.items)

    def tearDown(self):
        del self.s

    def test_check_content(self):
        if 'x' in self.s.items:
            log('x is there: value={}'.format(self.s.x))
            self.s.x += 1.1
        else:
            log('x is not there!')
            self.s.x = 1.0

        if 'y' in self.s.items:
            log('y is there:')
            log(self.s.y)
            self.s.y += 1
        else:
            log('y is not there!')
            self.s.y = pymm.ndarray((10,10,),dtype=np.uint32)
            self.s.y.fill(0)

        print(self.s.items)

        
if __name__ == '__main__':
    if os.path.isdir('/dev/dax1.0'):
        unittest.main()
    else:
        print('Omitting test; no /dev/dax1.0 detected')
