#!/usr/bin/python3 -m unittest
#
# testing mapstore backend
#
import unittest
import pymm
import numpy as np
import math

class TestMapstore(unittest.TestCase):
    def setUp(self):
        global force_new
        self.s = pymm.shelf('myShelf',size_mb=1024,backend="mapstore")
    
    def tearDown(self):
        del self.s
        
    def test_mapstore(self):
        self.s.x = pymm.ndarray((10,10,))
        self.s.x.fill(33)
        self.assertTrue(self.s.x[0][0] == 33)
        self.assertTrue(self.s.x[9][9] == 33)
        print(self.s.items)
        print(self.s.x)
        print(self.s.x._value_named_memory.addr())

if __name__ == '__main__':
    unittest.main()
