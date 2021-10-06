#!/usr/bin/python3 -m unittest
#
# basic base type test
#
import unittest
import pymm
import numpy as np
import math

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def log(*args):
    print(colored(0,255,255,*args))
    

force_new=True

class TestBaseTypes(unittest.TestCase):
    def setUp(self):
        global force_new
        self.s = pymm.shelf('myShelf',size_mb=1024,pmem_path='/mnt/pmem0',force_new=force_new)
        force_new=False

    def tearDown(self):
        del self.s
        
    def test_string(self):

        log("Testing: string")
        self.s.x = pymm.string("Hello world!")
        self.assertTrue(str(self.s.x) == "Hello world!")

        self.s.y = "Good Day sir!"
        self.assertTrue(self.s.y == "Good Day sir!")

        self.assertTrue("Day" in self.s.y)
        self.assertFalse("foobar" in self.s.y)

        self.assertTrue(self.s.y.capitalize() == "Good day sir!")

        self.assertTrue(self.s.x.count("world") == 1)

        self.s.x += " Brilliant!"
        print("Modified string >{}<".format(self.s.x))
        
        log("Testing: string OK!")

        
    def test_float_number(self):    
        log("Testing: float number")
        
        self.s.n = pymm.float_number(700.001)
        
        # in-place ops
        self.s.n *= 2
        self.s.n /= 2
    
        print(self.s.n)
        print(self.s.n * 2)
        print(self.s.n * 1.1)
    
        print(2 * self.s.n)
        print(1.1 * self.s.n)

        print(self.s.n / 2.11)
        print(2.11 / self.s.n)
        
        print(self.s.n // 2.11)
        print(2.11 // self.s.n)

        print(self.s.n - 2.11)
        print(2.11 - self.s.n)
    
        self.assertTrue(self.s.n * 2.0 == 1400.002)

        # from implicit cast
        self.s.m = 700.001
        self.assertTrue(self.s.m == self.s.n)
    
        log("Testing: number OK!")

        
    def Xtest_integer_number(self):    
        log("Testing: integer number")
        
        self.s.n = pymm.integer_number(700)
        print(self.s.n)
    
        # in-place ops
        self.s.n *= 2
        print(self.s.n)

        x = self.s.n * 2.1111
        print(x)
        self.assertTrue(x == 2955.54)

        self.s.n += 2
        print(self.s.n)
        self.s.n /= 2
        print(self.s.n)
    
        log("Testing: integer number OK!")


if __name__ == '__main__':
    unittest.main()

