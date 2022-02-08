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

class TestBytes(unittest.TestCase):

    def test_bytes(self):
        log("Testing: pymm.bytes shadow type")
        shelf.x = pymm.bytes('hello world','utf-8')
        print(shelf.x)
        shelf.x += b'!  '
        print(shelf.x)
        print(shelf.x.decode())
        self.assertTrue(shelf.x.decode() == 'hello world!  ')

        log("Testing: pymm.bytes iterable ctor")
        shelf.z = pymm.bytes(range(3))
        self.assertTrue(shelf.z == b'\x00\x01\x02')
        print(shelf.z)
        
        log("Testing: pymm.bytes methods")
        print(shelf.x.capitalize())
        print(shelf.x.hex())
        print(shelf.x.strip())
        shelf.y = shelf.x.strip()
        print(shelf.y.decode())
        self.assertTrue(shelf.y.decode() == 'hello world!')

        shelf.a = b'El ni\xc3\xb1o come camar\xc3\xb3n'
        print(shelf.a.decode())

        shelf.b =  b'\xd8\xe1\xb7\xeb\xa8\xe5 \xd2\xb7\xe1'
        print(shelf.b.decode('cp855'))        

        shelf.c = ord(b'n')
        print(shelf.c)

        log("Testing: pymm.bytes list conversion")
        shelf.d = bytes([72, 101, 108, 108, 111])
        print(shelf.d)
        print(str(shelf.d))

        log("Testing: pymm.bytes slice single element access")
        self.assertTrue(shelf.d[1] == 101)
        self.assertTrue(isinstance(shelf.d[1], int))
        self.assertTrue(chr(shelf.d[1]) == 'e')
        
        print(type(shelf.d[1:3]))
        self.assertTrue(isinstance(shelf.d[1:3], bytes))
        self.assertTrue(list(shelf.d[1:3]) == [101,108])
        
        log("Testing: pymm.bytes magic methods")
        print(str(shelf.d))

        log("Garbage collecting...")
        gc.collect()
        



if __name__ == '__main__':
    unittest.main()
    del shelf
