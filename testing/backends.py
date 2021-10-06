#!/usr/bin/python3 -m unittest
#
# testing for different backends
#
import unittest
import pymm
import numpy as np
import math
import torch
import os

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def print_error(*args):
    print(colored(255,0,0,*args))

def log(*args):
    print(colored(0,255,255,*args))

class TestBackends(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self,*args,**kwargs)

        # Find first available /mnt/pmem<n> directory
        self.pmem_root=''
        for i in range(0, 2):
            root='/mnt/pmem%d' % (i,)
            if os.path.isdir(root):
                self.pmem_root=root
                break
        self.assertNotEqual(self.pmem_root, '')

    def test_default(self):
        log("Running shelf with default backend ...")
        s = pymm.shelf('myShelf',size_mb=8,pmem_path=self.pmem_root,force_new=True)
        s.items
        log("OK!")

    def test_dram_mapstore(self):
        log("Running shelf with mapstore and default MM plugin ...")
        s = pymm.shelf('myShelf2',size_mb=8,backend='mapstore') # note, no force_new or pmem_path
        s.x = pymm.ndarray((10,10,))
        print(s.items)
        log("OK!")

    def test_dram_mapstore_jemalloc(self):
        log("Running shelf with mapstore and jemalloc MM plugin ...")
        s = pymm.shelf('myShelf3',size_mb=8,backend='mapstore',mm_plugin='libmm-plugin-jemalloc.so')
        s.x = pymm.ndarray((10,10,))
        s.y = pymm.ndarray((100,100,))
        print(s.items)
        log("OK!")

    def test_dram_mapstore_rcalb(self):
        log("Running shelf with mapstore and rcalb MM plugin ...")
        s = pymm.shelf('myShelf4',size_mb=8,backend='mapstore',mm_plugin='libmm-plugin-rcalb.so')
        s.x = pymm.ndarray((10,10,))
        s.y = pymm.ndarray((100,100,))
        print(s.items)
        log("OK!")

    def test_hstore_cc(self):
        log("Running shelf with hstore-cc ...")
        pmem_path="%s/test_hstore_cc" % (self.pmem_root,)
        os.system("rm -Rf %s" % (pmem_path,))
        os.mkdir(pmem_path)
        s = pymm.shelf('myShelf5',size_mb=8,backend='hstore-cc',pmem_path=pmem_path)
        s.x = pymm.ndarray((10,10,))
        s.y = pymm.ndarray((100,100,))
        print(s.items)
        log("OK!")

    def test_hstore_mm_default(self):
        log("Running shelf with hstore-mm and default MM plugin ...")
        pmem_path="%s/test_hstore_mm_default" % (self.pmem_root,)
        os.system("rm -Rf %s" % (pmem_path,))
        os.mkdir(pmem_path)
        s = pymm.shelf('myShelf_mm_default',size_mb=8,backend='hstore-mm',pmem_path=pmem_path)
        s.x = pymm.ndarray((10,10,))
        s.y = pymm.ndarray((100,100,))
        print(s.items)
        log("OK!")

    def test_hstore_mm_jemalloc(self):
        log("Running shelf with hstore-mm and jemalloc MM plugin ...")
        pmem_path="%s/test_hstore_mm_jemalloc" % (self.pmem_root,)
        os.system("rm -Rf %s" % (pmem_path,))
        os.mkdir(pmem_path)
        s = pymm.shelf('myShelf7',size_mb=8,backend='hstore-mm',pmem_path=pmem_path,mm_plugin='libmm-plugin-jemalloc.so')
        s.x = pymm.ndarray((10,10,))
        s.y = pymm.ndarray((100,100,))
        print(s.items)
        log("OK!")        

    def test_hstore_mm_rcalb(self):
        log("Running shelf with hstore-mm and rcalb MM plugin ...")
        pmem_path="%s/test_hstore_mm_rcalb" % (self.pmem_root,)
        os.system("rm -Rf %s" % (pmem_path,))
        os.mkdir(pmem_path)
        s = pymm.shelf('myShelf6',size_mb=8,backend='hstore-mm',pmem_path=pmem_path,mm_plugin='libmm-plugin-rcalb.so')
        s.x = pymm.ndarray((10,10,))
        s.y = pymm.ndarray((100,100,))
        print(s.items)
        log("OK!")


if __name__ == '__main__':
    unittest.main()
