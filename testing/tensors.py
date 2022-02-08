#!/usr/bin/python3 -m unittest
#
# testing transient memory (needs modified Numpy)
#
import unittest
import pymm
import numpy as np
import torch

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def log(*args):
    print(colored(0,255,255,*args))

shelf = pymm.shelf('myShelf',size_mb=128,pmem_path='/mnt/pmem0',force_new=True)

class TestTensors(unittest.TestCase):

    def test_torch_tensor_shadow_list(self):
        log("Testing: tensor ctor")
        shelf.x = pymm.torch_tensor([1,1,1,1,1])
        print(shelf.x)

    def test_torch_tensor_shadow_ndarray_A(self):
        log("Testing: tensor ctor")
        shelf.y = pymm.torch_tensor(np.arange(0,10))
        shelf.y.fill(-1.2)
        print(shelf.y)

    def test_torch_tensor_shadow_ndarray_B(self):
        log("Testing: tensor ctor")
        print(shelf.y)
        shelf.y = pymm.torch_tensor(np.arange(0,10))
        shelf.y.fill(-1.3)
        print(shelf.y)

    def test_torch_tensor_copy(self):
        log("Testing: tensor copy")
        T = torch.tensor([[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3]])
        U = torch.tensor([[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3]])
        print(T.shape)
        shelf.x = T
        print(shelf.x)
        self.assertTrue(shelf.x.equal(T))
        Q = torch.tensor([[1,2,3],[4,5,6]])

    def test_torch_ones(self):
        log("Testing: torch ones")
        shelf.x = torch.ones([3,5],dtype=torch.float64)
        print(shelf.x)
        shelf.x += 0.5
        print(shelf.x)

    def test_torch_leaf(self):
        log("Testing: torch tensor leaf")
        shelf.x = torch.randn(1, 1)
        self.assertTrue(shelf.x.is_leaf)

    def test_torch_zerodim_shadow(self):
        log("Testing: zero dim shadow")
        shelf.x = pymm.torch_tensor(1.0)
        self.assertTrue(shelf.x.dim() == 0)
        print(type(shelf.x))
        self.assertTrue(str(type(shelf.x)) == "<class 'pymm.torch_tensor.shelved_torch_tensor'>")

    def test_torch_zerodim(self):
        log("Testing: zero dim copy")
        shelf.y = torch.tensor(2.0)
        self.assertTrue(shelf.y.dim() == 0)
        self.assertTrue(str(type(shelf.y)) == "<class 'pymm.torch_tensor.shelved_torch_tensor'>")

    # NOT SUPPORTED
    def NORUN_test_torch_require_grad(self):
        log("Testing: requires_grad= param")
        shelf.x = torch.tensor(1.0, requires_grad = True)
        shelf.z = shelf.x ** 3
        shelf.z.backward() #Computes the gradient 
        print(shelf.x.grad.data) #Prints '3' which is dz/dx 
        
    def test_torch_tensor(self):
        log("Testing: torch_tensor")
        n = torch.Tensor(np.arange(0,1000))
        shelf.t = torch.Tensor(np.arange(0,1000)) #pymm.torch_tensor(n)

        # shelf type S
        self.assertTrue(str(type(shelf.t)) == "<class 'pymm.torch_tensor.shelved_torch_tensor'>")

        log("Testing: torch_tensor sum={}".format(sum(shelf.t)))        
        self.assertTrue(shelf.t.sum() == 499500)

        slice_sum = sum(shelf.t[10:20])
        log("Testing: torch_tensor slice sum={}".format(slice_sum))
        self.assertTrue(slice_sum == 145)

        # shelf type S after in-place operation
        self.assertTrue(str(type(shelf.t)) == "<class 'pymm.torch_tensor.shelved_torch_tensor'>")
        
        # shelf type S * NS (non-shelf type)
        self.assertTrue(str(type(shelf.t * n)) == "<class 'torch.Tensor'>")
        
        # shelf type NS * S
        self.assertTrue(str(type(n * shelf.t)) == "<class 'torch.Tensor'>")

        # shelf type S * shelf type S
        self.assertTrue(str(type(shelf.t * shelf.t)) == "<class 'torch.Tensor'>")
    
        shelf.t += 1
        shelf.t *= 2
        shelf.t -= 0.4
        shelf.t /= 2

        shelf.erase('t')

    def test_torch_reassign(self):
        log("Testing: torch_tensor reassign")
        shelf.c = torch.tensor([[1, 2, 3], [4, 5, 6], [9,10,11]])
        print(shelf.c)
        shelf.c = shelf.c.clone().view(9,-1)        
        print(shelf.c)
        with self.assertRaises(RuntimeError):
            shelf.c = shelf.c.view(9,-1)

if __name__ == '__main__':
    unittest.main()
