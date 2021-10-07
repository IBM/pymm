# 
#    Copyright [2021] [IBM Corporation]
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#        http://www.apache.org/licenses/LICENSE-2.0
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
import pymm
import pymmcore
import numpy as np

from .pymmtest import *
from .ndarray import ndarray
from .dlpack_array import dlpack_array
from .torch_tensor import torch_tensor
from .string import string
from .float_number import float_number
from .integer_number import integer_number
from .bytes import bytes
from .linkedlist import linked_list


from .shelf import shelf
from .shelf import ShelvedCommon
from .memoryresource import MemoryResource

from .demo import demo

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def print_warning(*args):
    print(colored(255,0,0,*args))

def enable_transient_memory(backing_directory='/tmp', pmem_file=None, pmem_file_size_gb=0):
    # check that the modified version of numpy is loaded
    if np.__version__ != '1.19.6.dev0+78b5f9b':
        print_warning('[WARNING]: unable to detect modified NumPy installation; transient memory is not enable')
        return
    if (pmem_file == None or pmem_file_size_gb == 0):
        pymm.pymmcore.enable_transient_memory(backing_directory=backing_directory)
    else:
        pymm.pymmcore.enable_transient_memory(backing_directory=backing_directory,pmem_file=pmem_file,pmem_file_size_gb=pmem_file_size_gb)
    
def disable_transient_memory():
    pymm.pymmcore.disable_transient_memory()

