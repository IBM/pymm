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

import pymmcore
import numpy as np

from .memoryresource import MemoryResource
from .shelf import Shadow
from .shelf import ShelvedCommon

class dictionary(Shadow):
    '''
    dictionary stored in memory resource
    '''
    def __init__(self, *args, **kwargs):
        self.args = []
        for arg in args:
            self.args.append(arg)

        self.kwargs = []
        for k, v in kwargs.items():
            self.kwargs.append((k,v))

    def make_instance(self, memory_resource: MemoryResource, name: str):
        '''
        Create a concrete instance from the shadow
        '''
        return shelved_dictionary(memory_resource, name, self.args, self.kwargs)


class shelved_dictionary(ShelvedCommon):
    def __new__(subtype, memory_resource, name, args, kwargs):

        if not isinstance(name, str):
            raise RuntimeException("invalid name type")
        
        root = memory_resource.open_named_memory(name)

        if root == None:
            # create new entry
            value_len = int(80 + len(name))
            root_memref = memory_resource.create_named_memory(name, value_len, 8, False)
            print("len-->", value_len)
            # prefix used for values corresponding to dictionary items
            member_prefix = "__dictionary_" + str(root_memref.addr()[0]) + "__" + name + "_"
            print(member_prefix)
        else:
            print("dictionary already exists!!")
    

#class dict(mapping, **kwarg)
#class dict(iterable, **kwarg)
    
