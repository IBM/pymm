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
import struct
import gc
import numpy as np

from .metadata import *
from .memoryresource import MemoryResource
from .shelf import Shadow, ShelvedCommon
from .float_number import float_number
from .check import methodcheck

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

# TODO
# __reversed__(self)
#     Called to implement behavior for the reversed() built in function. Should return a reversed version of the sequence. Implement this only if the sequence class is ordered, like list or tuple.
# __contains__(self, item)
#     __contains__ defines behavior for membership tests using in and not in. Why isn't this part of a sequence protocol, you ask? Because when __contains__ isn't defined, Python just iterates over the sequence and returns True if it comes across the item it's looking for.
# __missing__(self, key)
#     __missing__ is used in subclasses of dict. It defines behavior for whenever a key is accessed that does not exist in a dictionary (so, for instance, if I had a dictionary d and said d["george"] when "george" is not a key in the dict, d.__missing__("george") would be called).

class linked_list(Shadow):
    '''
    Floating point number that is stored in the memory resource. Uses value cache
    '''
    def __init__(self):
        print(colored(255,0,0, 'WARNING: linked_list is experimental and unstable!'))

    def make_instance(self, shelf, name: str):
        '''
        Create a concrete instance from the shadow
        '''
        return shelved_linked_list(shelf, name)

    def existing_instance(shelf, name: str):
        '''
        Determine if an persistent named memory object corresponds to this type
        '''
        memory_resource = shelf.mr
        buffer = memory_resource.get_named_memory(name)
        if buffer is None:
            raise RuntimeError('bad object name')

        # cast header structure on buffer
        hdr = construct_header_from_buffer(buffer)

        if (hdr.type == DataType_LinkedList):
            return (True, shelved_linked_list(shelf, name))

        # not a string
        return (False, None)

    def build_from_copy(memory_resource: MemoryResource, name: str, value):
        raise RuntimeError("not implemented!")
#        return shelved_linked_list(memory_resource, name, value=value)

MEMORY_INCREMENT_SIZE=128*1024

class shelved_linked_list(ShelvedCommon):
    '''
    Shelved floating point number
    '''
    def __init__(self, shelf, name):

        self._shelf = shelf # retain reference to shelf
        self._tag = 0
        
        memory_resource = shelf.mr
        memref = memory_resource.open_named_memory(name)

        if memref == None:

            # create metadata (data is separate)
            memref = memory_resource.create_named_memory(name, HeaderSize, 1, False)

            # create value memory
            self._value_named_memory = memory_resource.create_named_memory(name + '-value',
                                                                           MEMORY_INCREMENT_SIZE, 64, False)


            memref.tx_begin(value_named_memory=self._value_named_memory, check=False)
            hdr = construct_header_on_buffer(memref.buffer, DataType_LinkedList, txbits=TXBIT_DIRTY)            
            memref.tx_commit()

            self._metadata_named_memory = memref

            self._internal = pymmcore.List(buffer=self._value_named_memory.buffer, rehydrate=False)

        else:
            # validates
            construct_header_from_buffer(memref.buffer)

            self._metadata_named_memory = memref
            # rehydrate internal structure
            self._value_named_memory = memory_resource.open_named_memory(name + '-value')
            self._internal = pymmcore.List(buffer=self._value_named_memory.buffer, rehydrate=True)

        # save name
        self.name = name

        
    def append(self, element):
        '''
        Add element to end of list
        '''
        if issubclass(type(element), ShelvedCommon): # implies it already on the shelf
            return self._internal.append(element=None, name=element.name)
        elif (isinstance(element, float) or isinstance(element, int)): # inline value
            return self._internal.append(element)
        elif (isinstance(element, np.ndarray) or
              isinstance(element, str)):
            # use shelf to store value
            self._tag += 1
            tag = self._tag
            name = '_' + self.name + '_' + str(tag)
            self._shelf.__setattr__(name, element) # like saying shelf.name = element
            return self._internal.append(element=None, tag=tag)

        raise RuntimeError('unhandled type')

    @methodcheck(types=[int])
    def __erase_tagged_object__(self, tag):
        '''
        Erase a tagged object from the shelf (iff tag > 0)
        '''
        if isinstance(tag, int):
            if tag > 0:
                name = self.__build_tagged_object_name__(tag)
                print("Erasing object (it was overwritten) :{}".format(name))
                self._shelf.erase(name)

    @methodcheck(types=[int])
    def __build_tagged_object_name__(self, tag):
        '''
        Generate name of a tagged object
        '''
        return '_' + self.name + '_' + str(tag)
                    

    def __len__(self):
        '''
        Return number of elements in list
        '''
        return self._internal.size()

    
    def __getitem__(self, item):
        '''
        Magic method for [] index access
        '''
        if isinstance(item, int):
            value, item_in_index = self._internal.getitem(item)
            if item_in_index: # reference to item in index
                name = self.__build_tagged_object_name__(value)
                try:
                    return self._shelf.__getattr__(name)
                except RuntimeError:
                    raise RuntimeError('list member is missing from main index; did it get deleted?')
            return value
        else:
            raise RuntimeError('slice criteria not supported')

    def __setitem__(self, item, value):
        '''
        Magic method for item assignment
        '''
        if isinstance(item, int):

            if issubclass(type(value), ShelvedCommon): # implies it already on the shelf
                rc = self._internal.setitem(item=item, value=None, name=element.name)
            elif (isinstance(value, float) or isinstance(value, int)): # inline value
                rc = self._internal.setitem(item=item, value=value)
            elif (isinstance(value, np.ndarray) or # other items that will be created implicitly as shelf items
                  isinstance(value, str)):
                # use shelf to store value
                self._tag += 1
                tag = self._tag
                name = '_' + self.name + '_' + str(tag)
                self._shelf.__setattr__(name, value) # like saying shelf.name = element
                rc = self._internal.setitem(item=item, value=None, tag=tag)
            else:
                raise RuntimeError('unhandled type')

            self.__erase_tagged_object__(rc)
            return
        else:
            raise RuntimeError('slice criteria not supported')

        print(key)
        
    def __iter__(self):
        '''
        Iterator method
        '''
        self.__iterpos = 0
        return self

    def __next__(self):
        '''
        Iterator next method
        '''
        if self.__iterpos >= self.__len__():
            raise StopIteration
        result = self.__getitem__(self.__iterpos)
        self.__iterpos += 1
        return result

    @methodcheck(types=[int])
    def __delitem__(self, item):
        '''
        Magic method for del self[key] operations
        '''
        rc = self._internal.delitem(item)
        self.__erase_tagged_object__(rc)
        return

    def __str__(self):
        '''
        Produce human-readable output
        '''
        return str(list(self))

    def __repr__(self):
        '''
        Produce machine readable output
        '''
        return str(list(self))

        
              
        

