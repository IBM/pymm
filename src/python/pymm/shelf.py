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

#    Here is useful info for shelf type implementation:
#    https://rszalski.github.io/magicmethods/#representations

import pymmcore
import pymm
import gc
import sys
import copy
import numpy
import torch
import weakref
import numpy as np
import torch

from .metadata import *
from .memoryresource import MemoryResource
from .check import methodcheck

# globals
tx_vars = []

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def print_error(*args):
    print(colored(255,0,0,*args))

# common functions for shelved types
#

'''
Common superclass for shelved objects
'''
class ShelvedCommon:
    def __getattr__(self, name):
        if name == 'memory':
            return self._value_named_memory.addr()
        if name == 'namedmemory':
            return self._value_named_memory
            

class Shadow:
    '''
    Indicate type is a shadow type
    '''
    pass


class shelf():
    '''
    A shelf is a logical collection of variables held in CXL or persistent memory
    '''
    def __init__(self, name, pmem_path='/mnt/pmem0', size_mb=32, load_addr='0x900000000',
                 backend=None, mm_plugin=None, force_new=False, ):
        if not(isinstance(load_addr, str)):
            raise RuntimeError('shelf ctor parameter load_addr should be string')
        self.name = name
        self.mr = MemoryResource(name, size_mb, pmem_path=pmem_path, backend=backend,
                                 mm_plugin=mm_plugin, load_addr=load_addr, force_new=force_new)
        
        if self.mr == None:
            raise RuntimeError('shelf initialization failed')
        
        # todo iterate data and check value-metadata pairs
        items = self.mr.list_items()
        for varname in items:
            if not varname in self.__dict__:

                # read metadata header (always connect to name=key)
                buffer = self.mr.get_named_memory(varname)
                if buffer is None:
                    print("WARNING: no named memory for '{}'".format(varname))
                    continue
                    
                hdr = MetaHeader.from_buffer(memoryview(buffer))
                
                if (hdr.magic != int(HeaderMagic)):
                    print("WARNING: (shelf.py) bad magic {} number for variable: {}".format(hdr.magic,varname))
                    continue
                
                stype = hdr.type
                # call appropriate existing_instance for detected type
                
                # type: pymm.string
                if (stype == DataType_String):
                    (existing, value) = pymm.string.existing_instance(self.mr, varname)
                    if existing == True:
                        self.__dict__[varname] = value
                        print("Value '{}' has been made available on shelf '{}'!".format(varname, name))
                        continue

                elif (stype == DataType_Bytes):
                    (existing, value) = pymm.bytes.existing_instance(self.mr, varname)
                    if existing == True:
                        self.__dict__[varname] = value
                        print("Value '{}' has been made available on shelf '{}'!".format(varname, name))
                        continue
                    
                # type: pymm.ndarray
                elif (stype == DataType_NumPyArray):
                    (existing, value) = pymm.ndarray.existing_instance(self.mr, varname)
                    if existing == True:
                        self.__dict__[varname] = value
                        print("Value '{}' has been made available on shelf '{}'!".format(varname, name))
                        continue

                # type: pymm.torch_tensor
                elif (stype == DataType_TorchTensor):
                    (existing, value) = pymm.torch_tensor.existing_instance(self.mr, varname)
                    if existing == True:
                        self.__dict__[varname] = value
                        print("Value '{}' has been made available on shelf '{}'!".format(varname, name))
                        continue

                # type: pymm.dlpack_array
                elif (stype == DataType_DLTensor):
                    (existing, value) = pymm.dlpack_array.existing_instance(self.mr, varname)
                    if existing == True:
                        self.__dict__[varname] = value
                        print("Value '{}' has been made available on shelf '{}'!".format(varname, name))
                        continue                    

                # type: pymm.float_number
                elif (stype == DataType_NumberFloat):
                    (existing, value) = pymm.float_number.existing_instance(self.mr, varname)
                    if existing == True:
                        self.__dict__[varname] = value
                        print("Value '{}' has been made available on shelf '{}'!".format(varname, name))
                        continue

                # type: pymm.integer_number
                elif (stype == DataType_NumberInteger):
                    (existing, value) = pymm.integer_number.existing_instance(self.mr, varname)
                    if existing == True:
                        self.__dict__[varname] = value
                        print("Value '{}' has been made available on shelf '{}'!".format(varname, name))
                        continue
                    
                # type: pymm.linked_list (needs shelf)
                elif (stype == DataType_LinkedList):
                    (existing, value) = pymm.linked_list.existing_instance(self, varname)
                    if existing == True:
                        self.__dict__[varname] = value
                        print("Value '{}' has been made available on shelf '{}'!".format(varname, name))
                        continue
                    
                print("Value '{}' is unknown type!".format(varname))


    def __del__(self):
        gc.collect()
        for i in self.__dict__:
            if i == 'mr':
                continue
            this_id = id(self.__dict__[i])
            for j in gc.get_objects():                
                if id(j) == this_id:
                    if sys.getrefcount(j) > 4:
                        #raise RuntimeError('trying to delete shelf with outstanding references to contents')
                        print_error('WARNING: deleting shelf with outstanding reference {}-->>{}'.format(i,sys.getrefcount(j)))
        

    def __setattr__(self, name, value):
        '''
        Handle attribute assignment
        '''
        # constant members
        if self.mr and name == 'mr':
            raise RuntimeError('invalid assignment')

        # selective pass through
        if name in ['items','name','mr']:
            self.__dict__[name] = value
            return
            
        if name in self.__dict__:
            if issubclass(type(value), pymm.ShelvedCommon):
                # this happens when an in-place __iadd__ or like operation occurs.  I'm not
                # quite sure why it happens?
                return
            elif name == 'name' or name == 'mr':
                raise RuntimeError('cannot change shelf attribute')

        # currently we allow reassignment of shelf variables
        # TODO: we might want to make a back up of it then later delete
        if name in self.__dict__:
            # PyTorch: check for assignment of own's view
            if isinstance(self.__dict__[name], torch.Tensor) and isinstance(value, torch.Tensor):
                # check if this is just a view on the same object
                if value.data_ptr() == self.__dict__[name].data_ptr():
                    raise RuntimeError('cannot reassign self-referring view: use clone')

            gc.collect()
            self.erase(name)
            
        # check for supported shadow types
        if self._is_supported_shadow_type(value):
            self.__dict__[name] = value.make_instance(self.mr, name)
            print("made instance '{}' on shelf".format(name))
        elif isinstance(value, pymm.linked_list):
            # pass shelf itself as param
            self.__dict__[name] = value.make_instance(self, name)
            print("made instance '{}' on shelf".format(name))
        elif isinstance(value, numpy.ndarray): # perform a copy instantiation (ndarray)
            self.__dict__[name] = pymm.ndarray.build_from_copy(self.mr, name, value)
            print("made ndarray instance from copy '{}' on shelf".format(name))
        elif isinstance(value, torch.Tensor): # perform a copy instantiation (ndarray)
            self.__dict__[name] = pymm.torch_tensor.build_from_copy(self.mr, name, value)
            print("made torch_tensor instance from copy '{}' on shelf".format(name))
        elif isinstance(value, str):
            self.__dict__[name] = pymm.string.build_from_copy(self.mr, name, value)
        elif isinstance(value, float):
            self.__dict__[name] = pymm.float_number.build_from_copy(self.mr, name, value)
        elif isinstance(value, int):
            self.__dict__[name] = pymm.integer_number.build_from_copy(self.mr, name, value)
        elif isinstance(value, bytes):
            self.__dict__[name] = pymm.bytes.build_from_copy(self.mr, name, value)
        elif issubclass(type(value), pymm.ShelvedCommon):
            raise RuntimeError('persistent reference not yet supported - use a volatile one!')            
        elif type(value) == type(None):
            pass
        else:
            raise RuntimeError('non-pymm type ({}, {}) cannot be put on the shelf'.format(name,type(value)))

    def __getattr__(self, name):
        '''
        Handle attribute access
        '''
        if name == 'items':
            return self.get_item_names()
        elif name == 'used':
            return self.mr.get_percent_used()
        if name in ['mr']:
            if not name in self.__dict__:
                return None
            return self.__dict__[name]
        if not name in self.__dict__:
            return None
            #raise RuntimeError('invalid member {}'.format(name))
        else:
            return self.__dict__[name]
#            return weakref.ref(self.__dict__[name])
            
    @methodcheck(types=[])
    def get_item_names(self, all=False):
        '''
        Get names of items on the shelf
        '''
        items = []
        for s in self.__dict__:
            if s == 'name' or s == 'mr':
                continue
            if all==False and s[0] == '-':
                continue
            items.append(s)
                
                
        return items
            
        
    @methodcheck(types=[str])
    def erase(self, name):
        '''
        Erase and remove variable from the shelf
        '''
        # check the thing we are trying to erase is on the shelf
        if not name in self.__dict__:
            raise RuntimeError('attempting to erase something that is not on the shelf')

        # sanity check
        if gc.is_tracked(self.__dict__[name]):
            gc.collect() # force gc        
            count = sys.getrefcount(self.__dict__[name])
            if count != 2: 
                raise RuntimeError('erase failed due to outstanding references ({})'.format(count))
            
            self.__dict__.pop(name)
            gc.collect() # force gc
        
        # then remove the named memory from the store
        self.mr.erase_named_memory(name)


    @methodcheck(types=[])
    def get_percent_used(self):
        return self.mr.get_percent_used()

    @methodcheck(types=[])
    def persist(self):
        '''
        Persist memory for all variables on the shelf
        '''
        for varname in self.get_item_names():
            # get memory resource associated with variable
            metadata_memory_resource = self.__dict__[varname]._metadata_named_memory
            if self.__dict__[varname]._metadata_named_memory == None:
                print('error: var {} has no metadata'.format(varname))
                continue
            else:
                metadata_memory_resource.persist()
            value_memory_resource = self.__dict__[varname]._value_named_memory
            if value_memory_resource != None:
                value_memory_resource.persist()
        


    @methodcheck(types=[bool])
    def inspect(self, verbose=False):
        '''
        Inspect variables on shelf
        '''
        for varname in self.get_item_names(all=verbose):

            # get header info
            if self.__dict__[varname]._metadata_named_memory == None:
                print('error: var {} has no metadata'.format(varname))
                continue
            
            metadata_memory = self.__dict__[varname]._metadata_named_memory.buffer

            try:
                hdr = construct_header_from_buffer(metadata_memory)

                print('var {}: addr={} magic={} type={} subtype={} txbits={} ver={} refcnt={}'
                      .format(varname,
                              self.__dict__[varname]._metadata_named_memory.addr(),
                              hex(hdr.magic),
                              hdr.type,
                              hdr.subtype,
                              hdr.txbits,
                              hdr.version,
                              hdr.refcnt))

                if verbose:
                    print('var {}: {}'.format(varname, bytes(metadata_memory[:HeaderSize])))
                    
            except RuntimeError:
                print('var {}: failed header check!! (buffer_len={})'.format(varname, len(metadata_memory)))
                print('var {}: {}'.format(varname, bytes(metadata_memory)))
                pass

            value_memory = self.__dict__[varname]._value_named_memory
            if value_memory != None:
                print('var {} has value memory {}'.format(varname, value_memory.addr()))
            

    def _is_supported_shadow_type(self, value):
        '''
        Helper function to return True if value is of a shadow type
        '''
        return (isinstance(value, pymm.ndarray) or
                isinstance(value, pymm.dlpack_array) or
                isinstance(value, pymm.torch_tensor) or
                isinstance(value, pymm.string) or
                isinstance(value, pymm.float_number) or
                isinstance(value, pymm.integer_number) or
                isinstance(value, pymm.bytes)
        )

    def supported_types(self):
        '''
        Report types that are supported by the shelf
        '''
        return ["pymm.ndarray", "pymm.torch_tensor", "pymm.string", "pymm.float_number", "pymm.integer_number",
                "pymm.bytes"]


    def tx_begin(self):
        '''
        Begin shelf-wide transaction. Currently new values created on the shelf are not included
        as part of the transaction.
        '''        
        # iterate through shelf variables and set TXBIT_MULTIVAR
        global tx_vars
        for varname in self.get_item_names(all=False):

            # get header info
            if self.__dict__[varname]._metadata_named_memory == None:
                raise RuntimeError('var {} has no metadata'.format(varname))
            
            metadata_memory = self.__dict__[varname]._metadata_named_memory.buffer

            hdr = construct_header_from_buffer(metadata_memory)
            metadata_set_tx_bit(hdr, TXBIT_MULTIVAR)
            tx_vars.append(varname)
                    
        return vars

    def tx_end(self):
        '''
        Commit shelf-wide transaction.
        '''
        global tx_vars
        for varname in tx_vars:
            entry = self.__dict__[varname]
            metadata = entry._metadata_named_memory
            assert metadata != None
            metadata.tx_multivar_commit(entry._value_named_memory)
        tx_vars = []

