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
import gc
import chardet

from .metadata import *
from .memoryresource import MemoryResource
from .shelf import Shadow, ShelvedCommon

# create alias for Python bytes type
python_type_bytes = bytes

class bytes(Shadow):
    '''
    String object that is stored in the memory resource.
    '''
    def __init__(self, param0=None, encoding=None, errors=None, iterable_of_ints=None):
        if isinstance(param0, str):
            if errors == None:
                self.bytes_value = python_type_bytes(param0, encoding)
            else:
                self.bytes_value = python_type_bytes(param0, encoding, errors)
            self.encoding = encoding
        else:
            try:
                # check for iterable
                iter(param0)
                self.bytes_value = python_type_bytes(param0)
                self.encoding = None
                return
            except:
                raise RuntimeError("given bytes ctor parameters not handled")
            
        
    def make_instance(self, memory_resource: MemoryResource, name: str):
        '''
        Create a concrete instance from the shadow
        '''
        return shelved_bytes(memory_resource,
                             name,
                             bytes_value=self.bytes_value,
                             encoding=self.encoding)

    def existing_instance(memory_resource: MemoryResource, name: str):
        '''
        Determine if an persistent named memory object corresponds to this type
        '''                        
        buffer = memory_resource.get_named_memory(name)
        if buffer is None:
            return (False, None)

        # cast header structure on buffer
        hdr = construct_header_from_buffer(buffer)

        if (hdr.type == DataType_Bytes):

            data_subtype = hdr.subtype
            if data_subtype == DataSubType_Ascii:
                return (True, shelved_bytes(memory_resource, name, buffer[HeaderSize:], 'ascii'))
            elif data_subtype == DataSubType_Utf8:
                return (True, shelved_bytes(memory_resource, name, buffer[HeaderSize:], 'utf-8'))
            elif data_subtype == DataSubType_Utf16:
                return (True, shelved_bytes(memory_resource, name, buffer[HeaderSize:], 'utf-16'))
            elif data_subtype == DataSubType_Latin1:
                return (True, shelved_bytes(memory_resource, name, buffer[HeaderSize:], 'latin-1'))

        # not a bytes object
        return (False, None)

    def build_from_copy(memory_resource: MemoryResource, name: str, value):
        encoding = chardet.detect(value)['encoding']
        return shelved_bytes(memory_resource, name, bytes_value=value, encoding=encoding)


class shelved_bytes(ShelvedCommon):
    '''
    Shelved bytes with multiple encoding support
    '''
    def __init__(self, memory_resource, name, bytes_value, encoding):

        if not isinstance(name, str):
            raise RuntimeError("invalid name type")

        memref = memory_resource.open_named_memory(name)

        if memref == None:
            
            # create new value
            total_len = HeaderSize + len(bytes_value)
            memref = memory_resource.create_named_memory(name, total_len, 8, False)

            memref.tx_begin(value_named_memory=None, check=False)
            hdr = construct_header_on_buffer(memref.buffer, DataType_Bytes, txbits=TXBIT_DIRTY)
            
            if encoding == 'ascii':
                hdr.subtype = DataSubType_Ascii
            elif encoding == 'utf-8':
                hdr.subtype = DataSubType_Utf8
            elif encoding == 'utf-16':
                hdr.subtype = DataSubType_Utf16
            elif encoding == 'latin-1':
                hdr.subtype = DataSubType_Latin1
            else:
                hdr.subtype = DataSubType_Ascii
            
            # copy value into memory resource
            memref.buffer[HeaderSize:] = bytes_value
            memref.tx_commit()

        self.view = memoryview(memref.buffer[HeaderSize:])

        # hold a reference to the memory resource
        self._memory_resource = memory_resource
        self._metadata_named_memory = memref
        self._value_named_memory = None
        self.encoding = encoding
        self.name = name

    def __repr__(self):
        return python_type_bytes(self.view)

    def __str__(self):
        return str(python_type_bytes(self.view)) #,self.encoding)

    def __len__(self):
        return len(self.view)

    def __getitem__(self, key):
        '''
        Magic method for slice handling
        '''
        if isinstance(key, int):
            return int(self.view[key])
        if isinstance(key, slice):
            return python_type_bytes(self.view.__getitem__(key))
        else:
            raise TypeError

    def persist(self):
        '''
        Flush cache and persistent all value memory
        '''
        self._metadata_named_memory.persist()        
    
    def __getattr__(self, name):
        if name not in ("encoding"):
            raise AttributeError("'{}' object has no attribute '{}'".format(type(self),name))
        else:
            return self.__dict__[name]

    def __add__(self, value): # in-place append e.g, through +=

        # build a new value with different name, then swap & delete
        new_bytes = python_type_bytes(self.view.tobytes()).__add__(value)
        memory = self._memory_resource

        total_len = HeaderSize + len(new_bytes)
        memref = memory.create_named_memory(self.name + "-tmp", total_len, 8, False)

        hdr = init_header_from_buffer(memref.buffer)
        hdr.type = DataType_Bytes
        
        if self.encoding == 'ascii':
            hdr.subtype = DataSubType_Ascii
        elif self.encoding == 'utf-8':
            hdr.subtype = DataSubType_Utf8
        elif self.encoding == 'utf-16':
            hdr.subtype = DataSubType_Utf16
        elif self.encoding == 'latin-1':
            hdr.subtype = DataSubType_Latin1
        else:
            raise RuntimeError('shelved string does not recognize encoding {}'.format(self.encoding))
    
        
        # copy into memory resource
        memref.tx_begin() # don't need transaction here?
        memref.buffer[HeaderSize:] = new_bytes
        memref.tx_commit()

        del memref # this will force release
        del self._metadata_named_memory # this will force release
        gc.collect()

        # swap names
        memory.atomic_swap_names(self.name, self.name + "-tmp")

        # erase old data
        memory.erase_named_memory(self.name + "-tmp")

        # open new data
        memref = memory.open_named_memory(self.name)
        self._metadata_named_memory = memref
        self._value_named_memory = None
        self.view = memoryview(memref.buffer[HeaderSize:])
        return self

    def __eq__(self, value): # == operator
        return (python_type_bytes(self.view).__eq__(value))

    def __ge__(self, value): # == operator
        return (python_type_bytes(self.view).__ge__(value))

    def __gt__(self, value): # == operator
        return (python_type_bytes(self.view).__gt__(value))

    def __le__(self, value): # == operator
        return (python_type_bytes(self.view).__le__(value))

    def __lt__(self, value): # == operator
        return (python_type_bytes(self.view).__lt__(value))

    def __ne__(self, value): # == operator
        return (python_type_bytes(self.view).__ne__(value))

    def __contains__(self, value): # in / not in operator
        return (value in python_type_bytes(self.view))

    def capitalize(self):
        return python_type_bytes(self.view).capitalize()

    def center(self, width, fillchar=' '):
        return python_type_bytes(self.view).center(width, fillchar)
    
    def count(self, start=0, end=0):
        if end > 0:
            return python_type_bytes(self.view).count(start, end)
        else:
            return python_type_bytes(self.view).count(start)

    def decode(self, encoding='utf-8'):
        return python_type_bytes(self.view).decode(encoding)
        
    def encode(self, encoding='utf-8', errors='strict'):
        return python_type_bytes(self.view).encode(encoding, errors)

    def hex(self):
        return python_type_bytes(self.view).hex()

    def strip(self, bytes=None):
        return python_type_bytes(self.view).strip(bytes)

    def lstrip(self, bytes=None):
        return python_type_bytes(self.view).lstrip(bytes)

    def rstrip(self, bytes=None):
        return python_type_bytes(self.view).rstrip(bytes)
    
    


# bytes object methods
#'
# ['__add__', '__class__', '__contains__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getnewargs__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__mod__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__rmod__', '__rmul__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'capitalize', 'center', 'count', 'decode', 'endswith', 'expandtabs', 'find', 'fromhex', 'hex', 'index', 'isalnum', 'isalpha', 'isdigit', 'islower', 'isspace', 'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'maketrans', 'partition', 'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit', 'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase', 'title', 'translate', 'upper', 'zfill']
