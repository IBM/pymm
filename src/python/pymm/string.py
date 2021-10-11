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

from .metadata import *
from .memoryresource import MemoryResource
from .shelf import Shadow, ShelvedCommon

class string(Shadow):
    '''
    String object that is stored in the memory resource.
    '''
    def __init__(self, string_value, encoding='utf-8'):
        self.string_value = string_value
        self.encoding = encoding

    def make_instance(self, memory_resource: MemoryResource, name: str):
        '''
        Create a concrete instance from the shadow
        '''
        return shelved_string(memory_resource, name, string_value=self.string_value, encoding=self.encoding)

    def existing_instance(memory_resource: MemoryResource, name: str):
        '''
        Determine if an persistent named memory object corresponds to this type
        '''                        
        buffer = memory_resource.get_named_memory(name)
        if buffer is None:
            raise RuntimeError('bad object name')

        # cast header structure on buffer
        hdr = construct_header_from_buffer(buffer)

        if (hdr.type == DataType_String):

            data_subtype = hdr.subtype
            if data_subtype == DataSubType_Ascii:
                return (True, shelved_string(memory_resource, name, buffer[HeaderSize:], 'ascii'))
            elif data_subtype == DataSubType_Utf8:
                return (True, shelved_string(memory_resource, name, buffer[HeaderSize:], 'utf-8'))
            elif data_subtype == DataSubType_Utf16:
                return (True, shelved_string(memory_resource, name, buffer[HeaderSize:], 'utf-16'))
            elif data_subtype == DataSubType_Latin1:
                return (True, shelved_string(memory_resource, name, buffer[HeaderSize:], 'latin-1'))

        # not a string
        return (False, None)

    def build_from_copy(memory_resource: MemoryResource, name: str, value):
        return shelved_string(memory_resource, name, string_value=value, encoding='utf-8')


class shelved_string(ShelvedCommon):
    '''Shelved string with multiple encoding support'''
    def __init__(self, memory_resource, name, string_value, encoding):

        if not isinstance(name, str):
            raise RuntimeError("invalid name type")

        memref = memory_resource.open_named_memory(name)

        if memref == None:

            # create new value
            total_len = len(string_value) + HeaderSize
            memref = memory_resource.create_named_memory(name, total_len, 8, False)

            memref.tx_begin(value_named_memory=None, check=False)
            hdr = construct_header_on_buffer(memref.buffer, DataType_String, txbits=TXBIT_DIRTY)
            
            if encoding == 'ascii':
                hdr.subtype = DataSubType_Ascii
            elif encoding == 'utf-8':
                hdr.subtype = DataSubType_Utf8
            elif encoding == 'utf-16':
                hdr.subtype = DataSubType_Utf16
            elif encoding == 'latin-1':
                hdr.subtype = DataSubType_Latin1
            else:
                raise RuntimeError('shelved string does not recognize encoding {}'.format(encoding))
                    
            # copy data into memory resource            
            memref.buffer[HeaderSize:] = bytes(string_value, encoding)
            memref.tx_commit()

        self.view = memoryview(memref.buffer[HeaderSize:])

        # hold a reference to the memory resource
        self._memory_resource = memory_resource
        self._metadata_named_memory = memref
        self._value_named_memory = None
        self.encoding = encoding
        self.name = name

    def __repr__(self):
        # TODO - some how this is keeping a reference? gc.collect() clears it.
        #
        # creates a new string
        return str(self.view,self.encoding)

    def __len__(self):
        return len(self.view)

    def __getitem__(self, key):
        '''
        Magic method for slice handling
        '''
        s = str(self.view,'utf-8')
        if isinstance(key, int):
            return s[key]
        if isinstance(key, slice):
            return s.__getitem__(key)
        else:
            raise TypeError

    def persist(self):
        '''
        Flush cache and persistent all value memory
        '''
        self._value_named_memory.persist()        
    
    def __getattr__(self, name):
        if name not in ("encoding"):
            raise AttributeError("'{}' object has no attribute '{}'".format(type(self),name))
        else:
            return self.__dict__[name]

    def __add__(self, value): # in-place append e.g, through +=

        # build a new value with different name, then swap & delete
        new_str = str(self.view,self.encoding).__add__(value)
        
        memory = self._memory_resource
        total_len = HeaderSize + len(new_str)
        memref = memory.create_named_memory(self.name + "-tmp", total_len, 8, False)

        hdr = init_header_from_buffer(memref.buffer)
        hdr.type = DataType_String
        
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
        memref.buffer[HeaderSize:] = bytes(new_str, self.encoding)
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
        self.view = memoryview(memref.buffer[HeaderSize:])
        return self

    def __eq__(self, value): # == operator
        return (str(self.view,self.encoding).__eq__(value))

    def __ge__(self, value): # == operator
        return (str(self.view,self.encoding).__ge__(value))

    def __gt__(self, value): # == operator
        return (str(self.view,self.encoding).__gt__(value))

    def __le__(self, value): # == operator
        return (str(self.view,self.encoding).__le__(value))

    def __lt__(self, value): # == operator
        return (str(self.view,self.encoding).__lt__(value))

    def __ne__(self, value): # == operator
        return (str(self.view,self.encoding).__ne__(value))

    def __contains__(self, value): # in / not in operator
        return (value in str(self.view,self.encoding))

    def capitalize(self):
        return str(self.view,self.encoding).capitalize()

    def center(self, width, fillchar=' '):
        return str(self.view,self.encoding).center(width, fillchar)
    
    def casefold(self):
        return str(self.view,self.encoding).casefold()

    def count(self, start=0, end=0):
        if end > 0:
            return str(self.view,self.encoding).count(start, end)
        else:
            return str(self.view,self.encoding).count(start)
        
    def encode(self, encoding='utf-8', errors='strict'):
        return str(self.view,self.encoding).encode(encoding, errors)

    # TODO MOSHIK TO FINISH ..

# string object methods
#
#  ['__add__', '__class__', '__contains__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getnewargs__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__mod__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__rmod__', '__rmul__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'capitalize', 'casefold', 'center', 'count', 'encode', 'endswith', 'expandtabs', 'find', 'format', 'format_map', 'index', 'isalnum', 'isalpha', 'isdecimal', 'isdigit', 'isidentifier', 'islower', 'isnumeric', 'isprintable', 'isspace', 'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'maketrans', 'partition', 'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit', 'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase', 'title', 'translate', 'upper', 'zfill']
