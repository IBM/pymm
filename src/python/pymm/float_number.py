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

from .metadata import *
from .memoryresource import MemoryResource
from .shelf import Shadow, ShelvedCommon

class float_number(Shadow):
    '''
    Floating point number that is stored in the memory resource. Uses value cache
    '''
    def __init__(self, number_value):
        self.number_value = number_value

    def make_instance(self, memory_resource: MemoryResource, name: str):
        '''
        Create a concrete instance from the shadow
        '''
        return shelved_float_number(memory_resource, name, number_value=self.number_value)

    def existing_instance(memory_resource: MemoryResource, name: str):
        '''
        Determine if an persistent named memory object corresponds to this type
        '''                        
        buffer = memory_resource.get_named_memory(name)
        if buffer is None:
            return (False, None)

        hdr = construct_header_from_buffer(buffer)

        if (hdr.type == DataType_NumberFloat):
            f = float.fromhex((buffer[HeaderSize:]).decode())
            return (True, shelved_float_number(memory_resource, name, f))

        # not a string
        return (False, None)

    def build_from_copy(memory_resource: MemoryResource, name: str, value):
        return shelved_float_number(memory_resource, name, number_value=value)


class shelved_float_number(ShelvedCommon):
    '''
    Shelved floating point number
    '''
    def __init__(self, memory_resource, name, number_value):

        memref = memory_resource.open_named_memory(name)

        if memref == None:
            
            # create new value
            value_bytes = str.encode(number_value.hex())
            total_len = HeaderSize + len(value_bytes)
            memref = memory_resource.create_named_memory(name, total_len, 8, False)

            memref.tx_begin(value_named_memory=None, check=False)
            hdr = construct_header_on_buffer(memref.buffer, DataType_NumberFloat, txbits=TXBIT_DIRTY)
                        
            # copy data into memory resource
            memref.buffer[HeaderSize:] = value_bytes
            memref.tx_commit()
        else:
            # validates
            construct_header_from_buffer(memref.buffer)

        self._cached_value = float(number_value)
        self.name = name
        # hold a reference to the memory resource
        self._memory_resource = memory_resource
        self._metadata_named_memory = memref
        self._value_named_memory = None

    def _atomic_update_value(self, value):
        if not isinstance(value, float):
            raise TypeError('bad type for atomic_update_value')

        # because we are doing swapping create new new
        value_bytes = str.encode(value.hex())
        total_len = HeaderSize + len(value_bytes)

        memory = self._memory_resource
        memref = memory.create_named_memory(self.name + '-tmp', total_len, 8, False)

        memref.tx_begin(value_named_memory=None, check=False) # not sure if we need this
        hdr = construct_header_on_buffer(memref.buffer, DataType_NumberFloat, txbits=TXBIT_DIRTY)
        
        # copy into memory resource
        memref.buffer[HeaderSize:] = value_bytes
        memref.tx_commit()

        del memref # this will force release
        del self._metadata_named_memory # this will force release
        gc.collect()

        # swap names
        memory.atomic_swap_names(self.name, self.name + "-tmp")

        # erase old data
        memory.erase_named_memory(self.name + "-tmp")
        
        memref = memory.open_named_memory(self.name)
        self._metadata_named_memory = memref
        self._value_named_memory = None
        self._cached_value = value
        # materialization alternative - self._view = memoryview(memref.buffer[Constants.Constants().HdrSize + 4:])
        return self

        
    def _get_value(self):
        '''
        Materialize the value either from persistent memory or cached value
        '''
        return self._cached_value
        # materialization alternative - return float.fromhex((bytearray(self._view)).decode())


    def __repr__(self):
        return str(self._get_value())

    def __float__(self):
        return float(self._get_value())

    def __int__(self):
        return int(self._get_value())

    def __bool__(self):
        return bool(self._get_value())

    # in-place arithmetic
    def __iadd__(self, value): # +=
        return self._atomic_update_value(float(self._get_value()).__add__(value))

    def __imul__(self, value): # *=
        return self._atomic_update_value(float(self._get_value()).__mul__(value))

    def __isub__(self, value): # -=
        return self._atomic_update_value(float(self._get_value()).__sub__(value))

    def __itruediv__(self, value): # /=
        return self._atomic_update_value(float(self._get_value()).__truediv__(value))

    def __imod__(self, value): # %=
        return self._atomic_update_value(float(self._get_value()).__mod__(value))

    def __ipow__(self, value): # **=
        return self._atomic_update_value(float(self._get_value()).__pow__(value))

    def __ilshift__(self, value): # <<=
        return self._atomic_update_value(float(self._get_value()).__lshift__(value))

    def __irshift__(self, value): # >>=
        return self._atomic_update_value(float(self._get_value()).__rshift__(value))

    def __iand__(self, value): # &=
        return self._atomic_update_value(float(self._get_value()).__and__(value))

    def __ixor__(self, value): # ^=
        return self._atomic_update_value(float(self._get_value()).__xor__(value))

    def __ior__(self, value): # |=
        return self._atomic_update_value(float(self._get_value()).__or__(value))
        
        

    # arithmetic operations
    # not using magic method seems to help with implicit casting
    def __add__(self, value):
        return self._get_value() + value

    def __radd__(self, value):
        return value + self._get_value()

    def __sub__(self, value):
        return self._get_value() - value

    def __rsub__(self, value):
        return value - self._get_value()

    def __mul__(self, value):
        return self._get_value() * value

    def __rmul__(self, value):
        return value * self._get_value()

    def __truediv__(self, value):
        return self._get_value() / value

    def __rtruediv__(self, value):
        return self._get_value() / value

    def __floordiv__(self, value):
        return self._get_value() // value

    def __rfloordiv__(self, value):
        return value // self._get_value()
    
    def __divmod__(self, x):
        return divmod(self._get_value(),x)

    def __eq__(self, x):
        return self._get_value() == x

    def __le__(self, x):
        return self._get_value() <= x

    def __lt__(self, x):
        return self._get_value() < x

    def __ge__(self, x):
        return self._get_value() >= x

    def __gt__(self, x):
        return self._get_value() > x

    def __ne__(self, x):
        return self._get_value() != x

    def __round__(self, places):
        return float(self._get_value()).__round__(places)


    # TODO: MOSHIK TO FINISH
    
    def hex(self):
        return float(self._get_value()).hex()

        
# ['__abs__', '__add__', '__and__', '__bool__', '__ceil__', '__class__', '__delattr__', '__dir__', '__divmod__', '__doc__', '__eq__', '__float__', '__floor__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getnewargs__', '__gt__', '__hash__', '__index__', '__init__', '__init_subclass__', '__int__', '__invert__', '__le__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__', '__neg__', '__new__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rlshift__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__trunc__', '__xor__', 'bit_length', 'conjugate', 'denominator', 'from_bytes', 'imag', 'numerator', 'real', 'to_bytes']
