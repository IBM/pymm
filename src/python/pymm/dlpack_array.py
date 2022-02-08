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

#    Notes: not all methods hooked.

import pymmcore
import numpy as np
import copy
import os

from .metadata import *
from numpy import uint8, ndarray, dtype, float
from .memoryresource import MemoryResource
from .shelf import Shadow, ShelvedCommon

dtypedescr = np.dtype
wbinvd_threshold = 1073741824

# shadow type for ndarray
#
class dlpack_array(Shadow):
    '''
    ndarray that is stored in a memory resource
    '''
    def __init__(self, shape, dtype=np.float64, strides=None, zero=False):
        self.__p_shape = shape
        self.__p_dtype = dtype
        self.__p_strides = strides
        self.__p_zero = zero

    def make_instance(self, memory_resource: MemoryResource, name: str):
        '''
        Create a concrete instance from the shadow
        '''
        return shelved_dlpack_array(memory_resource,
                                    name,
                                    shape = self.__p_shape,
                                    dtype = self.__p_dtype,
                                    strides = self.__p_strides,
                                    zero = self.__p_zero)

    def existing_instance(memory_resource: MemoryResource, name: str):
        '''
        Determine if an persistent named memory object corresponds to this type
        '''
        print(memory_resource)
        assert isinstance(memory_resource, MemoryResource)
        
        metadata = memory_resource.get_named_memory(name)
        if metadata is None:
            raise RuntimeError('bad object name')

        # cast header structure on buffer
        hdr = construct_header_from_buffer(metadata)

        if hdr.type == DataType_DLTensor:
            return (True, shelved_dlpack_array(memory_resource, name, shape = None))
        else:
            return (False, None)
            

    def __str__(self):
        print('shadow ndarray')


    def build_from_copy(memory_resource: MemoryResource, name: str, array):
        raise RuntimeError('not implemented')
        # new_array = shelved_dlpack_array(memory_resource,
        #                                  name,
        #                                  shape = array.shape,
        #                                  dtype = array.dtype,
        #                                  strides = array.strides)

        # # now copy the data
        # #new_array[:] = array
        # np.copyto(new_array, array)
        # return new_array


# concrete subclass for ndarray
#
class shelved_dlpack_array(ShelvedCommon):
    '''
    DLPack array that is stored in a memory resource
    '''
    def __init__(self, memory_resource, name, shape, dtype=np.float64, strides=None, type=0, zero=False):

        #
        # determine size of memory needed
        #
        descr = dtypedescr(dtype)
        bytes_per_item = descr.itemsize

        if shape != None:
            size = np.intp(1)  # avoid default choice of np.int_, which might overflow

            if isinstance(shape, tuple) or isinstance(shape, list):
                for k in shape:
                    size *= k
            else:
                raise RuntimeError('unhandled condition in shelved_ndarray shape handling (shape={})'.format(shape))

        # the meta data is always accessible by the plain key name
        # the value if not concated after meta data is in a separate key with -value suffix
        value_key = name + '-value'
        metadata_key = name

        metadata_memory = memory_resource.open_named_memory(metadata_key)
        value_memory = memory_resource.open_named_memory(value_key)

        if value_memory == None: # does not exist yet
            #
            # create a newly allocated named memory from MemoryResource
            #
            memory_size = int(size * bytes_per_item)
            if memory_size < 8:
                alignment = 1
            else:
                alignment = 8

            # construct metadata in volatile memory
            #
            (hdr_bytes, data_size) = pymmcore.dlpack_construct_meta(dtypedescr=descr,
                                                                    shape=shape,
                                                                    strides=strides)
            memory_resource.put_named_memory(metadata_key, hdr_bytes)
            metadata_memory = memory_resource.open_named_memory(metadata_key)
            
            value_memory = memory_resource.create_named_memory(value_key,
                                                               data_size,
                                                               256, # as per DLpack spec
                                                               zero) # zero memory        
            assert value_memory != None
            # fix up the embedded pointers (using memoryviews)
            pymmcore.dlpack_fix_pointers(metadata_memory.buffer, value_memory.buffer)

        else:
            # entity already exists, load metadata
            assert metadata_memory != None
            assert value_memory != None

            # if the load address is different the pointers need fixing; we would not need
            # this if the same load address is always used
            pymmcore.dlpack_fix_pointers(metadata_memory.buffer, value_memory.buffer)
            
            # hdr = pymmcore.ndarray_read_header(memoryview(metadata_memory.buffer),type=type)
            # self = np.ndarray.__new__(subtype, dtype=hdr['dtype'], shape=hdr['shape'], buffer=value_memory.buffer,
            #                           strides=hdr['strides'], order=order)

            
        # hold a reference to the memory resource
        self._memory_resource = memory_resource
        self._value_named_memory = value_memory
        self._metadata_named_memory = metadata_memory
        self._metadata_key = metadata_key
        self._value_key = value_key
        self._capsule_ref_count = 0 # number of outstanding references through capsule exposure
        self.name = name

    def as_capsule(self):
        return pymmcore.dlpack_get_capsule(self._metadata_named_memory.buffer)

#        return self

    def __delete__(self, instance):
        raise RuntimeError('cannot delete item: use shelf erase')

    def __array_wrap__(self, out_arr, context=None):
        # Handle scalars so as not to break ndimage.
        # See http://stackoverflow.com/a/794812/1221924
        if out_arr.ndim == 0:
            return out_arr[()]
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __getattr__(self, name):
        if name == 'addr':
            return self._value_named_memory.addr()
        elif name not in super().__dict__:
            raise AttributeError("'{}' object has no attribute '{}'".format(type(self),name))
        else:
            return super().__dict__[name]

    def asndarray(self):
        return None

    def __str__(self):
        return pymmcore.dlpack_as_str(self._metadata_named_memory.buffer)
    
    def __repr__(self):
        return repr(self.asndarray())

    def dim(self):
        return len(super().shape)
    
    def update_metadata(self, array):
        metadata = pymmcore.ndarray_header(array,np.dtype(dtype).str)
        self._memory_resource.put_named_memory(self._metadata_key, metadata)

    # each type will handle its own transaction methodology.  this
    # is because metadata may be dealt with differently
    #
    def _value_only_transaction(self, F, *args):
        if self is None:
            return
        self._value_named_memory.tx_begin()
        result = F(*args)
        self._value_named_memory.tx_commit()
        return result

    # all methods that perform writes are implicitly used to define transaction
    # boundaries (at least most fine-grained)
    #
    # reference: https://numpy.org/doc/stable/reference/routines.array-manipulation.html
    #

    # in-place methods need to be transactional
    def fill(self, value):
        return self._value_only_transaction(super().fill, value)

    def byteswap(self, inplace):
        if inplace == True:
            return self._value_only_transaction(super().byteswap, True)
        else:
            return super().byteswap(False)

    # in-place arithmetic
    def __iadd__(self, value): # +=
        return self._value_only_transaction(super().__iadd__, value)

    def __imul__(self, value): # *=
        return self._value_only_transaction(super().__imul__, value)

    def __isub__(self, value): # -=
        return self._value_only_transaction(super().__isub__, value)

    def __idiv__(self, value): # /=
        return self._value_only_transaction(super().__idiv__, value)

    def __imod__(self, value): # %=
        return self._value_only_transaction(super().__imod__, value)

    def __ipow__(self, value): # **=
        return self._value_only_transaction(super().__ipow__, value)

    def __ilshift__(self, value): # <<=
        return self._value_only_transaction(super().__ilshift__, value)

    def __irshift__(self, value): # >>=
        return self._value_only_transaction(super().__irshift__, value)

    def __iand__(self, value): # &=
        return self._value_only_transaction(super().__iand__, value)

    def __ixor__(self, value): # ^=
        return self._value_only_transaction(super().__ixor__, value)

    def __ior__(self, value): # |=
        return self._value_only_transaction(super().__ior__, value)

    
    # out-of-place we need to convert back to ndarray
    def __add__(self, value):
        return super().__add__(value).asndarray()

    def __radd__(self, value):
        return super().__radd__(value).asndarray()

    def __mul__(self, value):
        return super().__mul__(value).asndarray()

    def __rmul__(self, value):
        return super().__rmul__(value).asndarray()

    def __sub__(self, value):
        return super().__sub__(value).asndarray()

    def __div__(self, value):
        return super().__div__(value).asndarray()

    def __rdivmod__(self, value):
        return super().__rdivmod__(value).asndarray()

    def __divmod__(self, value):
        return super().__divmod__(value).asndarray()

    def __mod__(self, value):
        return super().__mod__(value).asndarray()

    def __rmod__(self, value):
        return super().__rmod__(value).asndarray()

    def __pow__(self, value):
        return super().__pow__(value).asndarray()

    def __lshift__(self, value):
        return super().__lshift__(value).asndarray()

    def __rshift__(self, value):
        return super().__rshift__(value).asndarray()

    def __and__(self, value):
        return super().__and__(value).asndarray()

    def __or__(self, value):
        return super().__or__(value).asndarray()
    
    def __xor__(self, value):
        return super().__xor__(value).asndarray()

    

    # set item, e.g. x[2] = 2
    # if we want transactionality on this we override it - but beware, it will cost!
#    def __setitem__(self, position, x):
#        return self._value_only_transaction(super().__setitem__, position, x)

    def flip(self, m, axis=None):
        return self._value_only_transaction(super().flip, m, axis)


    # operations that return new views on same data.  we want to change
    # the behavior to give a normal volatile version
#    def reshape(self, shape, order='C'):
#        x = np.array(self) # copy constructor
#        return x.reshape(shape)
        

    def __array_finalize__(self, obj):
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        if obj is None: return

        # From view casting and new-from-template
        self.info = getattr(obj, 'info', None)
        self._memory_resource = getattr(obj, '_memory_resource', None)
        self._value_named_memory = getattr(obj, '_value_named_memory', None)
        self._metadata_named_memory = getattr(obj, '_metadata_named_memory', None)
        self._metadata_key = getattr(obj, '_metadata_key', None)
        self.name = getattr(obj, 'name', None)

    def persist(self):
        '''
        Flush cache and persistent all value memory
        '''
        if self._use_wbinvd and ((super().size * super().itemsize) > wbinvd_threshold):
            os.system("cat /proc/wbinvd")
            return
        
        self._value_named_memory.persist()

        
