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
import os

from .metadata import *
from .check import methodcheck, paramcheck

class TxHandler:
    '''
    Transaction handler for persistent memory
    '''
    def __init__(self, name:str, memory_view:memoryview, memory_resource: pymmcore.MemoryResource):
        self.name = name
        self.memory_view = memory_view
        self.memory_resource = memory_resource
        self.tx_log = []  # undo log

    def tx_add(self):
        #pymmcore.valgrind_trigger(1)
        self.__tx_add_undocopy()

    def tx_begin(self):
        #pymmcore.valgrind_trigger(1)
        self.__tx_add_undocopy()

    def tx_commit(self):
        #pymmcore.valgrind_trigger(2)
        self.__tx_commit_undocopy()

    def __tx_add_undocopy(self):
        '''
        Start consistent transaction (very basic copy-off undo-log)
        '''
        name = self.name + '-' + str(len(self.tx_log)) + '-tx' 
        (tx_handle, mem) = self.memory_resource._MemoryResource_create_named_memory(name, len(self.memory_view))
        if tx_handle is None:
            raise RuntimeError('tx_begin failed')

        self.tx_log.append((tx_handle, name))
        # copy data, then persist
        mem[:]= self.memory_view
        self.memory_resource._MemoryResource_persist_memory_view(self.memory_view)
        print('tx_begin: copy of {}:{} to ({}, {})'
              .format(hex(pymmcore.memoryview_addr(self.memory_view)), len(self.memory_view), name, hex(pymmcore.memoryview_addr(mem))))

    def __tx_commit_undocopy(self):
        '''
        Commit consistent transaction
        '''
        for tx_entry in self.tx_log:
            self.memory_resource.release_named_memory_by_handle(tx_entry[0])
            self.memory_resource.erase_named_memory(tx_entry[1])
        self.tx_log = []
        print('tx_commit OK!')
        


    
class MemoryReference():
    '''
    MemoryReference represents a contiguous region of memory associated with a variable
    value or metadata region
    '''
    def __init__(self, internal_handle, memory_resource, memview, name, tx_handler: TxHandler):
        self.handle = internal_handle
        self.mr = memory_resource
        self.buffer = memview
        self.varname = name
        self.tx_handler = tx_handler

        if os.getenv('PYMM_DEBUG') != None:
            self._debug_level = int(os.getenv('PYMM_DEBUG'))
        else:
            self._debug_level = 0

        if os.getenv('PYMM_USE_SW_TX') != None:
            self._use_sw_tx = (int(os.getenv('PYMM_USE_SW_TX')) > 0)
        else:
            self._use_sw_tx = False

    def __del__(self):
        if self._debug_level > 0:
            print("releasing named memory {} @ {}".format(self.varname, hex(pymmcore.memoryview_addr(self.buffer))))
        self.persist()
        self.mr.release_named_memory_by_handle(self.handle)

    def addr(self):
        '''
        For debugging. Get address of memory view
        '''
        return (hex(pymmcore.memoryview_addr(self.buffer)), len(self.buffer))

    def tx_begin(self, value_named_memory=None, check=True):
        '''
        Add region of memory to transaction.  This is called on metadata named memory,
        and passed the value named memory if it exists (i.e. if it is not contiguous)
        '''
        if value_named_memory != None:
            assert isinstance(value_named_memory, MemoryReference)

        # sanity check - but not fool proof!
        if check:
            hdr = construct_header_from_buffer(self.buffer)
            if metadata_check_tx_bit(hdr, TXBIT_DIRTY):
                return # nested, don't do anything
        else:
            hdr = construct_header_on_buffer(self.buffer)
            
        metadata_set_dirty_tx_bit(hdr)
            
        if self._use_sw_tx: # optional undo logging hook
            self.tx_handler.tx_begin(value_named_memory)

    def tx_commit(self, value_named_memory=None):
        '''
        Commit transaction for variable
        '''        
        hdr = construct_header_from_buffer(self.buffer)

        # if we are in a multi-variable transaction, then delay commit
        if metadata_check_tx_bit(hdr, TXBIT_MULTIVAR):
            return

        self.persist()
        if value_named_memory != None:
            assert isinstance(value_named_memory, MemoryReference)
            value_named_memory.persist()

        metadata_clear_dirty_tx_bit(hdr)
        
        if self._use_sw_tx: # optional undo logging hook
            self.tx_handler.tx_commit(value_named_memory)



        
    def tx_multivar_commit(self, value_named_memory=None):
        '''
        Commit variable as part of multi-variable transaction
        '''
        hdr = construct_header_from_buffer(self.buffer)
        hdr.version += 1
        
        # we might not have dirtied the memory
        if not metadata_check_dirty_tx_bit(hdr):
            metadata_clear_tx_bit(hdr, TXBIT_MULTIVAR | TXBIT_DIRTY)
            return

        # dirty bit is set
        if self._use_sw_tx: # optional undo logging hook
            self.tx_handler.tx_commit(value_named_memory)      

        if value_named_memory != None:
            assert isinstance(value_named_memory, MemoryReference)
            value_named_memory.persist()

        metadata_clear_tx_bit(hdr, TXBIT_MULTIVAR | TXBIT_DIRTY)
        
            
    def persist(self):
        '''
        Flush any cached memory (normally for persistence)
        '''
        self.mr._MemoryResource_persist_memory_view(self.buffer)

        
class MemoryResource(pymmcore.MemoryResource):
    '''
    MemoryResource represents a heap allocator and physical memory
    resources.  It is backed by an MCAS store component and corresponds
    to a pool.
    '''
    def __init__(self, name, size_mb, pmem_path, load_addr, backend=None, mm_plugin=None, force_new=False):
        self._named_memory = {}

        if os.getenv('PYMM_DEBUG') != None:
            debug_level = int(os.getenv('PYMM_DEBUG'))
        else:
            debug_level = 0

        super().__init__(pool_name=name, size_mb=size_mb, pmem_path=pmem_path,
                         load_addr=load_addr, backend=backend, mm_plugin=mm_plugin, force_new=force_new, debug_level=debug_level)
        
        # todo check for outstanding transactions
        all_items = super()._MemoryResource_get_named_memory_list()
        recoveries = [val for val in all_items if val.endswith('-tx')]
        
        if len(recoveries) > 0:
            raise RuntimeError('detected outstanding undo log condition: recovery not yet implemented')

    @methodcheck(types=[])        
    def list_items(self):
        all_items = super()._MemoryResource_get_named_memory_list()
        # exclude values
        return [val for val in all_items if not val.endswith('-value')]
    
    @methodcheck(types=[str,int,int,bool])
    def create_named_memory(self, name, size, alignment=256, zero=False):
        '''
        Create a contiguous piece of memory and name it
        '''
        (handle, mview) = super()._MemoryResource_create_named_memory(name, size, alignment, zero)
        if handle == None:
            return None

        return MemoryReference(handle, self, mview, name, TxHandler(name, mview, self))

    @methodcheck(types=[str])
    def open_named_memory(self, name):
        '''
        Open existing named memory
        '''
        (handle, mview) = super()._MemoryResource_open_named_memory(name)
        if handle == None:
            return None
        return MemoryReference(handle, self, mview, name, TxHandler(name, mview, self))

    @methodcheck(types=[MemoryReference])
    def release_named_memory(self, ref : MemoryReference):
        '''
        Release a contiguous piece of memory (i.e. unlock)
        '''
        super()._MemoryResource_release_named_memory(ref.handle)

    @methodcheck(types=[int])
    def release_named_memory_by_handle(self, handle):
        '''
        Release a contiguous piece of memory (i.e. unlock)
        '''
        super()._MemoryResource_release_named_memory(handle)

    @methodcheck(types=[str])
    def erase_named_memory(self, name):
        '''
        Erase a named-memory object from the memory resource
        '''
        super()._MemoryResource_erase_named_memory(name)

    @methodcheck(types=[str,bytearray])
    def put_named_memory(self, name, value):
        '''
        Copy-based crash-consistent put of named memory value
        '''
        if not isinstance(value, bytearray):
            raise RuntimeError('put_named_memory requires bytearray data')
            
        super()._MemoryResource_put_named_memory(name, value)

    @methodcheck(types=[str])
    def get_named_memory(self, name):
        '''
        Copy-based get of named memory value
        '''
        return super()._MemoryResource_get_named_memory(name)

    
    def get_percent_used(self):
        '''
        Get percentage of memory used in memory resource
        '''
        return super()._MemoryResource_get_percent_used()


    def atomic_swap_names(self, a: str, b: str):
        '''
        Swap the names of two named memories; must be released
        '''
        return super()._MemoryResource_atomic_swap_names(a,b)

