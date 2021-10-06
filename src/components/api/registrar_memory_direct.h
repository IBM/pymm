/*
  Copyright [2017-2021] [IBM Corporation]
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#ifndef _REGISTRAR_MEMORY_DIRECT__
#define _REGISTRAR_MEMORY_DIRECT__

#include <common/byte_span.h>
#include <common/types.h>

#define DECLARE_OPAQUE_TYPE(NAME)               \
  struct Opaque_##NAME {                        \
    virtual ~Opaque_##NAME() {}                 \
  }

namespace component
{
/**
 * registrar of local memory regions (for use in DMA)D
 */

class Registrar_memory_direct
{
protected:
  ~Registrar_memory_direct() {}
public:
  DECLARE_OPAQUE_TYPE(memory_region); /* Buffer_manager::buffer_t need this */
  using memory_handle_t = Opaque_memory_region*;
  
  /**
   * Register memory for zero copy DMA
   *
   * @param bytes Appropriately aligned memory buffer to register
   *
   * @return Memory handle or NULL on not supported.
   */
  virtual memory_handle_t register_direct_memory(common::const_byte_span bytes) = 0;

  /**
   * Register memory for zero copy DMA
   *
   * @param vaddr Appropriately aligned memory buffer
   * @param len Length of memory buffer in bytes
   *
   * @return Memory handle or NULL on not supported.
   */
  memory_handle_t register_direct_memory(void* vaddr, const size_t len)
  {
    return register_direct_memory(common::make_const_byte_span(vaddr, len));
  }

  /**
   * Direct memory regions should be unregistered before the memory is released
   * on the client side.
   *
   * @param handle (as returned by register_direct_memory) of region to deregister.
   *
   * @return S_OK on success
   */
  virtual status_t unregister_direct_memory(const memory_handle_t handle) = 0;
};

}  // namespace component

#endif
