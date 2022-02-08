/*
   IBM Corporation Copyright (C) 2020

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.

   As a special exception, if you link the code in this file with
   files compiled with a GNU compiler to produce an executable,
   that does not cause the resulting executable to be covered by
   the GNU Lesser General Public License.  This exception does not
   however invalidate any other reasons why the executable file
   might be covered by the GNU Lesser General Public License.
   This exception applies to code released by its copyright holders
   in files containing the exception.
*/
#ifndef __COMMON_BYTE_BUFFER_H__
#define __COMMON_BYTE_BUFFER_H__

#include <common/exceptions.h>

namespace common
{

/** 
 * Basic FIFO buffer stream
 * 
 */
class Byte_buffer
{
public:

  /** 
   * Take data from the FIFO
   * 
   * @param buffer Destination 
   * @param buffer_size Size in bytes requested
   */
  void pull(void * buffer, const size_t buffer_size) {
    if(buffer_size > _remaining_len)
      throw API_exception("%s: underflow pop", __func__ );

    memcpy(buffer, _head, buffer_size);
    _remaining_len -= buffer_size;

    if(_remaining_len > 0) {
      _head += buffer_size;
    }
    else {
      ::free(_remaining);
      _remaining = _head = nullptr;
      _remaining_len = 0;
    }
  }

  /** 
   * Push into the FIFO
   * 
   * @param buffer Source
   * @param buffer_size Size in bytes
   */
  void push(const void * buffer, const size_t buffer_size) {
    if(_remaining)
      throw API_exception("%s: remaining not empty", __func__ );
    
    _remaining_len = buffer_size;
    _head = _remaining = reinterpret_cast<byte*>(::malloc(buffer_size));    
    memcpy(_remaining, buffer, buffer_size);
  }

  /** 
   * Provide remaining number of bytes
   * 
   */
  inline size_t remaining() const { return _remaining_len; }

private:
  byte * _head = nullptr;
  byte * _remaining = nullptr;
  size_t _remaining_len = 0;
  
};

}

#endif // __COMMON_BYTE_BUFFER_H__
