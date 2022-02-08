/*
  Copyright [2017-2019] [IBM Corporation]
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
#ifndef __CCPM_FIXED_ARRAY_H__
#define __CCPM_FIXED_ARRAY_H__

#include <libpmem.h> /* pmem_memset */
#include <stdexcept>

namespace ccpm
{

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

template <typename T, size_t S>
class Fixed_array
{
  
public:
  typedef T&                               reference;
  typedef const T&                         const_reference;
  typedef size_t                           size_type;
  typedef ptrdiff_t                        difference_type;
  typedef T                                value_type;
  typedef T*                               pointer;
  typedef const T*                         const_pointer;

private:
  struct Proxy_object {
    Fixed_array& array;
    pointer      element;
    
    operator pointer() const { return element; }

    T& operator=(T new_val) {
      if(S > 8) {
        array._undo_log = *element;
        pmem_flush(&array._undo_log, sizeof(T)); /* must be before writing location */
        array._undo_location = element;
        pmem_flush(&array._undo_location, sizeof(T*));
      }

      *element = new_val; 
      pmem_flush(element, sizeof(T));
      
      if(S > 8) { /* clear undo */
        array._undo_location = nullptr;
        /*  no need to flush since this is power-atomic and 
            late flush will just be conservative on undo test
        */
        //pmem_flush(&array._undo_location, sizeof(T*));
      }

      return *element;      
    }
    
    /* we can't override the dot operator so this is 
       the best we can do */
    reference ref() const { return *element; }
  };

public:

  /**
   * @brief      Class for crash-consistent fixed array; uses undo log for elements > 8 bytes
   */
  Fixed_array() {    
    if(!check_aligned(this, 8))
      throw std::invalid_argument("memory is not 64-bit aligned");
    
    if(_type_id == Type_id::None)
      initialize();
    else
      check_undo_log();
  }

  Proxy_object operator[](size_type n) {
    if(n >= S) throw std::range_error("invalid index");
    return Proxy_object{*this,&_data[n]};
  }

  bool check_ok() const { return _type_id == Type_id::Fixed_array; }

  /**
   * @brief      Get size of array in elements
   *
   * @return     Number of elements in the area
   */
  static size_t size() { return S; }

  /**
   * @brief      Initializes the data structure if type id is not set or forces
   *
   * @param[in]  force  Set true to force initialization
   */
  void initialize(bool force = false) {
    if(_type_id != Type_id::Fixed_array || force) {
      pmem_memset(&_data, 0, (sizeof(T) * S), 0); /* zero the slots */
      _type_id = Type_id::Fixed_array;
      pmem_flush(&_type_id, sizeof(_type_id));
    }
  }

  Type_id type_id() const { return Type_id::Fixed_array; }

private:
  void check_undo_log() {
    if(_undo_location != nullptr) {
      pmem_memcpy(_undo_location, &_undo_log, sizeof(T), 0);
      _undo_location = nullptr;
      pmem_flush(&_undo_location, sizeof(_undo_location));
      /* no need to clear undo log content */
    }
  }
  
private:
  ccpm::Type_id _type_id;
  T*            _undo_location; 
  T             _undo_log;
  T             _data[S];
};


#pragma GCC diagnostic pop
}


#endif // __CCPM_FIXED_ARRAY_H__
