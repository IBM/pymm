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

#ifndef _MCAS_POOL_INSTANCE_H_
#define _MCAS_POOL_INSTANCE_H_

#include "numa_node_mask.h"
#include "mm_plugin_itf.h"

#include <api/kvstore_itf.h> /* string_view_key, string_view_value */
#include <common/less_getter.h>
#include <common/rwlock.h>
#include <common/time.h> /* tsc_time_t */
#include <nupm/region_descriptor.h>
#include <common/logging.h>
#include <city.h> /* CityHash64 */
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>

struct bitmask;

struct value_type {
  value_type() : _ptr(nullptr), _length(0), _value_lock(), _tsc() {
  }

  value_type(void* ptr, size_t length, common::RWLock * value_lock) :
    _ptr(ptr), _length(length), _value_lock(value_lock), _tsc() {
  }

  value_type(const value_type &) = delete;
  value_type &operator=(const value_type &) = delete;
  void * _ptr;
  size_t _length;
  common::RWLock * _value_lock; /*< read write lock */
  common::tsc_time_t _tsc;
};

class Key_hash {
public:
  size_t operator()(component::IKVStore::string_view_key k) const {
    return CityHash64(common::pointer_cast<char>(k.data()), k.size());
  }
};

struct region_memory;

/**
 * Pool instance class
 *
 */
class Pool_instance {

private:
  using aac_t = MM_plugin_cxx_allocator<char>;
  using string_t = std::basic_string<common::byte, std::char_traits<common::byte>, aac_t>;
  using aam_t = MM_plugin_cxx_allocator<std::pair<string_t, value_type>>;
  using map_t = std::unordered_map<string_t, value_type, Key_hash, std::equal_to<string_t>, aam_t>;

  unsigned debug_level() const { return _debug_level; }

  std::unique_ptr<region_memory> allocate_region_memory(size_t size);

  static const Pool_instance *checked_pool(const Pool_instance * pool)
  {
    if ( pool == nullptr )
      throw Logic_exception("checked_pool bad param");

    return pool;
  }

  class Iterator {
  public:
    explicit Iterator(const Pool_instance * pool)
      : _pool(checked_pool(pool)),
        _mark(_pool->writes()),
        _iter(_pool->_map->begin()),
        _end(_pool->_map->end())
    {}

    bool is_end() const { return _iter == _end; }
    bool check_mark(uint32_t writes) const { return _mark == writes; }

    const Pool_instance * _pool;
    uint32_t              _mark;
    map_t::const_iterator _iter;
    map_t::const_iterator _end;
  };

  using IKVStore = component::IKVStore;
  using string_view = common::string_view;
  using string_view_key = IKVStore::string_view_key;
  using string_view_value = IKVStore::string_view_value;
public:
  Pool_instance(const unsigned debug_level,
                const common::string_view mm_plugin_path,
                const common::string_view name_,
                size_t nsize,
                const bitmask *numa_node_mask_,
                unsigned flags_);
  Pool_instance(const Pool_instance &) = delete;
  Pool_instance &operator=(const Pool_instance &) = delete;

  ~Pool_instance();
  const std::string& name() const { return _name; }

private:

  unsigned                   _debug_level;
  std::mutex                 _ref_mutex;
  size_t                     _nsize; /*< order important */
  numa_node_mask             _numa_node_mask;
  std::string                _name; /*< pool name */
  int                        _fdout;
  std::vector<std::unique_ptr<region_memory>> _regions; /*< regions supporting pool */
  std::mutex                 _mm_plugin_mutex;
  MM_plugin_wrapper          _mm_plugin;
  /* use a pointer so we can make sure it gets stored before memory is freed */
  common::RWLock             _map_lock; /*< read write lock */
  std::unique_ptr<map_t>     _map; /*< hash table based map */
  unsigned int               _flags;
  /* Note: Using Iterator * as a comparable is a slight cheat, because pointers
   * from separate allocations are, strictly speaking, not comparable.
   */
  std::set<std::unique_ptr<Iterator>, common::less_getter<std::unique_ptr<Iterator>>> _iterators;
  /*
    We use this counter to see if new writes have come in
    during an iteration.  This is essentially an optmistic
    locking strategy.
  */
  uint32_t _writes __attribute__((aligned(4)));

  inline void write_touch() { _writes++; }
  inline uint32_t writes() const { return _writes; }

  /* allocator adapters over reconstituting allocator */
  aac_t aac{_mm_plugin}; /* for keys */
  using aal_t = MM_plugin_cxx_allocator<common::RWLock>;
  aal_t aal{_mm_plugin}; /* for locks */

  /* unguarded inner lock function (caller must hold _mm_plugin_mutex) */
  status_t lock_unguarded(const std::lock_guard<std::mutex> &, string_view_key key,
                IKVStore::lock_type_t type,
                void *&out_value,
                size_t &inout_value_len,
                size_t alignment,
                IKVStore::key_t& out_key,
                const char ** out_key_ptr);

public:
  status_t put(string_view_key key, const void *value,
               const size_t value_len, unsigned int flags);

  status_t get(string_view_key key, void *&out_value, size_t &out_value_len);

  status_t get_direct(string_view_key key, void *out_value,
                      size_t &out_value_len);

  status_t get_attribute(const IKVStore::Attribute attr,
                         std::vector<uint64_t> &out_attr,
                         const string_view_key key);

  status_t swap_keys(const string_view_key key0,
                     const string_view_key key1);

  status_t resize_value(string_view_key key,
                        const size_t new_size,
                        const size_t alignment);

  status_t lock(string_view_key key,
                IKVStore::lock_type_t type,
                void *&out_value,
                size_t &inout_value_len,
                size_t alignment,
                IKVStore::key_t& out_key,
                const char ** out_key_ptr);

  status_t unlock(IKVStore::key_t key_handle);

  status_t erase(string_view_key key);

  size_t count();

  status_t map(std::function<int(string_view_key key,
                                 string_view_value value)> function);

  status_t map(std::function<int(string_view_key key,
                                 string_view_value value,
                                 const common::tsc_time_t timestamp)> function,
                                 const common::epoch_time_t t_begin,
                                 const common::epoch_time_t t_end);

  status_t map_keys(std::function<int(string_view_key key)> function);

  status_t get_pool_regions(nupm::region_descriptor::address_map_t &out_regions);

  status_t grow_pool(const size_t increment_size, size_t &reconfigured_size);

  status_t free_pool_memory(const void *addr, const size_t size = 0);

  status_t allocate_pool_memory(const size_t size,
                                const size_t alignment,
                                void *&out_addr);

  IKVStore::pool_iterator_t open_pool_iterator();

  status_t deref_pool_iterator(IKVStore::pool_iterator_t iter,
                               const common::epoch_time_t t_begin,
                               const common::epoch_time_t t_end,
                               IKVStore::pool_reference_t& ref,
                               bool& time_match,
                               bool increment = true);

  status_t close_pool_iterator(IKVStore::pool_iterator_t iter);
};

#endif
