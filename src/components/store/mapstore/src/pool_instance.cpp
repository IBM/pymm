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

#include "pool_instance.h"

#include "region_memory_mmap.h"
#include "region_memory_numa_pin.h"

#include <common/env.h> /* env_value */
#include <common/rwlock.h> /* RWLock, RWLock_guard */
#include <common/utils.h> /* wmb */
#include <common/string_view.h>
#include <sys/mman.h> /* mmap */
#include <sys/types.h> /* open */
#include <sys/stat.h> /* open stat */
#include <fcntl.h> /* open */
#include <unistd.h> /* ftruncate, syncfs */

#define DEFAULT_ALIGNMENT 8
#define SINGLE_THREADED
#define MIN_POOL (1ULL << DM_REGION_LOG_GRAIN_SIZE)

/*
 * There is one use of new (but not delete): RWLock.
 *  - RWLock used placement new and custom allocator/deallocator
 */
namespace
{
  int open_region_file(const common::string_view pool_name)
  {
    char * backing_store_dir = ::getenv("MAPSTORE_BACKING_STORE_DIR");
    if ( backing_store_dir )
    {
      /* create file */
      struct stat st;
      if (::stat(backing_store_dir,&st) == 0) {
        if (st.st_mode & (S_IFDIR != 0)) {
          using namespace std::string_literals;
          std::string filename = backing_store_dir + "/mapstore_backing_"s + std::string(pool_name) + ".dat";

          ::mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
          FINF("backing file ({}))", filename);
          return ::open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, mode);
        }
      }
    }
    return -1;
  }

  size_t choose_alignment(size_t size)
  {
    if((size >= 4096) && (size % 4096 == 0)) return 4096;
    if((size >= 64) && (size % 64 == 0)) return 64;
    if((size >= 16) && (size % 16 == 0)) return 16;
    if((size >= 8) && (size % 8 == 0)) return 8;
    if((size >= 4) && (size % 4 == 0)) return 4;
    return 1;
  }

  const bool needs_pinned_pages = ! common::env_value("USE_ODP", true);

  struct dummy_guard /* dummy guard */
  {
    dummy_guard(common::RWLock &, int = common::RWLock_guard::READ) {}
  };

#ifdef SINGLE_THREADED
  using RWLock_guard = dummy_guard;
#else /* real guard */
  using RWLock_guard = common::RWLock_guard;
#endif
}

Pool_instance::Pool_instance(const unsigned debug_level,
                common::string_view mm_plugin_path,
                common::string_view name_,
                size_t nsize,
                const bitmask *numa_node_mask_,
                unsigned flags_)
    : _debug_level(debug_level),
      _ref_mutex{},
      _nsize(0),
      _numa_node_mask(numa_node_mask_),
      _name(name_),
      _fdout(open_region_file(_name)),
      _regions{},
      _mm_plugin_mutex{},
      _mm_plugin(std::string(mm_plugin_path)), /* plugin path for heap allocator */
      _map_lock{},
      _map(std::make_unique<map_t>(aam_t(_mm_plugin))),
      _flags{flags_},
      _iterators{},
      _writes{}
{
  grow_pool(nsize < MIN_POOL ? MIN_POOL : nsize, _nsize);
  CFLOGM(1, "new pool instance {}", name_);
}

Pool_instance::~Pool_instance()
{
  CFLOGM(1, "freeing regions for pool ({})", _name);

  if ( 0 <= _fdout )
  {
    /* github 185: clear memory on pool deletion */
    if ( 0 != ::ftruncate(_fdout, 0) )
    {
      FWRNM("error {} truncating backing file, data may leak", errno);
    }
    syncfs(_fdout);
    close(_fdout);
  }
}

status_t Pool_instance::put(string_view_key key,
			    const void *value,
			    const size_t value_len,
			    unsigned int flags)
{
  if (!value || !value_len || value_len > _nsize) {
    PWRN("Map_store: invalid parameters (value=%p, value_len=%lu)", value, value_len);
    return E_INVAL;
  }

  common::string_view key_svc(common::pointer_cast<char>(key.data()), key.size());
  CFLOGM(2, "({}) {}", key_svc, common::string_view(static_cast<const char *>(value), value_len));

  RWLock_guard guard(_map_lock, common::RWLock_guard::WRITE);

  write_touch(); /* this could be early, but over-conservative is ok */

  std::lock_guard g{_mm_plugin_mutex}; /* aac, _mm_plugin.aligned_allocate, aal */
  string_t k(key.data(), key.length(), aac);

  auto i = _map->find(k);

  if (i != _map->end()) {

    if (flags & IKVStore::FLAGS_DONT_STOMP) {
      PWRN("put refuses to stomp (%*.s)", int(key.size()), common::pointer_cast<char>(common::pointer_cast<char>(key.data())));
      return IKVStore::E_KEY_EXISTS;
    }

    /* take lock */
    int rc;
    if((rc = (*_map)[k]._value_lock->write_trylock()) != 0) {
      PWRN("put refuses, already locked (%d)",rc);
      assert(rc == EBUSY);
      return E_LOCKED;
    }

    auto &p = i->second;

    if (p._length == value_len) {
      memcpy(p._ptr, value, value_len);
    }
    else {
      /* different size, reallocate */
      auto p_to_free = p._ptr;
      auto len_to_free = p._length;

      CFLOGM(3, "allocating {} bytes alignment {}", value_len, choose_alignment(value_len));

      if(_mm_plugin.aligned_allocate(value_len, choose_alignment(value_len),&p._ptr) != S_OK)
        throw General_exception("plugin aligned_allocate failed");

      memcpy(p._ptr, value, value_len);

      /* update entry */
      i->second._length = value_len;
      i->second._ptr = p._ptr;

      /* release old memory*/
      try {  _mm_plugin.deallocate(&p_to_free, len_to_free);      }
      catch(...) {  throw Logic_exception("unable to release old value memory");   }
    }

    wmb();
    i->second._tsc.update(); /* update timestamp */

    /* release lock */
    (*_map)[k]._value_lock->unlock();
  }
  else { /* key does not already exist */

    CFLOGM(3, "allocating {} bytes alignment {}", value_len, choose_alignment(value_len));

    void * buffer = nullptr;
    if(_mm_plugin.aligned_allocate(value_len, choose_alignment(value_len), &buffer) != S_OK)
      throw General_exception("memory plugin aligned_allocate failed");

    memcpy(buffer, value, value_len);
    //    common::RWLock * p = new (aal.allocate(1, DEFAULT_ALIGNMENT)) common::RWLock();
    common::RWLock * p = new (aal.allocate(1)) common::RWLock();

    /* create map entry */
    _map->try_emplace(k, buffer, value_len, p);
  }

  return S_OK;
}

status_t Pool_instance::get(const string_view_key key,
                            void *&out_value,
                            size_t &out_value_len)
{
  common::string_view key_svc(common::pointer_cast<char>(key.data()), key.size());
  CFLOGM(1, "get({},{},{})", key_svc, out_value, out_value_len);

  RWLock_guard guard(_map_lock);

  std::lock_guard g{_mm_plugin_mutex}; /* aac */
  string_t k(key.data(), aac);
  auto i = _map->find(k);

  if (i == _map->end()) return IKVStore::E_KEY_NOT_FOUND;

  /* if out_value is provided as non-NULL, and its large enough
     then use it. If it is not large enough, then send back 
     the size needed.
  */
  auto buffer_len = i->second._length;
  out_value_len = i->second._length;

  if(out_value) {
    if(buffer_len < i->second._length) return E_INSUFFICIENT_BUFFER;
  }
  else {
    /* result memory allocated with ::malloc */
    out_value = malloc(out_value_len);
  }

  if ( out_value == nullptr )  {
    PWRN("Map_store: malloc failed");
    return IKVStore::E_TOO_LARGE;
  }

  memcpy(out_value, i->second._ptr, i->second._length);
  return S_OK;
}

status_t Pool_instance::get_direct(const string_view_key key,
                                   void *out_value,
                                   size_t &out_value_len)
{
  common::string_view key_svc(common::pointer_cast<char>(key.data()), key.size());
  CFLOGM(1, " key=({}) ", key_svc);

  if (out_value == nullptr || out_value_len == 0)
    throw API_exception("invalid parameter");

  RWLock_guard guard(_map_lock);
  std::lock_guard g{_mm_plugin_mutex}; /* aac */
  string_t k(key.data(), key.size(), aac);
  auto i = _map->find(k);

  if (i == _map->end()) {
    if (debug_level()) PERR("Map_store: error key not found");
    return IKVStore::E_KEY_NOT_FOUND;
  }

  if (out_value_len < i->second._length) {
    if (debug_level()) PERR("Map_store: error insufficient buffer");

    return E_INSUFFICIENT_BUFFER;
  }

  out_value_len = i->second._length; /* update length */
  memcpy(out_value, i->second._ptr, i->second._length);

  return S_OK;
}

status_t Pool_instance::get_attribute(const IKVStore::Attribute attr,
                                      std::vector<uint64_t> &out_attr,
                                      const string_view_key key)
{
  switch (attr) {
  case IKVStore::Attribute::MEMORY_TYPE: {
    out_attr.push_back(IKVStore::MEMORY_TYPE_DRAM);
    break;
  }
  case IKVStore::Attribute::VALUE_LEN: {
    if (key.data() == nullptr) return E_INVALID_ARG;
    RWLock_guard guard(_map_lock);
    std::lock_guard g{_mm_plugin_mutex}; /* aac */
    string_t k(key.data(), key.size(), aac);
    auto i = _map->find(k);
    if (i == _map->end()) return IKVStore::E_KEY_NOT_FOUND;
    out_attr.push_back(i->second._length);
    break;
  }
  case IKVStore::Attribute::WRITE_EPOCH_TIME: {
    RWLock_guard guard(_map_lock);
    std::lock_guard g{_mm_plugin_mutex}; /* aac */
    string_t k(key.data(), key.size(), aac);
    auto i = _map->find(k);
    if (i == _map->end()) return IKVStore::E_KEY_NOT_FOUND;
    out_attr.push_back(boost::numeric_cast<uint64_t>(i->second._tsc.to_epoch().seconds()));
    break;
  }
  case IKVStore::Attribute::COUNT: {
    out_attr.push_back(_map->size());
    break;
  }
  case IKVStore::Attribute::NUMA_MASK: {
    out_attr.push_back(_numa_node_mask.get64());
    break;
  }
  default:
    return E_INVALID_ARG;
  }

  return S_OK;
}

status_t Pool_instance::swap_keys(const string_view_key key0,
                                  const string_view_key key1)
{
  std::lock_guard g{_mm_plugin_mutex}; /* aac, twice */
  string_t k0(key0.data(), key0.length(), aac);
  auto i0 = _map->find(k0);
  if(i0 == _map->end()) return IKVStore::E_KEY_NOT_FOUND;

  string_t k1(key1.data(), key1.length(), aac);
  auto i1 = _map->find(k1);
  if(i1 == _map->end()) return IKVStore::E_KEY_NOT_FOUND;

  /* lock both k-v pairs */
  auto& left = i0->second;
  if(left._value_lock->write_trylock() != 0)
    return E_LOCKED;

  auto& right = i1->second;
  if(right._value_lock->write_trylock() != 0) {
    left._value_lock->unlock();
    return E_LOCKED;
  }

  /* swap keys */
  auto tmp_ptr = left._ptr;
  auto tmp_len = left._length;
  left._ptr = right._ptr;
  left._length = right._length;
  right._ptr = tmp_ptr;
  right._length = tmp_len;

  /* release locks */
  left._value_lock->unlock();
  right._value_lock->unlock();

  return S_OK;
}

status_t Pool_instance::lock_unguarded(const std::lock_guard<std::mutex> &
  , const string_view_key key,
                             IKVStore::lock_type_t type,
                             void *&out_value,
                             size_t &inout_value_len,
                             size_t alignment,
                             IKVStore::key_t& out_key,
                             const char ** out_key_ptr)
{

  void *buffer = nullptr;
  string_t k(key.data(), key.size(), aac);
  bool created = false;
  common::string_view key_svc(common::pointer_cast<char>(key.data()), key.size());

  auto i = _map->find(k);

  CFLOGM(1, "lock looking for key:({})", key_svc);

  if (i == _map->end()) { /* create value */

    /* lock API has semantics of create on demand */
    if (inout_value_len == 0) {
      out_key = IKVStore::KEY_NONE;
      CFLOGM(1, "could not on-demand allocate without length:({}) {}", key_svc, inout_value_len);
      return IKVStore::E_KEY_NOT_FOUND;
    }

    write_touch();

    CFLOGM(1, "is on-demand allocating:({}) {}", key_svc, inout_value_len);

    if(alignment == 0)
      alignment = choose_alignment(inout_value_len);

    if(_mm_plugin.aligned_allocate(inout_value_len, alignment, &buffer) != S_OK)
      throw General_exception("memory plugin alloc failed");

    if (buffer == nullptr)
      throw General_exception("Pool_instance::lock on-demand create allocate_memory failed (len=%lu)",
                              inout_value_len);
    created = true;

    CFLOGM(1, "creating on demand key=({}) len={}",
          key_svc, inout_value_len);

    common::RWLock * p = new (aal.allocate(1)) common::RWLock();

    CFLOGM(2, "created RWLock at {}", p);
    _map->try_emplace(k, buffer, inout_value_len, p);
  }

  CFLOGM(1, "lock call has got key {}", key_svc);

  if (type == IKVStore::STORE_LOCK_READ) {
    if((*_map)[k]._value_lock->read_trylock() != 0) {
      if(debug_level())
        FWRNM("key ({}) unable to take read lock", key_svc);

      out_key = IKVStore::KEY_NONE;
      return E_LOCKED;
    }
  }
  else if (type == IKVStore::STORE_LOCK_WRITE) {

    write_touch();

    if((*_map)[k]._value_lock->write_trylock() != 0) {
      if(debug_level())
        FWRNM("Map_store: key ({}) unable to take write lock", key_svc);

      out_key = IKVStore::KEY_NONE;
      return E_LOCKED;
    }

  }
  else throw API_exception("invalid lock type");

  out_value = (*_map)[k]._ptr;
  inout_value_len = (*_map)[k]._length;

  out_key = reinterpret_cast<IKVStore::key_t>((*_map)[k]._value_lock);

  /* C++11 standard: § 23.2.5/8

     The elements of an unordered associative container are organized
     into buckets. Keys with the same hash code appear in the same
     bucket. The number of buckets is automatically increased as
     elements are added to an unordered associative container, so that
     the average number of elements per bucket is kept below a
     bound. Rehashing invalidates iterators, changes ordering between
     elements, and changes which buckets elements appear in, but does
     not invalidate pointers or references to elements. For
     unordered_multiset and unordered_multimap, rehashing preserves
     the relative ordering of equivalent elements.
  */
  if(out_key_ptr) {
    auto element = _map->find(k);
    *out_key_ptr = common::pointer_cast<char>(element->first.data());
  }

  return created ? S_OK_CREATED : S_OK;
}

status_t Pool_instance::lock(const string_view_key key,
                             IKVStore::lock_type_t type,
                             void *&out_value,
                             size_t &inout_value_len,
                             size_t alignment,
                             IKVStore::key_t& out_key,
                             const char ** out_key_ptr)
{
  std::lock_guard g{_mm_plugin_mutex}; /* aac, and later _mm_plugin.aligned_allocate, aal */
  return lock_unguarded(g, key, type, out_value, inout_value_len, alignment, out_key, out_key_ptr);
}

status_t Pool_instance::unlock(IKVStore::key_t key_handle)
{
  if(key_handle == nullptr) {
    PWRN("Map_store: unlock argument key handle invalid (%p)",
         reinterpret_cast<void*>(key_handle));
    return E_INVAL;
  }

  /* TODO: how do we know key_handle is valid? */
  if(reinterpret_cast<common::RWLock *>(key_handle)->unlock() != 0) {
    PWRN("Map_store: bad parameter to unlock");
    return E_INVAL;
  }

  CFLOGM(2, "unlocked key (handle={})", key_handle);
  return S_OK;
}

status_t Pool_instance::erase(const string_view_key key)
{
  const common::string_view key_svc(common::pointer_cast<char>(key.data()), key.size());
  RWLock_guard guard(_map_lock, common::RWLock_guard::WRITE);
  std::lock_guard g{_mm_plugin_mutex}; /* aac, _mm_plugin.deallocate, aal */
  string_t k(key.data(), key.size(), aac);
  auto i = _map->find(k);

  if (i == _map->end()) return IKVStore::E_KEY_NOT_FOUND;

  if ( i->second._value_lock->write_trylock() != 0 ) { /* check pair is not locked */
    if(debug_level())
      FWRNM("key ({}) unable to take write lock", key_svc);

    return E_LOCKED;
  }

  write_touch();
  _map->erase(i);

  _mm_plugin.deallocate(&i->second._ptr, i->second._length);
  i->second._value_lock->unlock();
  i->second._value_lock->~RWLock();
  aal.deallocate(i->second._value_lock, 1); //, DEFAULT_ALIGNMENT);

  return S_OK;
}

size_t Pool_instance::count() {
  RWLock_guard guard(_map_lock);
  return _map->size();
}

status_t Pool_instance::map(std::function<int(const string_view_key key,
                                              string_view_value value)> function)
{
  RWLock_guard guard(_map_lock);

  for (auto &pair : *_map) {
    const auto &val = pair.second;
    function(pair.first, string_view_value(static_cast<string_view_value::value_type *>(val._ptr), val._length));
  }

  return S_OK;
}

status_t Pool_instance::map(std::function<int(string_view_key key,
                                              string_view_value value,
                                              const common::tsc_time_t timestamp)> function,
                                              const common::epoch_time_t t_begin,
                                              const common::epoch_time_t t_end)
{
  RWLock_guard guard(_map_lock);

  common::tsc_time_t begin_tsc(t_begin);
  common::tsc_time_t end_tsc(t_end);

  for (auto &pair : *_map) {
    const auto &val = pair.second;

    if(val._tsc >= begin_tsc && (end_tsc == 0 || val._tsc <= end_tsc)) {
      if(function(pair.first,
                  string_view_value(static_cast<string_view_value::value_type *>(val._ptr), val._length),
                  val._tsc) < 0) {
        return S_MORE; /* break out of the loop if function returns < 0 */
      }
    }
  }

  return S_OK;
}


status_t Pool_instance::map_keys(std::function<int(string_view_key key)> function)
{
  RWLock_guard guard(_map_lock);

  for (auto &pair : *_map) function(pair.first);

  return S_OK;
}

status_t Pool_instance::resize_value(const string_view_key key,
                                     const size_t new_size,
                                     const size_t alignment)
{
  const common::string_view key_svc(common::pointer_cast<char>(key.data()), key.size());

  CFLOGM(1, "resize_value (key={}, new_size={}, align={}",
        key_svc, new_size, alignment);

  if (new_size == 0) return E_INVAL;

  RWLock_guard guard(_map_lock);

  std::lock_guard g{_mm_plugin_mutex}; /* aac, _mm_plugin.aligned_allocate */
  auto i = _map->find(string_t(key.data(), key.size(), aac));

  if (i == _map->end()) return IKVStore::E_KEY_NOT_FOUND;
  if (i->second._length == new_size) {
    CFLOGM(2, "resize_value request for same size! {}", new_size);
    return E_INVAL;
  }

  write_touch();

  /* perform resize */
  void * buffer = nullptr;
  if(_mm_plugin.aligned_allocate(new_size, alignment, &buffer) != S_OK)
    throw General_exception("memory plufin aligned_allocate failed");

  /* lock KV-pair */
  void *out_value;
  size_t inout_value_len;
  IKVStore::key_t out_key_handle = IKVStore::KEY_NONE;

  status_t s = lock_unguarded(g, key,
                    IKVStore::STORE_LOCK_WRITE,
                    out_value,
                    inout_value_len,
                    alignment,
                    out_key_handle,
                    nullptr);

  if (out_key_handle == IKVStore::KEY_NONE) {
    CFLOGM(2, "bad lock result {}", 0);
    return E_INVAL;
  }

  CFLOGM(2, "resize_value locked key-value pair", 0);

  size_t size_to_copy = std::min<size_t>(new_size, boost::numeric_cast<size_t>(i->second._length));

  memcpy(buffer, i->second._ptr, size_to_copy);

  /* free previous memory */
  _mm_plugin.deallocate(&i->second._ptr, i->second._length);

  i->second._ptr = buffer;
  i->second._length = new_size;

  /* release lock */
  if(unlock(out_key_handle) != S_OK)
    throw General_exception("unlock in resize failed");

  CFLOGM(2, "resize_value re-unlocked key-value pair", 0);
  return s;
}

status_t Pool_instance::get_pool_regions(nupm::region_descriptor::address_map_t &out_regions)
{
  if (_regions.empty())
    return E_INVAL;

  for (const auto &region : _regions)
    out_regions.push_back(nupm::region_descriptor::address_map_t::value_type
                          (common::make_byte_span(region->iov_base, region->iov_len)));
  return S_OK;
}

status_t Pool_instance::grow_pool(const size_t increment_size,
                                  size_t &reconfigured_size)
{
  if (increment_size <= 0)
    return E_INVAL;

  size_t rounded_increment_size = round_up_page(increment_size);

  auto new_region = allocate_region_memory(rounded_increment_size);
  std::lock_guard g{_mm_plugin_mutex};
  _mm_plugin.add_managed_region(new_region->iov_base, new_region->iov_len);
  _regions.push_back(std::move(new_region));
  reconfigured_size = _nsize;
  return S_OK;
}

status_t Pool_instance::free_pool_memory(const void *addr, const size_t size) {

  if (!addr || _regions.empty())
    return E_INVAL;

  std::lock_guard g{_mm_plugin_mutex};
  if(size)
    _mm_plugin.deallocate(const_cast<void **>(&addr), size);
  else
    _mm_plugin.deallocate_without_size(const_cast<void **>(&addr));

  /* the region memory is not freed, only memory in region */
  return S_OK;
}

status_t Pool_instance::allocate_pool_memory(const size_t size,
                                             const size_t alignment,
                                             void *&out_addr) {

  if (size == 0 || size > _nsize || _regions.empty()) {
    PWRN("Map_store: invalid %s request", __func__);
    return E_INVAL;
  }

  if ( (alignment & (alignment-1)) != 0 )
  {
    PWRN("Map_store: invalid %s alignment 0x%zx (neither 0 nor a power of 2)", __func__, alignment);
    return IKVStore::E_BAD_ALIGNMENT;
  }

  try {
    /* we can't fully support alignment choice */
    out_addr = 0;

    std::lock_guard g{_mm_plugin_mutex};
    if( _mm_plugin.aligned_allocate(size, (alignment > 0) && (size % alignment == 0) ?
                                    alignment : choose_alignment(size), &out_addr) != S_OK)
      throw General_exception("memory plugin aligned_allocate failed");

    CFLOGM(1, "allocated pool memory ({} {})", out_addr, size);
  }
  catch(...) {
    PWRN("Map_store: unable to allocate (%lu) bytes aligned by %lu", size, choose_alignment(size));
    return E_INVAL;
  }

  return S_OK;
}


auto Pool_instance::open_pool_iterator() -> IKVStore::pool_iterator_t
{
  auto it = _iterators.insert(std::make_unique<Iterator>(this));
  return reinterpret_cast<IKVStore::pool_iterator_t>(it.first->get());
}

status_t Pool_instance::deref_pool_iterator(IKVStore::pool_iterator_t iter,
                                            const common::epoch_time_t t_begin,
                                            const common::epoch_time_t t_end,
                                            IKVStore::pool_reference_t& ref,
                                            bool& time_match,
                                            bool increment)
{
  const auto i = reinterpret_cast<Iterator*>(iter);
  if(_iterators.count(i) != 1) return E_INVAL;
  if(i->is_end()) return E_OUT_OF_BOUNDS;
  if(!i->check_mark(_writes)) return E_ITERATOR_DISTURBED;

  common::tsc_time_t begin_tsc(t_begin);
  common::tsc_time_t end_tsc(t_end);

  auto r = i->_iter;
  ref.key = r->first.data();
  ref.key_len = r->first.length();
  ref.value = r->second._ptr;
  ref.value_len = r->second._length;

  ref.timestamp = r->second._tsc.to_epoch();

  /* leave condition in timestamp cycles for better accuracy */
  try {
    time_match = (r->second._tsc >= begin_tsc) && (end_tsc == 0 || r->second._tsc <= end_tsc);
  }
  catch(...) {
    PWRN("bad time parameter");
    return E_INVAL;
  }

  if(increment) {
    try {
      i->_iter++;
    }
    catch(...) {
      return E_ITERATOR_DISTURBED;
    }
  }

  return S_OK;
}

status_t Pool_instance::close_pool_iterator(IKVStore::pool_iterator_t iter)
{
  const auto it = _iterators.find(reinterpret_cast<Iterator *>(iter));
  if (it == _iterators.end()) return E_INVAL;
  _iterators.erase(it);
  return S_OK;
}

std::unique_ptr<region_memory> Pool_instance::allocate_region_memory(size_t size)
{
  std::unique_ptr<region_memory> rm;
  assert(size > 0);

  assert(size % PAGE_SIZE == 0);

  auto prot = PROT_READ | PROT_WRITE;
  auto flags = MAP_SHARED;
  /* create space in file */
  if ( 0 <= _fdout && ftruncate(_fdout, _nsize + size) == 0 )
  {
    auto p = mmap(reinterpret_cast<char *>(0xff00000000) + _nsize, /* help debugging */
             size,
             prot,
             flags, /* paging means no MAP_LOCKED */
             _fdout, /* file */
             _nsize /* offset */);
    if (p != MAP_FAILED)
    {
      FINF("using backing file for {} MiB", REDUCE_MB(size));
      rm = std::make_unique<region_memory_mmap>(debug_level(), p, size);
    }
  }

  if (! rm) {
    if ( numa_bitmask_weight(_numa_node_mask.get()) == 0 )
    {
      auto addr = reinterpret_cast<char *>(0x800000000) + _nsize; /* help debugging */
      /* memory to be freed with munmap */
      auto p = ::mmap(addr,
             size,
             prot,
             flags | MAP_ANONYMOUS | (needs_pinned_pages ? MAP_LOCKED : 0),
             -1, /* file */
             0 /* offset */);

      if ( p == MAP_FAILED ) {
        auto e = errno;
        std::ostringstream msg;
        msg << __FILE__ << " allocate_region_memory mmap failed on DRAM for region allocation"
            << " size=" << std::dec << size << " :" << strerror(e);
        throw General_exception("%s", msg.str().c_str());
      }
      rm = std::make_unique<region_memory_mmap>(debug_level(), p, size);
    }
    else
    {
      rm = std::make_unique<region_memory_numa_pin>(debug_level(), size, _numa_node_mask.get(), needs_pinned_pages);
    }
  }

#if 0
  if(madvise(p, size, MADV_DONTFORK) != 0)
    throw General_exception("madvise 'don't fork' failed unexpectedly (%p %lu)", p, size);
#endif

  CFLOGM(1, "allocated_region_memory ({},{})", rm->iov_base, size);
  _nsize += size;
  return rm;
}
