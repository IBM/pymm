#ifndef __CCPM_VALUE_H__
#define __CCPM_VALUE_H__

#include <libpmem.h> /* pmem_memset */
#include <common/cycles.h>
#include <common/utils.h>

namespace ccpm
{

enum Structured_value_type : uint32_t {
  TYPE_STRING_TABLE=1,
  TYPE_RECORD=2,
  TYPE_POINTER=3,
  TYPE_UNKNOWN=0xff,
};
    
class Structured_value_base
{
private:

  static constexpr uint32_t MAGIC = 0xFEEDF00D;

  struct Persistent_header { /* we do this so we can have both volatile and non-volatile members */
    uint32_t   magic;
    uint32_t   type;
    size_t     size;
  } __attribute__((packed)) * _root;


protected:  
  
  Structured_value_base(void * base, size_t size, Structured_value_type type, bool force_init = false) :
    _root(reinterpret_cast<Persistent_header*>(base)), _size(size)
  {    
    if(base==nullptr || !check_aligned(base, 8))
      throw Constructor_exception("Structured_value_base requires 64-bit alignment");

    assert(_root);
    PLOG("Structured_value_base: _root=%p", common::p_fmt(_root));
    if(force_init || !check_integrity() || _root->size != size) {
      PINF("Structured_value_base: rebuilding (base=%p,size=%lu)", base, size);
      /* rebuild */
      _root->magic = MAGIC;
      _root->size = size;
      _root->type = type;
      flush_header();
      _rebuilt = true;
    }
    else {
      PINF("Structured_value_base: reconstituting (base=%p,size=%lu)", base, _root->size);
    }
    dump_info();
  }

  bool rebuilt() const { return _rebuilt; }

  iovec remaining_memory() const {
    return iovec{reinterpret_cast<void*>(&_root[1]), _size - sizeof(*this)};
  }

  void dump_info() {
    PINF("Structured_value_base: (size=%lu, type=%u)", _root->size, _root->type);
  }

private:
  
  bool check_integrity() const { return _root->magic == MAGIC; }
  void flush_header() { pmem_persist(_root, sizeof(Persistent_header)); }

protected:
  const size_t _size;
  bool _rebuilt = false;
};



/** 
 * Class to manage pointers to N versions
 *
 */
template <typename T>
class Cow_value_pointer : protected Structured_value_base
{
private:
  static constexpr unsigned VERSION_COUNT = 3;

public:
  struct Persistent_version {
    T * ptr;
    size_t length;
    uint64_t timestamp;
  } __attribute__((packed));

private:
  struct Persistent_header {    
    std::array<Persistent_version,VERSION_COUNT> slots;
  } __attribute__((packed)) * _root;
  
public:
  explicit Cow_value_pointer(void * base, size_t size, Structured_value_type type, bool force_init = false)
    : Structured_value_base(base,size,type,force_init),
      _root(reinterpret_cast<Persistent_header*>(Structured_value_base::remaining_memory().iov_base))
  {
    PLOG("Cow_value_pointer: (%p, %lu, %d, %s)", base, size, type, force_init ? "y":"n");
    if(size < sizeof(*this))
      throw Constructor_exception("insufficient memory for Cow_value_pointer");
    
    if(force_init || rebuilt()) {
      PLOG("Cow_value_pointer: rebuild zeroing");
      pmem_memset(_root, 0, sizeof(Persistent_header), 0);
      _rebuilt = true;
    }
    else {
      PLOG("Cow_value_pointer: using existing");
    }

    uint64_t most_recent_timestamp = _root->slots[0].timestamp;
    _most_recent_slot = 0;
    for(unsigned s=1; s < VERSION_COUNT; s++) {
      if(_root->slots[s].timestamp > most_recent_timestamp) {
        most_recent_timestamp = _root->slots[s].timestamp;
        _most_recent_slot = s;
      }
    }
  }

  iovec remaining_memory() {
    return iovec{reinterpret_cast<void*>(&_root[1]), _size - sizeof(*this)};
  }
  
  status_t get_version(unsigned slot, Persistent_version*& out_version) {
    if(slot >= VERSION_COUNT) return E_OUT_OF_BOUNDS;
    out_version = _root->slots[slot];
    return S_OK;
  }

  const Persistent_version * get_current_version() const {
    return &_root->slots[_most_recent_slot];
  }

  unsigned get_new_version_slot(Persistent_version *& out_version) {
    unsigned s = 0;
    unsigned oldest = 0;
    /* find oldest slot or timestamp == 0 */
    while(s < VERSION_COUNT) {
      const auto ts = _root->slots[s].timestamp;
      if(ts == 0) {
        out_version = &_root->slots[s];
        return s;
      }
      const auto oldest_ts = _root->slots[oldest].timestamp;
      if(oldest_ts == 0 || ts < oldest_ts)
        oldest = s;
      s++;
    }
    Persistent_version * v = &_root->slots[oldest];
    v->timestamp = 0; /* clear timestamp */
    pmem_persist(&v->timestamp, sizeof(uint64_t));    
    out_version = v;
    return oldest;
  }
   
  void atomic_commit_version(unsigned slot) {
    pmem_persist(&_root->slots[slot], sizeof(Persistent_version));
    _root->slots[slot].timestamp = rdtsc();
    pmem_persist(&_root->slots[slot].timestamp, sizeof(uint64_t));
    _most_recent_slot = slot;
  }

  void dump_info() const {
    unsigned index = 0;
    PINF("active slot:%u", _most_recent_slot);
    for(auto& s : _root->slots) {      
      PINF("slot[%u]: %p %lu %lu", index, s.ptr, s.length, s.timestamp);
      index++;
    }
  }

  bool rebuilt() const { return _rebuilt; }

private:
  bool _rebuilt = false;
  unsigned _most_recent_slot = 0;
};



} // ccpm namespace

#endif // __CCPM_VALUE_H__
