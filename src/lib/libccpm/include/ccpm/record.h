#ifndef __CCPM_RECORD_H__
#define __CCPM_RECORD_H__

#include <memory>
#include <ccpm/value.h>

namespace ccpm
{

class Record_type {
public:
  const static Structured_value_type type_id = Structured_value_type::TYPE_RECORD;
};

template <class Allocator_T>
class Versioned_record : protected Cow_value_pointer<Record_type>
{
public:
  explicit Versioned_record(void * base, size_t size)
    : Cow_value_pointer<Record_type>(base,size,Record_type::type_id)
  {
    if(Cow_value_pointer<Record_type>::rebuilt()) {
      PLOG("Versioned record need rebuilding");
      Persistent_version * pv = nullptr;
      auto slot = get_new_version_slot(pv);
      assert(pv);
      pv->ptr = reinterpret_cast<Record_type*>(0xFFF); //allocator.allocate(100));
      pv->length = 100;
      atomic_commit_version(slot);
    }
    dump_info();
    auto rm = remaining_memory();
    PLOG("allocator memory=%p", rm.iov_base, rm.iov_len);
    //    _allocator = new Allocator_T(next_base()
  }

  
private:
  
  Allocator_T * _allocator;
  //  Immutable_allocator_base alloc(ptr, size);
};

}
#endif
