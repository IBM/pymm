#ifndef __NUPM_CCPM_STRING_TABLE_H__
#define __NUPM_CCPM_STRING_TABLE_H__

#include <common/byte_span.h>
#include <string>
#include <iterator>
#include <ccpm/immutable_allocator.h>

namespace ccpm
{

/** 
 * String table that is append only (write once, read many)
 * 
 */
template <class CharT = char>
class Immutable_string_table :
    private Immutable_allocator_base
{
  using byte_span = common::byte_span;
public:
  struct String_record
  {
    size_t length; /*< length of string data in CharT */
  } __attribute__((packed));

  // class iterator : public std::iterator<std::bidirectional_iterator_tag,
  //                                       CharT *>
  // {
  //   explicit iterator(long _num = 0) : num(_num) {}
  //   iterator& operator++() {num = TO >= FROM ? num + 1: num - 1; return *this;}
  //   iterator operator++(int) {iterator retval = *this; ++(*this); return retval;}
  //   bool operator==(iterator other) const {return num == other.num;}
  //   bool operator!=(iterator other) const {return !(*this == other);}
  //   reference operator*() const {return num;}
  // };

public:
#if 0
  Immutable_string_table(void * buffer, size_t buffer_size)
    : Immutable_allocator_base(buffer, buffer_size) {
  }
#endif
  Immutable_string_table(region_vector_t regions, bool force_init)
    : Immutable_allocator_base(regions, ccpm::accept_all, force_init) {
  }

  const char * add_string(const std::basic_string<CharT>& str)
  {
    size_t string_data_len = str.size() * sizeof(CharT);
    auto record_len = string_data_len + sizeof(CharT) + sizeof(String_record);
    auto record = static_cast<String_record*>(allocate(record_len));
    record->length = 0;
    CharT * dst = reinterpret_cast<CharT*>(&record[1]);
    memcpy(dst, str.data(), string_data_len);
    dst[str.size()] = '\0';
    pmem_flush(record, record_len);

    /* final write length to indicate string is fully written */
    record->length = str.size();
    pmem_flush(&record->length, sizeof(record->length));
    return dst;
  }

  const char * add_string(const char * c_str, size_t c_str_len) {
    auto record_len = c_str_len + sizeof(CharT) + sizeof(String_record);
    auto record = static_cast<String_record*>(allocate(record_len));
    record->length = 0;
    CharT * dst = reinterpret_cast<CharT*>(&record[1]);
    memcpy(dst, c_str, c_str_len);
    dst[c_str_len] = '\0';
    pmem_flush(record, record_len);

    /* final write length to indicate string is fully written */
    record->length = c_str_len;
    pmem_flush(&record->length, sizeof(record->length));
    return dst;
  }

  std::basic_string<CharT> read_string(void * addr) const
  {
    std::basic_string<CharT> result;
    auto hdr = reinterpret_cast<String_record*>(addr) - 1;
    result.assign(static_cast<CharT*>(addr), hdr->length);
    return result;
  }

  inline void expand(byte_span region) {
    Immutable_allocator_base::expand(region);
  }
   
  bool is_valid() const { return Immutable_allocator_base::is_valid(); }

  
};

}

#endif // __NUPM_CCPM_STRING_TABLE_H__
