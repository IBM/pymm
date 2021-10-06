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


#ifndef __NUPM_DAX_DATA_H__
#define __NUPM_DAX_DATA_H__

#include <libpmem.h>
#include <city.h> /* CityHash */
#include <common/pointer_cast.h>
#include <common/string_view.h>
#include <common/types.h>
#include <common/utils.h>
#include <boost/icl/split_interval_map.hpp>
#include <gsl/span>
#include <algorithm>
#include <list>
#include <stdexcept>

namespace
{
	std::uint64_t make_uuid(const common::string_view name_)
	{
		if ( 255 < name_.size() )
		{
			throw std::invalid_argument("invalid file name (too long)");
		}
		auto region_id = ::CityHash64(name_.begin(), name_.size());
		if (region_id == 0) throw std::invalid_argument("invalid region_id (and extreme bad luck!)");
		return region_id;
	}
}

namespace nupm
{
static std::uint32_t constexpr DM_REGION_MAGIC = 0xC0070000;
static unsigned constexpr DM_REGION_NAME_MAX_LEN = 1024;
static std::uint32_t constexpr DM_REGION_VERSION = 3;
static unsigned constexpr dm_region_log_grain_size = DM_REGION_LOG_GRAIN_SIZE; // log2 granularity (CMake default is 25, i.e. 32 MiB)

class DM_undo_log {
  static constexpr unsigned MAX_LOG_COUNT = 4;
  static constexpr unsigned MAX_LOG_SIZE  = 272;
  struct log_entry_t {
    byte   log[MAX_LOG_SIZE];
    void * ptr;
    size_t length; /* zero indicates log freed */
  };

  void log(void *ptr, size_t length)
  {
    assert(length > 0);
    assert(ptr);

    if (length > MAX_LOG_SIZE)
      throw std::invalid_argument("log length exceeds max. space");

    for (unsigned i = 0; i < MAX_LOG_COUNT; i++) {
      if (_log[i].length == 0) {
        _log[i].length = length;
        _log[i].ptr    = ptr;
        pmem_memcpy_nodrain(_log[i].log, ptr, length);
        // TODO
        //memcpy(_log[i].log, ptr, length);
        //mem_flush(&_log[i], sizeof(log_entry_t));
        return;
      }
    }
    throw API_exception("undo log full");
  }

 public:
  template <typename T>
    void log(T *ptr)
    {
      static_assert(sizeof *ptr <= MAX_LOG_SIZE, "object too big to log");
      log(ptr, sizeof *ptr);
    }

  void clear_log()
  {
    for (unsigned i = 0; i < MAX_LOG_COUNT; i++) _log[i].length = 0;
  }

  void check_and_undo()
  {
    for (unsigned i = 0; i < MAX_LOG_COUNT; i++) {
      if (_log[i].length > 0) {
        PLOG("undo log being applied (ptr=%p, len=%lu).", _log[i].ptr,
             _log[i].length);
        // TODO
        pmem_memcpy_persist(_log[i].ptr, _log[i].log, _log[i].length);
        //memcpy(_log[i].ptr, _log[i].log, _log[i].length);
        //        mem_flush_nodrain(_log[i].ptr, _log[i].length);
        _log[i].length = 0;
      }
    }
  }

 private:
  log_entry_t _log[MAX_LOG_COUNT];
} __attribute__((packed));

class DM_region {
public:
  /* "grain" is the unit of suballocation within a region. It is a fixed power
   * of 2 common to all regions in the devdax namespace.
   */
  using grain_offset_t = uint32_t;
  grain_offset_t offset_grain;
  grain_offset_t length_grain;
  uint64_t region_id;
  /* File name. Saved and returned.
   * Internal logic uses region_id (file name hash) to reduce code changes.
   */
  char file_name[256];

 public:
  /* re-zeroing constructor */
  DM_region() : offset_grain(0), length_grain(0), region_id(0) { assert(check_aligned(this, 8)); }

  void initialize(size_t space_size, std::size_t grain_size)
  {
    offset_grain = 0;
    length_grain = boost::numeric_cast<uint32_t>(space_size / grain_size);
    region_id = 0; /* zero indicates free */
  }

  friend class DM_region_header;
} __attribute__((packed));

/* Note: "region" has at least two meanings:
 *  1. The space starting with, and described by, DM_region_header. ndctl calls this a namespace.
 e  2. The space described by DM_region
 */
class DM_region_header {
 private:
  static constexpr uint16_t DEFAULT_MAX_REGIONS = 1024;
  using string_view = common::string_view;

  uint32_t    _magic;         // 4
  uint32_t    _version;       // 8
  uint64_t    _device_size;   // 16
  uint32_t    _region_count;  // 20
  uint16_t    _log_grain_size; // 22
  uint16_t    _resvd;         // 24
  uint8_t     _padding[40];   // 64
  DM_undo_log _undo_log;

  /*
   * Following this data, there is
   * 1. region_table_base, immediately following (this + 1)
   * 2. arena_base, at ptr_cast<byte>(this) + grain_size. Code assumes
   *    (but does not verify) that DM_region_header plus all DM_region
   *    descriptors fit within a single grain.
   *
   * grain 0:
   *    "region_header"
   *    "region_table"
   * grains 1 and following:
   *    "arena"
   */

 public:
  auto grain_size() const { return std::size_t(1) << _log_grain_size; }

  /* Rebuilding constructor */
  DM_region_header(size_t device_size)
    : _magic(DM_REGION_MAGIC)
    , _version(DM_REGION_VERSION)
    , _device_size(device_size)
    , _region_count( (::pmem_flush(this, sizeof(DM_region_header)), DEFAULT_MAX_REGIONS) )
    , _log_grain_size(dm_region_log_grain_size)
    , _resvd()
    , _undo_log()
  {
    (void)_resvd; // unused
    (void)_padding; // unused
    DM_region *region_p = region_table_base();
    /* initialize first region with all capacity with size of space, and size of grain */
    region_p->initialize(device_size - grain_size(), grain_size());
    _undo_log.clear_log();
    region_p++;

    for (uint16_t r = 1; r < _region_count; r++) {
      new (region_p) DM_region();
      _undo_log.clear_log();
      region_p++;
    }
    major_flush();
  }

  void check_undo_logs()
  {
    _undo_log.check_and_undo();
  }

  void debug_dump()
  {
    PINF("DM_region_header:");
    PINF(
        " magic [0x%8x]\n version [%u]\n device_size [%lu]\n region_count [%u]",
        _magic, _version, _device_size, _region_count);
    PINF(" base [%p]", common::p_fmt(this));

    const auto regions = region_table_span();
    for (auto const & reg : regions ) {
      if (reg.region_id > 0) {
        PINF(" - USED: %lu (%lx-%lx)", reg.region_id,
             grain_to_bytes(reg.offset_grain),
             grain_to_bytes(reg.offset_grain + reg.length_grain) - 1);
        assert(reg.length_grain > 0);
      }
      else if (reg.length_grain > 0) {
        PINF(" - FREE: %lu (%lx-%lx)", reg.region_id,
             grain_to_bytes(reg.offset_grain),
             grain_to_bytes(reg.offset_grain + reg.length_grain) - 1);
      }
    }
  }

  void *get_region(string_view name_, size_t *out_size)
  {
    auto region_id = make_uuid(name_);

    const auto regions = region_table_span();
    for (auto const & reg : regions ) {
      if (reg.region_id == region_id) {
#if 0
        PLOG("%s: found matching region (%lx)", __func__, region_id);
#endif
        if (out_size) *out_size = grain_to_bytes(reg.length_grain);
        return arena_base() + grain_to_bytes(reg.offset_grain);
      }
    }
    return nullptr; /* not found */
  }

  void erase_region(string_view name_)
  {
    auto region_id = make_uuid(name_);

    const auto regions = region_table_span();
    for (auto & reg : regions ) {
      if (reg.region_id == region_id) {
        reg.region_id = 0; /* power-fail atomic */
        pmem_flush(&reg.region_id, sizeof(reg.region_id));
        return;
      }
    }
    throw std::runtime_error("region not found");
  }

  void *allocate_region(string_view name_, DM_region::grain_offset_t size_in_grain)
  {
    auto region_id = make_uuid(name_);

    const auto regions = region_table_span();
    for (auto & reg : regions ) {
      if (reg.region_id == region_id)
        throw std::bad_alloc();
    }

    bool found = false;
    for (DM_region & reg : regions)
    {
      /* If we have found a sufficiently large free region */
      if (reg.region_id == 0 && reg.length_grain >= size_in_grain) {
        if (reg.length_grain == size_in_grain) {
          /* exact match */
          void *rp = arena_base() + grain_to_bytes(reg.offset_grain);
          /* write file name to region */
          pmem_memcpy_persist(reg.file_name, name_.begin(), name_.size());
          pmem_memcpy_persist(&reg.file_name[name_.size()], "\0", 1);
          // claim region
          tx_atomic_write(&reg, region_id);
          return rp;
        }
        else {
          /* cut out */
          const uint32_t new_offset = reg.offset_grain;

          const auto changed_length = reg.length_grain - size_in_grain;
          const auto changed_offset = reg.offset_grain + size_in_grain;

	  /* reg_n is the new region for unallocated space */
          auto reg_n = std::find_if(regions.begin(), regions.end(), [] (const DM_region &r) { return r.region_id == 0 && r.length_grain == 0; });
          if ( reg_n != regions.end() )
          {
            void *rp = arena_base() + grain_to_bytes(new_offset);
            /* write file name to region */
            pmem_memcpy_persist(reg.file_name, name_.begin(), name_.size());
            pmem_memcpy_persist(&reg.file_name[name_.size()], "\0", 1);
            // claim region
            tx_atomic_write(&*reg_n, boost::numeric_cast<uint16_t>(changed_offset), boost::numeric_cast<uint16_t>(changed_length), &reg,
                            new_offset, size_in_grain, region_id);
            return rp;
          }
        }
      }
    }
    if (!found)
      throw General_exception("no more regions (size in grain=%u)", size_in_grain);

    throw General_exception("no spare slots");
  }

  size_t get_max_available() const
  {
    auto regions = region_table_span();
    auto max_grain_element =
      std::max_element(
        regions.begin()
	, regions.end()
        , [] (const DM_region &a, const DM_region &b) -> bool { return a.length_grain < b.length_grain; }
      );
    return grain_to_bytes( max_grain_element == regions.end() ? 0 : max_grain_element->length_grain);
  }

  inline size_t grain_to_bytes(unsigned grain) const { return size_t(grain) << _log_grain_size; }

  inline void major_flush()
  {
    pmem_flush(this, sizeof(DM_region_header) + (sizeof(DM_region) * _region_count));
  }

  bool check_magic() const
  {
    return (_magic == DM_REGION_MAGIC) && (_version == DM_REGION_VERSION);
  }

  std::list<std::string> names_list()
  {
    std::list<std::string> nl;
    auto regions = region_table_span();
    for ( const auto &r : regions )
    {
      if ( 0 != r.region_id )
      {
        nl.push_back(std::string(r.file_name));
      }
    }
    return nl;
  }

 private:
  void tx_atomic_write(DM_region *dst, uint64_t region_id)
  {
#pragma GCC diagnostic push
#if 9 <= __GNUC__
#pragma GCC diagnostic ignored "-Waddress-of-packed-member"
#endif
    _undo_log.log(&dst->region_id);
#pragma GCC diagnostic pop
    dst->region_id = region_id;
    pmem_flush(&dst->region_id, sizeof(region_id));
    _undo_log.clear_log();
  }

  void tx_atomic_write(DM_region *dst0, // offset0, size0, offset1, size1 all expressed in grains
                       uint32_t   offset0,
                       uint32_t   size0,
                       DM_region *dst1,
                       uint32_t   offset1,
                       uint32_t   size1,
                       uint64_t   region_id1)
  {
    _undo_log.log(dst0);
    _undo_log.log(dst1);

    dst0->offset_grain = offset0;
    dst0->length_grain = size0;
    pmem_flush(dst0, sizeof(DM_region));

    dst1->region_id = region_id1;
    dst1->offset_grain = offset1;
    dst1->length_grain = size1;
    pmem_flush(dst1, sizeof(DM_region));

    _undo_log.clear_log();
  }

  inline unsigned char *arena_base()
  {
    return common::pointer_cast<unsigned char>(this) + grain_size();
  }

  /* region descriptors immediately follow the DM_region_header. */
  inline DM_region *region_table_base() { return common::pointer_cast<DM_region>(this + 1); }
  inline const DM_region *region_table_base() const { return common::pointer_cast<const DM_region>(this + 1); }
  gsl::span<DM_region> region_table_span() { return gsl::span<DM_region>(region_table_base(), _region_count); }
  gsl::span<const DM_region> region_table_span() const { return gsl::span<const DM_region>(region_table_base(), _region_count); }

  inline DM_region *region(size_t idx)
  {
    if (idx >= _region_count) return nullptr;
    DM_region *p = static_cast<DM_region *>(region_table_base());
    return &p[idx];
  }

  void reset_header(size_t device_size)
  {
    _magic       = DM_REGION_MAGIC;
    _version     = DM_REGION_VERSION;
    _device_size = device_size;
    pmem_flush(this, sizeof(DM_region_header));
  }
} __attribute__((packed));

}  // namespace nupm

#endif  //__NUPM_DAX_DATA_H__
