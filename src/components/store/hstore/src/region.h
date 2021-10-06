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


#ifndef MCAS_HSTORE_NUPM_REGION_H
#define MCAS_HSTORE_NUPM_REGION_H

/* requires persist_data_t definition */
#include "hstore_config.h"
#include "alloc_key.h" /* AK_ACTUAL */
#include "persist_data.h"

#include "heap_mr.h"
#include "heap_rc.h"
#include "heap_mc.h"
#include "heap_cc.h"
#include "heap_mm.h"

#include <nupm/region_descriptor.h>
#include <common/byte_span.h>
#include <common/pointer_cast.h>
#include <common/string_view.h>
#include <memory>
#include <sstream>
#include <stdexcept>

struct dax_manager;

template <
  typename PersistData /* persistent data for the hash table: impl::persist_data<allocator_segment_t, table_type> */
  , typename Heap /* heap_[cr]c */
>
  struct alignas(64) region
  {
    using heap_access_t = heap_access<Heap>;
  private:
    static constexpr std::uint64_t magic_value = Heap::magic_value(); // 0xc74892d72eed493a;
    using byte_span = common::byte_span;
    using string_view = common::string_view;
  public:
    using heap_type = Heap;
    using persist_data_type = PersistData;

  private:
    std::uint64_t magic;
    /* The hashed value of the string which names the region.
     * Preserved only to form the basis for new strings generated
     * for devdax grow.
     */
    std::uint64_t _uuid;
    /* The storage allocator for persistent data.
     * One of two types: hstore_rc (reconsituting allocator)
     *  or hstore_cc (crash-consistent allocator)
     */
    heap_type _heap;
    /*
     * Allocated persistent storage.
     */
    persist_data_type _persist_data;

  public:

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
    region(
      AK_ACTUAL
      unsigned debug_level
#if HEAP_MM
			, common::string_view plugin_path
#endif
      , std::uint64_t uuid_
      , std::size_t size_
      , std::size_t expected_obj_count
      , unsigned numa_node_
      , string_view id_
      , string_view backing_file_
    )
      : magic(0)
      , _uuid(uuid_)
      , _heap(
        debug_level
#if HEAP_MM
			, plugin_path
#endif
        , _persist_data.ase()
        , _persist_data.aspd()
        , _persist_data.aspk()
        , _persist_data.asx()
        , byte_span(common::make_byte_span(this, size_))
        , byte_span(common::make_byte_span(this+1, adjust_size(size_)))
        , numa_node_
        , id_
        , backing_file_
      )
      , _persist_data(
        AK_REF
        expected_obj_count
        , typename persist_data_type::allocator_type(make_heap_access())
    )
    {
      magic = magic_value;
      persister_nupm::persist(this, sizeof *this);
    }
#pragma GCC diagnostic pop

    /* The "reanimate" (or "reconsitute") constructor */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winit-self"
#pragma GCC diagnostic ignored "-Wuninitialized"
    region(
      unsigned debug_level
#if HEAP_MM
			, common::string_view plugin_path
#endif
      , const std::unique_ptr<dax_manager> & dax_manager_
      , string_view id_
      , string_view backing_file_
      , const byte_span *iov_addl_first_
      , const byte_span *iov_addl_last_
    )
      : magic(0)
      , _uuid(this->_uuid)
      , _heap(
        debug_level
#if HEAP_MM
				, plugin_path
#endif
        , dax_manager_
        , id_
        , backing_file_
        , iov_addl_first_
        , iov_addl_last_
        , this->_persist_data.ase()
        , this->_persist_data.aspd()
        , this->_persist_data.aspk()
        , this->_persist_data.asx()
      )
      , _persist_data(std::move(this->_persist_data))
    {
      magic = magic_value;
      persister_nupm::persist(this, sizeof *this);
			if ( _heap.is_crash_consistent() )
			{
				/* any old values in the allocation states have been queried, as needed, by
				 * the crash-consistent allocator. Reset all allocation states.
				 */
				this->_persist_data.ase()->reset();
				this->_persist_data.aspd()->reset();
				this->_persist_data.aspk()->reset();
				this->_persist_data.asx()->reset();
			}
    }
#pragma GCC diagnostic pop

    auto adjust_size(std::size_t sz_)
    {
      if ( sz_ < sizeof *this )
      {
        std::ostringstream s;
        s << "Have " << std::hex << std::showbase << sz_ << " bytes. Cannot create a persisted region from less than " << sizeof *this << " bytes";
        throw std::range_error(s.str());
      }
      return sz_ - sizeof *this;
    }

    heap_access_t make_heap_access() { return heap_access_t(&_heap); }
    persist_data_type &persist_data() { return _persist_data; }
    bool is_initialized() const noexcept { return magic == magic_value; }
    unsigned percent_used() const { return _heap.percent_used(); }
    void quiesce() { _heap.quiesce(); }
    nupm::region_descriptor get_regions() const
    {
      return _heap.regions();
    }
    auto grow(
      const std::unique_ptr<dax_manager> & dax_manager_
      , std::size_t increment_
    ) -> std::size_t
    {
      return _heap.grow(dax_manager_, _uuid, increment_);
    }
    /* region used by heap_cc follows */
  };

#endif
