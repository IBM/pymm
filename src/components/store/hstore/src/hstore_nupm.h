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


#ifndef _MCAS_HSTORE_NUPM_H
#define _MCAS_HSTORE_NUPM_H

#include "pool_manager.h"

#include "alloc_key.h" /* AK_FORMAL */
#include "hstore_nupm_types.h"
#include "hstore_open_pool.h"
#include "persister_nupm.h"

#include <common/string_view.h>
#include <gsl/pointers>

#include <cinttypes> /* PRIx64 */
#include <cstdlib> /* getenv */
#include <string>
#include <vector>

struct dax_manager;

template <typename PersistData, typename Heap>
  struct region;

#pragma GCC diagnostic push
/* Note: making enable_shared_from_this private avoids the non-virtual-dtor error but
 * generates a different error with no error text (G++ 5.4.0)
 */
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"

/* Region is region<persist_data_t, heap_rc>, Table is hstore::table_t Allocator is table_t::allocator_type, LockType is hstore::locK_type_t */
template <typename Region, typename Table, typename Allocator, typename LockType>
  struct hstore_nupm
    : public pool_manager<::open_pool<non_owner<Region>>>
  {
    using region_type = Region;
  private:
    using table_t = Table;
    using allocator_t = Allocator;
    using lock_type_t = LockType;
    using string_view = common::string_view;
  public:
    using open_pool_handle = ::open_pool<non_owner<region_type>>;
    using base = pool_manager<open_pool_handle>;
  private:
#if HEAP_MM
		std::string _plugin_path;
#endif
    std::unique_ptr<dax_manager> _dax_manager;
    unsigned _numa_node;

    static unsigned name_to_numa_node(const string_view name);
  public:
    hstore_nupm(
			unsigned debug_level
#if HEAP_MM
			, const string_view plugin_path
#endif
			, const string_view owner
			, const string_view name
			, std::unique_ptr<dax_manager> mgr
		);

    virtual ~hstore_nupm();

    const std::unique_ptr<dax_manager> & get_dax_manager() const override { return _dax_manager; }
    void pool_create_check(std::size_t) override;

    auto pool_create_1(
      const pool_path &path
      , std::size_t size
    ) -> nupm::region_descriptor override;

    auto pool_create_2(
      AK_FORMAL
      const nupm::region_descriptor &rac
      , component::IKVStore::flags_t flags
      , std::size_t expected_obj_count
    ) -> std::unique_ptr<open_pool_handle> override;

    nupm::region_descriptor pool_open_1(
      const pool_path &path
    ) override;

    auto pool_open_2(
      AK_FORMAL
      const nupm::region_descriptor & v
      , component::IKVStore::flags_t flags
    ) -> std::unique_ptr<open_pool_handle> override;

    void pool_close_check(const string_view) override;

    void pool_delete(const pool_path &path_) override;

    std::list<std::string> names_list() const override;

    /* ERROR: want get_pool_regions(<proper type>, std::vector<::iovec>&) */
    nupm::region_descriptor pool_get_regions(const open_pool_handle &) const override;
  };
#pragma GCC diagnostic pop

#include "hstore_nupm.tcc"

#endif
