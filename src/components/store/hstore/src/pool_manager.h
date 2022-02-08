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


#ifndef _MCAS_HSTORE_POOL_MANAGER_H
#define _MCAS_HSTORE_POOL_MANAGER_H

#include <api/kvstore_itf.h> /* status_t */

#include "alloc_key.h" /* AK_FORMAL */
#include "pool_error.h"

#include <common/logging.h> /* log_source */
#include <common/string_view.h>
#include <nupm/region_descriptor.h>
#include <gsl/pointers>
#include <sys/uio.h>
#include <cstddef>
#include <functional>
#include <list>
#include <string>
#include <system_error>

struct pool_path;

struct dax_manager;

template <typename Pool>
  struct pool_manager
    : protected common::log_source
  {
    using string_view = common::string_view;
    pool_manager(
			unsigned debug_level_
		) : common::log_source(debug_level_)
		{}

    virtual ~pool_manager() {}

    virtual void pool_create_check(const std::size_t size_) = 0;

    virtual void pool_close_check(const string_view) = 0;

    virtual nupm::region_descriptor pool_get_regions(const Pool &) const = 0;

    /*
     * throws pool_error if create_region fails
     */
    virtual auto pool_create_1(
      const pool_path &path_
      , std::size_t size_
    ) -> nupm::region_descriptor = 0;

    virtual auto pool_create_2(
      AK_FORMAL
      const nupm::region_descriptor & rac
      , component::IKVStore::flags_t flags
      , std::size_t expected_obj_count
    ) -> std::unique_ptr<Pool> = 0;

    virtual auto pool_open_1(
      const pool_path &path_
    ) -> nupm::region_descriptor = 0;

    virtual auto pool_open_2(
      AK_FORMAL
      const nupm::region_descriptor & access_
      , component::IKVStore::flags_t flags_
    ) -> std::unique_ptr<Pool> = 0;

    virtual void pool_delete(const pool_path &path) = 0;
    virtual std::list<std::string> names_list() const = 0;
    virtual const std::unique_ptr<dax_manager> & get_dax_manager() const = 0;
  };

#endif
