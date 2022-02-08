/*
   Copyright [2020] [IBM Corporation]
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

#ifndef _MCAS_NUPM_ARENA_
#define _MCAS_NUPM_ARENA_

#include <nupm/region_descriptor.h>
#include <common/logging.h>
#include <common/string_view.h>
#include <gsl/pointers> /* not_null */
#include <sys/uio.h> /* ::iovec */

#include <cstddef>
#include <list>

namespace nupm
{
	struct registry_memory_mapped;
	struct space_registered;
}

struct arena
  : private common::log_source
{
  using region_descriptor = nupm::region_descriptor;
  using registry_memory_mapped = nupm::registry_memory_mapped;
  using space_registered = nupm::space_registered;
  using string_view = common::string_view;

  arena(const common::log_source &ls) : common::log_source(ls) {}
  virtual ~arena() {}
  virtual void debug_dump() const = 0;
  virtual region_descriptor region_get(string_view id) = 0;
  virtual region_descriptor region_create(string_view id, gsl::not_null<registry_memory_mapped *> mh, std::size_t size) = 0;
  virtual void region_resize(gsl::not_null<space_registered *> mh, std::size_t size) = 0;
  /* It is unknown whether region_erase may be used on an open region.
   * arena_fs assumes that it may, just as ::unlink can be used against
   * an open file.
   */
  virtual void region_erase(string_view id, gsl::not_null<registry_memory_mapped *> mh) = 0;
  virtual std::size_t get_max_available() = 0;
  virtual bool is_file_backed() const = 0;
  virtual std::string describe() const = 0;
  virtual std::list<std::string> names_list() const = 0;
protected:
  using common::log_source::debug_level;
};

#endif
