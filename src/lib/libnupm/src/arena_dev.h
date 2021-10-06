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

#ifndef _MCAS_NUPM_ARENA_DEV__
#define _MCAS_NUPM_ARENA_DEV__

#include "arena.h"

#include <common/logging.h>

namespace nupm
{
	class DM_region_header;
}

/* An arena implemented by a DM_region_header. */
struct arena_dev
  : arena
{
private:
  constexpr static const char *_cname = "arena_dev";
  std::string _path;
  gsl::not_null<nupm::DM_region_header *> _hdr;
public:
  arena_dev(const common::log_source &ls, string_view path, gsl::not_null<nupm::DM_region_header *> hdr);
  region_descriptor region_get(string_view id) override;
  region_descriptor region_create(string_view id, gsl::not_null<registry_memory_mapped *> mh, std::size_t size) override;
  void region_resize(gsl::not_null<space_registered *> mh, std::size_t size) override;
  void region_erase(string_view id, gsl::not_null<registry_memory_mapped *> mh) override;
  std::size_t get_max_available() override;
  bool is_file_backed() const override { return false; }
  void debug_dump() const override;
  std::string describe() const override { return _path; }
  std::list<std::string> names_list() const override;
};

#endif
