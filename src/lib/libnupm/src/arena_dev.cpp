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

#include "arena_dev.h"

#include "dax_data.h"
#include <cinttypes>

arena_dev::arena_dev(const common::log_source &ls_, string_view path_, gsl::not_null<nupm::DM_region_header *> hdr)
  : arena(ls_)
  , _path(path_)
  , _hdr(hdr)
{}

void arena_dev::debug_dump() const
{
  _hdr->debug_dump();
}

auto arena_dev::region_get(const string_view id_) -> region_descriptor
{
  std::size_t len = 0;
  auto base = _hdr->get_region(id_, &len);
  region_descriptor::address_map_t v;
  if ( base != nullptr )
  {
    v.push_back(common::make_byte_span(base, len));
  }
  return region_descriptor(v);
}

auto arena_dev::region_create(const string_view id_, gsl::not_null<registry_memory_mapped *>, const std::size_t size) -> region_descriptor
{
  auto size_in_grains = boost::numeric_cast<nupm::DM_region::grain_offset_t>(div_round_up(size, _hdr->grain_size()));

  CPLOG(2, "%s::%s: rounding up to %" PRIu32 " grains (%" PRIu64 " MiB)", _cname, __func__,
       size_in_grains, REDUCE_MiB((1UL << DM_REGION_LOG_GRAIN_SIZE)*size_in_grains));

  return
    region_descriptor(
      region_descriptor::address_map_t(
        1
        , common::make_byte_span(
            _hdr->allocate_region(id_, size_in_grains)
            , size_in_grains * _hdr->grain_size()
          )
      )
    ); /* allocates n grains */
}

void arena_dev::region_erase(const string_view id_, gsl::not_null<registry_memory_mapped *>)
{
  _hdr->erase_region(id_);
}

void arena_dev::region_resize(
  gsl::not_null<space_registered *>  // mh
  , std::size_t // size
)
{
  /* Dax_manager does not implement resize, although it could support shrinking a region */
}

std::size_t arena_dev::get_max_available() { return _hdr->get_max_available(); }

std::list<std::string> arena_dev::names_list() const
{
	return _hdr->names_list();
}
