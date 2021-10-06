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

#include "heap.h"

/* When used with ADO, this space apparently needs a 2MiB alignment.
 * 4 KiB alignment sometimes produces a disagreement between server and ADO mappings,
 * which manifests as incorrect key and data values as seen on the ADO side.
 */
heap::heap(
	byte_span pool0_full_
	, byte_span pool0_heap_
	, unsigned numa_node_
)
	: _pool0_full(pool0_full_)
	, _pool0_heap(pool0_heap_)
	, _numa_node(numa_node_)
	, _more_region_uuids_size(0)
	, _more_region_uuids()
{}

heap::~heap()
{}

#if 0
/* Common grow function. Not tested */
namespace
{
	using byte_span = common::byte_span;
	std::size_t region_size(const std::vector<byte_span> &v)
	{
		return
			std::accumulate(
				v.begin()
				, v.end()
				, std::size_t(0)
				, [] (std::size_t s, const byte_span iov) -> std::size_t
					{
						return s + ::size(iov);
					}
			);
	}
}

auto heap::grow(
	heap_ephemeral *eph_
	, const std::unique_ptr<nupm::dax_manager_abstract> & dax_manager_
	, std::uint64_t uuid_
	, std::size_t increment_
) -> std::size_t
{
	if ( 0 < increment_ )
	{
		if ( _more_region_uuids_size == _more_region_uuids.size() )
		{
			throw std::bad_alloc(); /* max # of regions used */
		}
		const auto hstore_grain_size = std::size_t(1) << (HSTORE_LOG_GRAIN_SIZE);
		auto size = ( (increment_ - 1) / hstore_grain_size + 1 ) * hstore_grain_size;

		/* fsdax regions can be resized. Try that first */

		auto grown = false;
		{
			const auto old_regions = eph_->get_managed_regions();
			const auto &old_region_list = old_regions.address_map();
			const auto old_list_size = old_region_list.size();
			const auto old_size = region_size(old_region_list);
			if ( old_regions.id().size() != 0 )
			{
				eph_->set_managed_regions(dax_manager_->resize_region(old_regions.id(),  _numa_node, old_size + increment_));
			}
			const auto new_region_list = eph_->get_managed_regions().address_map();
			const auto new_size = region_size(new_region_list);
			const auto new_list_size = new_region_list.size();

			if ( old_size <  new_size )
			{
				for ( auto i = old_list_size; i != new_list_size; ++i )
				{
					const auto &r = new_region_list[i];
					eph_->add_managed_region(r, r);
					hop_hash_log<trace_heap_summary>::write(
						LOG_LOCATION
						, " pool ", ::base(r), " .. ", ::end(r)
						, " size ", ::size(r)
						, " grow"
					);
				}
				grown = true;
			}
		}

		/* devdax regions cannot be resized. Request a secondary region. */

		if ( ! grown )
		{
			/* Generate next string ID from a known base value (string rep of
			 * initial UUID plus number of secondary regions). The string will
			 * probably map to an unused UUID, If not, increment and try again.
			 * The increment value which is ultimately accepeted by create_region
			 * as creating a previously unused string will be stored in the slot
			 * in _more_region_uuids.
			 */
			auto &uuid_incr = _more_region_uuids[_more_region_uuids_size];
			uuid_incr = _more_region_uuids_size;
			for ( ; uuid_incr != std::numeric_limits<std::uint64_t>::max(); ++uuid_incr )
			{
				auto string_id_new = std::to_string(uuid_ + uuid_incr);

				try
				{
					/* Note: crash between here and "Slot persist done" may cause dax_manager_
					 * to leak the region.
					 */
					std::vector<byte_span> rv = dax_manager_->create_region(string_id_new, _numa_node, size).address_map();
					{
						persister_nupm::persist(&uuid_incr, sizeof uuid_incr);
						/* Slot persist done */
					}
					{
						++_more_region_uuids_size;
						persister_nupm::persist(&_more_region_uuids_size, _more_region_uuids_size);
					}
					for ( const auto & r : rv )
					{
						eph_->add_managed_region(r, r);
						hop_hash_log<trace_heap_summary>::write(
							LOG_LOCATION
							, " pool ", ::base(r), " .. ", ::end(r)
							, " size ", ::size(r)
							, " grow"
						);
					}
					break;
				}
				catch ( const std::bad_alloc & )
				{
					/* probably means that the uuid is in use */
				}
				catch ( const General_exception & )
				{
					/* probably means that the space cannot be allocated */
					throw std::bad_alloc();
				}
			}
			if ( uuid_incr == std::numeric_limits<std::uint64_t>::max() )
			{
				throw std::bad_alloc(); /* no more UUIDs */
			}
		}
	}
	return eph_->capacity();
}
#endif
