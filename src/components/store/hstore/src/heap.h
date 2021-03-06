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


#ifndef MCAS_HSTORE_HEAP_H
#define MCAS_HSTORE_HEAP_H

#include "hstore_config.h"

#include <common/byte_span.h>
#include <common/exceptions.h> /* General_exception */
#include <common/string_view.h>

#include <array>
#include <cstddef> /* size_t, ptrdiff_t, uint64_t */
#include <memory> /* unique_ptr */

class heap_ephemeral;

namespace nupm
{
	struct dax_manager;
}

struct heap
{
protected:
	using byte_span = common::byte_span;
	using string_view = common::string_view;
	byte_span _pool0_full; /* entire extent of pool 0 */
	byte_span _pool0_heap; /* portion of pool 0 which can be used for the heap */
	unsigned _numa_node;
	std::size_t _more_region_uuids_size;
	std::array<std::uint64_t, 1024U> _more_region_uuids;

#if 0
	/* The grow should be common among all heaps */
	auto grow(
		heap_ephemeral *eph_
		, const std::unique_ptr<nupm::dax_manager> & dax_manager_
		, std::uint64_t uuid_
		, std::size_t increment_
	) -> std::size_t;
#endif

public:
	explicit heap(
		byte_span pool0_full
		, byte_span pool0_heap
		, unsigned numa_node
	);

	heap(const heap &) = default;
	heap &operator=(const heap &) = delete;

	~heap();
};

#endif
