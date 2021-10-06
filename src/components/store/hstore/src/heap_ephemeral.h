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


#ifndef MCAS_HSTORE_HEAP_EPHEMERAL_H
#define MCAS_HSTORE_HEAP_EPHEMERAL_H

#include <common/logging.h>

#include "hstore_config.h"

#include <common/byte_span.h>
#include <nupm/region_descriptor.h>
#include <cstddef> /* size_t */

struct heap_ephemeral
	: protected common::log_source
{
protected:
	virtual ~heap_ephemeral() {}
	/*
	 * Assume that pools, in general, are not thread safe, and therefore
	 * alloc/free calls need mutex protection
	 */
	hstore_impl::shared_mutex _alloc_mutex;
public:
	using byte_span = common::byte_span;
	explicit heap_ephemeral(unsigned debug_level)
		: common::log_source(debug_level)
		, _alloc_mutex()
	{}

	/* functions which ought to be virtualized across the heap_ephemeral types */
#if 0
	virtual void add_managed_region(byte_span r_full, byte_span r_heap) = 0;
	virtual nupm::region_descriptor get_managed_regions() const = 0;
	virtual nupm::region_descriptor set_managed_regions(nupm::region_descriptor n) = 0;
#endif
	virtual std::size_t allocated() const = 0;
	virtual std::size_t capacity() const = 0;
};

#endif
