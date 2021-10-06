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

#include "heap_mm_ephemeral.h"

#include "as_pin.h"
#include "as_emplace.h"
#include "as_extend.h"
#include "heap_mc_shim.h"
#include "mm_plugin_itf.h"
#include <common/errors.h> /* S_OK */
#include <cassert>
#include <cstdlib> /* getenv */
#include <memory> /* make_unique */
#include <numeric> /* accumulate */
#include <utility>

heap_mm_ephemeral::heap_mm_ephemeral(
	unsigned debug_level_
	, nupm::region_descriptor managed_regions_
)
	: heap_ephemeral(debug_level_)
	, _managed_regions(std::move(managed_regions_))
	, _hist_alloc()
	, _hist_inject()
	, _hist_free()
{
}

void heap_mm_ephemeral::add_managed_region(
	const byte_span r_full
	, const byte_span r_heap
)
{
	add_managed_region_to_heap(r_heap);
	CPLOG(0, "%s : %p.%zx", __func__, ::base(r_heap), ::size(r_heap));
	_managed_regions.address_map_push_back(r_full);
}

void heap_mm_ephemeral::reconstitute_managed_region(
	const byte_span r_full
	, const byte_span r_heap
	, ccpm::ownership_callback_t f
)
{
	reconstitute_managed_region_to_heap(r_heap, f);
	CPLOG(0, "%s : %p.%zx", __func__, ::base(r_heap), ::size(r_heap));
	_managed_regions.address_map_push_back(r_full);
}
