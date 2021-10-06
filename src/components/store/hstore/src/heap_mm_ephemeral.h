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

#ifndef MCAS_HSTORE_HEAP_MM_EPHEMERAL_H
#define MCAS_HSTORE_HEAP_MM_EPHEMERAL_H

#include "heap_ephemeral.h"

#include "hstore_config.h"
#include "histogram_log2.h"
#include "hop_hash_log.h"
#include <mm_plugin_itf.h>
#include "persistent.h"

#include <ccpm/interfaces.h> /* ownership_callback, (IHeap_expandable, region_vector_t) */
#include <common/byte_span.h>
#include <common/string_view.h>
#include <nupm/region_descriptor.h>
#include <gsl/span>

#include <algorithm> /* min, swap */
#include <cstddef> /* size_t */
#include <vector>

struct heap_mm_ephemeral
	: private heap_ephemeral
{
private:
	using byte_span = common::byte_span;

	/* Although the mm heap tracks managed regions, its definition of a managed
	 * region probably differs from what hstore needs:
	 *
	 * The hstore managed region must include the hstore metadata area, as that
	 * must be include in the "maanged regions" returned through the kvstore
	 * interface.
	 *
	 * The mm heap managed region cannot include the hstore metadata area, as
	 * that would allow the allocator to use the area for memory allocation.
	 *
	 * Therefore, heap_mr_ephemeral keeps its own copy of the managed regions.
	 */
	nupm::region_descriptor _managed_regions;
protected:
	using heap_ephemeral::_alloc_mutex;

protected:
	using hist_type = util::histogram_log2<std::size_t>;
	hist_type _hist_alloc;
	hist_type _hist_inject;
	hist_type _hist_free;
private:
	/* Rca_LB seems not to allocate at or above about 2GiB. Limit reporting to 16 GiB. */
	static constexpr unsigned hist_report_upper_bound = 24U;
	static constexpr unsigned log_min_alignment = 3U; /* log (sizeof(void *)) */
	static_assert(sizeof(void *) == 1U << log_min_alignment, "log_min_alignment does not match sizeof(void *)");

protected:
	virtual void add_managed_region_to_heap(byte_span r_heap) = 0;
public:
	template <bool B>
		void write_hist(const byte_span pool_) const
		{
			static bool suppress = false;
			if ( ! suppress )
			{
				hop_hash_log<B>::write(LOG_LOCATION, "pool ", ::base(pool_));
				std::size_t lower_bound = 0;
				auto limit = std::min(std::size_t(hist_report_upper_bound), _hist_alloc.data().size());
				for ( unsigned i = log_min_alignment; i != limit; ++i )
				{
					const std::size_t upper_bound = 1ULL << i;
					hop_hash_log<B>::write(LOG_LOCATION
						, "[", lower_bound, "..", upper_bound, "): "
						, _hist_alloc.data()[i], " ", _hist_inject.data()[i], " ", _hist_free.data()[i]
						, " "
					);
					lower_bound = upper_bound;
				}
				suppress = true;
			}
		}

	using common::log_source::debug_level;
	explicit heap_mm_ephemeral(
		unsigned debug_level
		, nupm::region_descriptor managed_regions
	);
	virtual ~heap_mm_ephemeral() {}

	virtual std::size_t allocated() const = 0;
	virtual std::size_t capacity() const = 0;

	virtual void allocate(persistent_t<void *> &p, std::size_t sz, std::size_t alignment) = 0;
	virtual std::size_t free(persistent_t<void *> &p_, std::size_t sz_) = 0;
	virtual void free_tracked(const void *p, std::size_t sz) = 0;

	nupm::region_descriptor get_managed_regions() const { return _managed_regions; }
	nupm::region_descriptor set_managed_regions(nupm::region_descriptor n)
	{
		using std::swap;
		swap(n, _managed_regions);
		return n;
	}
	void add_managed_region(byte_span r_full, byte_span r_heap);
	void reconstitute_managed_region(
		const byte_span r_full
		, const byte_span r_heap
		, ccpm::ownership_callback_t f
	);
	virtual void reconstitute_managed_region_to_heap(byte_span r_heap, ccpm::ownership_callback_t f) = 0;
	virtual bool is_crash_consistent() const = 0;
	virtual bool can_reconstitute() const = 0;
};

#endif
