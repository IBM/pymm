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

#include "heap_cc_ephemeral.h"

#include "as_pin.h"
#include "as_emplace.h"
#include "as_extend.h"
#include <ccpm/cca.h>
#include <common/errors.h> /* S_OK */
#include <cassert>
#include <cstdlib> /* getenv */
#include <memory> /* make_unique */
#include <numeric> /* accumulate */
#include <utility>

constexpr unsigned heap_cc_ephemeral::log_min_alignment;
constexpr unsigned heap_cc_ephemeral::hist_report_upper_bound;

namespace
{
	struct persister final
		: public ccpm::persister
	{
		void persist(common::byte_span s) override
		{
			::pmem_persist(::base(s), ::size(s));
		}
	};
	persister p_cc{};
}

/* initial cosntruction */

heap_cc_ephemeral::heap_cc_ephemeral(
	unsigned debug_level_
	, impl::allocation_state_emplace *ase_
	, impl::allocation_state_pin *aspd_
	, impl::allocation_state_pin *aspk_
	, impl::allocation_state_extend *asx_
	, std::unique_ptr<ccpm::IHeap_expandable> p
	, string_view id_
	, string_view backing_file_
	, const std::vector<byte_span> &rv_full_
	, byte_span pool0_heap_
)
	: heap_ephemeral(debug_level_)
	, _heap(std::move(p))
	, _managed_regions(id_, backing_file_, rv_full_)
	, _ase(ase_)
	, _aspd(aspd_)
	, _aspk(aspk_)
	, _asx(asx_)
	, _hist_alloc()
	, _hist_inject()
	, _hist_free()
{
  for ( const auto &r : rv_full_ )
  {
    CPLOG(2, "%s : %p.%zx", __func__, ::base(r), ::size(r));
  }
  CPLOG(2, "%s : pool0_heap: %p.%zx", __func__, ::base(pool0_heap_), ::size(pool0_heap_));
}

/* initial construction */
heap_cc_ephemeral::heap_cc_ephemeral(
	unsigned debug_level_
	, impl::allocation_state_emplace *ase_
	, impl::allocation_state_pin *aspd_
	, impl::allocation_state_pin *aspk_
	, impl::allocation_state_extend *asx_
	, string_view id_
	, string_view backing_file_
	, const std::vector<byte_span> &rv_full_
	, byte_span pool0_heap_
)
	: heap_cc_ephemeral(debug_level_, ase_, aspd_, aspk_, asx_, std::make_unique<ccpm::cca>(&p_cc, ccpm::region_span(&*ccpm::region_vector_t(pool0_heap_).begin(), 1)), id_, backing_file_, rv_full_, pool0_heap_)
{}

/* crash-consistent recovery */
heap_cc_ephemeral::heap_cc_ephemeral(
	unsigned debug_level_
	, impl::allocation_state_emplace *ase_
	, impl::allocation_state_pin *aspd_
	, impl::allocation_state_pin *aspk_
	, impl::allocation_state_extend *asx_
	, string_view id_
	, string_view backing_file_
	, const std::vector<byte_span> &rv_full_
	, byte_span pool0_heap_
	, ccpm::ownership_callback_t f
)
	: heap_cc_ephemeral(
		debug_level_
		, ase_, aspd_, aspk_, asx_
		, std::make_unique<ccpm::cca>(
			&p_cc
			, ccpm::region_span(&*ccpm::region_vector_t(pool0_heap_).begin(), 1)
			, f
		)
		, id_
		, backing_file_
		, rv_full_
		, pool0_heap_
	)
{}

void heap_cc_ephemeral::add_managed_region(
	byte_span r_full
	, byte_span r_heap
	, const unsigned // numa_node
)
{
	std::unique_lock<hstore_impl::shared_mutex> alloc_lk(_alloc_mutex);
	CPLOG(0, "%s before IHeap::add_regions size %zu", __func__, _heap->get_regions().size());
	for ( const auto &r : _heap->get_regions() )
	{
		CPLOG(0, "%s IHeap regions: %p.%zx", __func__, ::base(r), ::size(r));
	}
	ccpm::region_span::value_type rs[1] { r_heap };
	_heap->add_regions(rs);
	CPLOG(0, "%s : %p.%zx", __func__, ::base(r_heap), ::size(r_heap));
	_managed_regions.address_map_push_back(r_full);
	CPLOG(0, "%s after IHeap::add_regions size %zu", __func__, _heap->get_regions().size());
	for ( const auto &r : _heap->get_regions() )
	{
		CPLOG(0, "%s IHeap regions: %p.%zx", __func__, ::base(r), ::size(r));
	}
}

void heap_cc_ephemeral::allocate(
	persistent_t<void *> &p_
	, std::size_t sz_
	, std::size_t alignment_
)
{
	std::unique_lock<hstore_impl::shared_mutex> alloc_lk(_alloc_mutex);
	if ( S_OK != _heap->allocate(*reinterpret_cast<void **>(&p_), sz_, alignment_) )
	{
		throw std::bad_alloc{};
	}
	_hist_alloc.enter(sz_);
}

std::size_t heap_cc_ephemeral::free(persistent_t<void *> &p_, std::size_t sz_)
{
	std::unique_lock<hstore_impl::shared_mutex> alloc_lk(_alloc_mutex);
	/* Our free does not know the true size, because alignment is not known.
	 * But the pool free will know, as it can see how much has been allocated.
	 *
	 * The free, however, does not return a size. Pretend that it does.
	 */
	if ( is_crash_consistent() )
	{
	/* Note: order of testing is important. An extend arm+allocate) can occur while
	 * emplace is armed, but not vice-versa
	 */
		if ( _asx->is_armed() )
		{
			CPLOG(1, PREFIX "unexpected segment deallocation of %p of %zu", LOCATION, persistent_ref(p_), sz_);
			abort();
#if 0
			_asx->record_deallocation(&persistent_ref(p_), persister_nupm());
#endif
		}
		else if ( _ase->is_armed() )
		{
			_ase->record_deallocation(&persistent_ref(p_), persister_nupm());
		}
		else
		{
			CPLOG(1, PREFIX "leaky deallocation of %p of %zu", LOCATION, persistent_ref(p_), sz_);
		}
	}

	/* IHeap interface does not support abstract pointers. Cast to regular pointer */
	auto sz = (_heap->free(*reinterpret_cast<void **>(&p_), sz_), sz_);
	/* We would like to carry the persistent_t through to the crash-conssitent allocator,
	 * but for now just assume that the allocator has modifed p_, and call tick to indicate that.
	 */
	perishable::tick();
	_hist_free.enter(sz);
	return sz;
}

void heap_cc_ephemeral::free_tracked(const void *p_, std::size_t sz_, unsigned)
{
	std::unique_lock<hstore_impl::shared_mutex> alloc_lk(_alloc_mutex);
	_heap->free(const_cast<void *&>(p_), sz_);
	_hist_free.enter(sz_);
}

std::size_t heap_cc_ephemeral::capacity() const
{
	return std::accumulate(
		_managed_regions.address_map().begin()
		, _managed_regions.address_map().end()
		, std::size_t(0)
		, [] (std::size_t a, const auto &b) -> std::size_t
			{
				return a + ::size(b);
			}
	);
}

std::size_t heap_cc_ephemeral::allocated() const
{
	std::size_t r;
	auto rc = _heap->remaining(r);
	return capacity() - (rc == S_OK ? r : 0);
}
