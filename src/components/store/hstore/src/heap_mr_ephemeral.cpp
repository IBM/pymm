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

#include "heap_mr_ephemeral.h"

#include "hstore_config.h"
#include "heap_mr_shim.h"
#include "mm_plugin_itf.h"
#include <common/errors.h> /* S_OK, E_INVAL */
#include <memory> /* make_unique */
#include <shared_mutex> /* shared_lock, unique_lock */

constexpr unsigned heap_mr_ephemeral::log_min_alignment;
constexpr unsigned heap_mr_ephemeral::hist_report_upper_bound;

/* heap_mr version */
heap_mr_ephemeral::heap_mr_ephemeral(
	unsigned debug_level_
	, const string_view plugin_path_
	, const string_view id_
	, const string_view backing_file_
)
	: heap_mm_ephemeral(debug_level_, nupm::region_descriptor{id_, backing_file_, {}})
	, _heap(std::make_unique<heap_mr_shim>(plugin_path_))
	, _allocated(0)
	, _capacity(0)
	, _reconstituted()
{}

/* heap_mm version */
heap_mr_ephemeral::heap_mr_ephemeral(
	unsigned debug_level_
	, MM_plugin_wrapper &&pw_
	, const string_view id_
	, const string_view backing_file_
)
	: heap_mm_ephemeral(debug_level_, nupm::region_descriptor{id_, backing_file_, {}})
	, _heap(std::make_unique<heap_mr_shim>(std::move(pw_)))
	, _allocated(0)
	, _capacity(0)
	, _reconstituted()
{}

heap_mr_ephemeral::~heap_mr_ephemeral()
{}


void heap_mr_ephemeral::add_managed_region_to_heap(byte_span r_heap)
{
	_heap->add_managed_region(r_heap);
	_capacity += ::size(r_heap);
}

void heap_mr_ephemeral::inject_allocation(
	void *p_
	, std::size_t sz_
)
{
	std::unique_lock<hstore_impl::shared_mutex> alloc_lk(_alloc_mutex);
	_heap->inject_allocation(p_, sz_);
	{
		auto pc = static_cast<alloc_set_t::element_type>(p_);
		_reconstituted.add(alloc_set_t::segment_type(pc, pc + sz_));
	}
	_allocated += sz_;
	_hist_alloc.enter(sz_);
}

void heap_mr_ephemeral::allocate(
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
	_allocated += sz_;
	_hist_alloc.enter(sz_);
}

std::size_t heap_mr_ephemeral::free(persistent_t<void *> &p_, std::size_t sz_)
{
	std::unique_lock<hstore_impl::shared_mutex> alloc_lk(_alloc_mutex);
	_heap->free(*reinterpret_cast<void **>(&p_), sz_);
	_allocated -= sz_;
	_hist_free.enter(sz_);
	return sz_;
}

void heap_mr_ephemeral::free_tracked(const void *p_, std::size_t sz_)
{
	std::unique_lock<hstore_impl::shared_mutex> alloc_lk(_alloc_mutex);
	void *p = const_cast<void *>(p_);
	_heap->free(p, sz_);
	_allocated -= sz_;
	_hist_free.enter(sz_);
}

bool heap_mr_ephemeral::is_reconstituted(const void * p_)
{
	std::shared_lock<hstore_impl::shared_mutex> alloc_lk(_alloc_mutex);
	return contains(_reconstituted, static_cast<alloc_set_t::element_type>(p_));
}

bool heap_mr_ephemeral::is_crash_consistent() const { return false; }
bool heap_mr_ephemeral::can_reconstitute() const { return true; }
void heap_mr_ephemeral::reconstitute_managed_region_to_heap(byte_span, ccpm::ownership_callback_t)
{
	throw std::domain_error("mr cannot reconstitute");
}
