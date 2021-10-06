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

#ifndef MCAS_HSTORE_HEAP_MC_EPHEMERAL_H
#define MCAS_HSTORE_HEAP_MC_EPHEMERAL_H

#include "heap_mm_ephemeral.h"

#include "hstore_config.h"
#include "histogram_log2.h"
#include "hop_hash_log.h"
#include <mm_plugin_itf.h>
#include "persistent.h"

#include <ccpm/cca.h> /* ctor_args */
#include <ccpm/interfaces.h> /* ownership_callback, (IHeap_expandable, region_vector_t) */
#include <common/byte_span.h>
#include <common/string_view.h>
#include <nupm/region_descriptor.h>
#include <gsl/span>

#include <algorithm> /* min, swap */
#include <cstddef> /* size_t */
#include <functional> /* function */
#include <memory> /* unique_ptr */
#include <vector>

namespace impl
{
	struct allocation_state_pin;
	struct allocation_state_emplace;
	struct allocation_state_extend;
}

struct heap_mc_ephemeral
  : public heap_mm_ephemeral
{
private:
	using byte_span = common::byte_span;
	using string_view = common::string_view;
	std::unique_ptr<
		ccpm::IHeap_expandable
	> _heap;
	impl::allocation_state_emplace *_ase;
	impl::allocation_state_pin *_aspd;
	impl::allocation_state_pin *_aspk;
	impl::allocation_state_extend *_asx;

	static constexpr unsigned log_min_alignment = 3U; /* log (sizeof(void *)) */
	static_assert(sizeof(void *) == 1U << log_min_alignment, "log_min_alignment does not match sizeof(void *)");
	/* Rca_LB seems not to allocate at or above about 2GiB. Limit reporting to 16 GiB. */
	static constexpr unsigned hist_report_upper_bound = 34U;
	explicit heap_mc_ephemeral(
		unsigned debug_level_
		, bool restore_not_clear
		, impl::allocation_state_emplace *ase
		, impl::allocation_state_pin *aspd
		, impl::allocation_state_pin *aspk
		, impl::allocation_state_extend *asx
		, std::unique_ptr<ccpm::IHeap_expandable> p
		, string_view id
		, string_view backing_file
		, const std::vector<byte_span> rv_full
		, const byte_span pool0_heap
	);

public:
	friend struct heap_mc;
	friend struct heap_mm;

	using common::log_source::debug_level;

	/* heap_mm version */
	explicit heap_mc_ephemeral(
		unsigned debug_level
		, bool restore_not_clear
		, MM_plugin_wrapper &&pw
		, impl::allocation_state_emplace *ase
		, impl::allocation_state_pin *aspd
		, impl::allocation_state_pin *aspk
		, impl::allocation_state_extend *asx
		, string_view id
		, string_view backing_file
		, const std::vector<byte_span> rv_full
		, byte_span pool0_heap_
	);

	std::size_t allocated() const override; // { return _allocated; }
	std::size_t capacity() const override; // { return _capacity; }
	void allocate(persistent_t<void *> &p, std::size_t sz, std::size_t alignment) override;
	std::size_t free(persistent_t<void *> &p_, std::size_t sz_) override;
	void free_tracked(const void *p, std::size_t sz) override;
	heap_mc_ephemeral(const heap_mc_ephemeral &) = delete;
	heap_mc_ephemeral& operator=(const heap_mc_ephemeral &) = delete;
	void add_managed_region_to_heap(byte_span r_heap) override;
	void reconstitute_managed_region_to_heap(byte_span r_heap, ccpm::ownership_callback_t f) override;
	bool is_crash_consistent() const override;
	bool can_reconstitute() const override;
};

#endif
