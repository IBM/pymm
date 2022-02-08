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


#ifndef MCAS_HSTORE_HEAP_MR_H
#define MCAS_HSTORE_HEAP_MR_H

#include "hstore_config.h"

#include "histogram_log2.h"
#include "hop_hash_log.h"
#include "persistent.h"
#include "persister_nupm.h"
#include "pin_control.h"
#include "trace_flags.h"
#include "tracked_header.h"

#include <boost/icl/interval_set.hpp>
#include <common/byte_span.h>
#include <common/exceptions.h> /* General_exception */
#include <common/string_view.h>
#include <nupm/region_descriptor.h>

#include <algorithm>
#include <array>
#include <cstddef> /* size_t, ptrdiff_t */
#include <memory> /* unique_ptr */
#include <vector>

struct dax_manager;

namespace impl
{
	struct allocation_state_combined;
	struct allocation_state_emplace;
	struct allocation_state_extend;
	struct allocation_state_pin;
}

struct cptr;

struct heap_mm_ephemeral;

struct heap_mr
{
private:
	using byte_span = common::byte_span;
	using string_view = common::string_view;
	byte_span _pool0_full; /* entire extent of pool 0 */
	byte_span _pool0_heap; /* portion of pool 0 which can be used for the heap */
	unsigned _numa_node;
	std::size_t _more_region_uuids_size;
	std::array<std::uint64_t, 1024U> _more_region_uuids;
	tracked_header _tracked_anchor;
	std::unique_ptr<heap_mm_ephemeral> _eph;
	pin_control<heap_mr> _pin_data;
	pin_control<heap_mr> _pin_key;

	void pin_data_arm(cptr &cptr) const;
	void pin_key_arm(cptr &cptr) const;
	char *pin_data_get_cptr() const;
	char *pin_key_get_cptr() const;
	void pin_data_disarm() const;
	void pin_key_disarm() const;

public:
	explicit heap_mr(
		unsigned debug_level
		, string_view plugin_path
		, impl::allocation_state_emplace *ase
		, impl::allocation_state_pin *aspd
		, impl::allocation_state_pin *aspk
		, impl::allocation_state_extend *asx
		, byte_span pool0_full
		, byte_span pool0_heap
		, unsigned numa_node
		, string_view id
		, string_view backing_file
	);
	explicit heap_mr(
		unsigned debug_level
		, string_view plugin_path
		, const std::unique_ptr<dax_manager> &dax_manager
		, string_view id
		, string_view backing_file
		, const byte_span *iov_addl_first
		, const byte_span *iov_addl_last
		, impl::allocation_state_emplace *ase
		, impl::allocation_state_pin *aspd
		, impl::allocation_state_pin *aspk
		, impl::allocation_state_extend *asx
	);

	heap_mr(const heap_mr &) = delete;
	heap_mr &operator=(const heap_mr &) = delete;

	~heap_mr();

	static constexpr std::uint64_t magic_value() { return 0xc74892d72eed493a; }

	auto grow(
		const std::unique_ptr<dax_manager> & dax_manager
		, std::uint64_t uuid
		, std::size_t increment
	) -> std::size_t;

	void quiesce();

	void alloc(persistent_t<void *> &p, std::size_t sz, std::size_t alignment);
	void *alloc_tracked(std::size_t sz, std::size_t alignment);

	void inject_allocation(const void * p, std::size_t sz);

	void free(persistent_t<void *> &p, std::size_t sz);
	void free_tracked(const void *p, std::size_t sz);

	void extend_arm() const {};
	void extend_disarm() const {};
	void emplace_arm() const {};
	void emplace_disarm() const {};

	impl::allocation_state_pin *aspd() const;
	impl::allocation_state_pin *aspk() const;

	const pin_control<heap_mr> &pin_control_data() const { return _pin_data; }
	const pin_control<heap_mr> &pin_control_key() const { return _pin_key; }

	unsigned percent_used() const;

	bool is_reconstituted(const void * p) const;

	nupm::region_descriptor regions() const;

	bool is_crash_consistent() const;
	bool can_reconstitute() const;
};

#endif
