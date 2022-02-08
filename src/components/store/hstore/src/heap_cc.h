/*
   Copyright [2017-2020] [IBM Corporation]
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


#ifndef MCAS_HSTORE_HEAP_CC_H
#define MCAS_HSTORE_HEAP_CC_H

#include "heap.h"

#include "as_emplace.h"
#include "cptr.h"
#include "histogram_log2.h"
#include "hop_hash_log.h"
#include "persistent.h"
#include "persister_nupm.h"
#include "pin_control.h"
#include "trace_flags.h"
#include "valgrind_memcheck.h"

#include <boost/icl/interval_set.hpp>
#include <ccpm/interfaces.h>
#include <common/byte_span.h>
#include <common/exceptions.h> /* General_exception */
#include <common/logging.h> /* log_source */
#include <common/string_view.h>
#include <nupm/region_descriptor.h>

#include <algorithm>
#include <array>
#include <cstddef> /* size_t, ptrdiff_t */
#include <memory>
#include <vector>

struct dax_manager;

namespace impl
{
	struct allocation_state_pin;
	struct allocation_state_extend;
}

struct heap_cc_ephemeral;

struct heap_cc
	: public heap
{
	using byte_span = common::byte_span;
	using string_view = common::string_view;
private:
	std::unique_ptr<heap_cc_ephemeral> _eph;
	pin_control<heap_cc> _pin_data;
	pin_control<heap_cc> _pin_key;

	void pin_data_arm(cptr &cptr) const;
	void pin_key_arm(cptr &cptr) const;
	char *pin_data_get_cptr() const;
	char *pin_key_get_cptr() const;
	void pin_data_disarm() const;
	void pin_key_disarm() const;

public:
	explicit heap_cc(
		unsigned debug_level
		, impl::allocation_state_emplace *ase
		, impl::allocation_state_pin *aspd
		, impl::allocation_state_pin *aspk
		, impl::allocation_state_extend *asx
		, byte_span pool0_full
		, byte_span pool0_heap
		, unsigned numa_node
		, string_view id_
		, string_view backing_file_
	);

	explicit heap_cc(
		unsigned debug_level
		, const std::unique_ptr<dax_manager> &dax_manager
		, string_view id
		, string_view backing_file
		, const byte_span *iov_addl_first_
		, const byte_span *iov_addl_last_
		, impl::allocation_state_emplace *ase
		, impl::allocation_state_pin *aspd
		, impl::allocation_state_pin *aspk
		, impl::allocation_state_extend *asx
	);

	heap_cc(const heap_cc &) = delete;
	heap_cc &operator=(const heap_cc &) = delete;

	~heap_cc();

	static constexpr std::uint64_t magic_value() { return 0x7c84297de2de94a3; }
	auto grow(
		const std::unique_ptr<dax_manager> & dax_manager_
		, std::uint64_t uuid_
		, std::size_t increment_
	) -> std::size_t;

	void quiesce();

	void alloc(persistent_t<void *> &p, std::size_t sz, std::size_t alignment);
	void *alloc_tracked(std::size_t sz, std::size_t alignment);
	void inject_allocation(const void * p, std::size_t sz);
	void free(persistent_t<void *> &p, std::size_t sz);
	void free_tracked(const void *p, std::size_t sz);

	void emplace_arm() const;
	void emplace_disarm() const;

	impl::allocation_state_pin *aspd() const;
	impl::allocation_state_pin *aspk() const;

	const pin_control<heap_cc> &pin_control_data() const { return _pin_data; }
	const pin_control<heap_cc> &pin_control_key() const { return _pin_key; }
	void extend_arm() const;
	void extend_disarm() const;

	unsigned percent_used() const;

	bool is_reconstituted(const void *) const { return false; }

	nupm::region_descriptor regions() const;

	bool is_crash_consistent() const { return true; }
	bool can_reconstitute() const { return false; }
};

#endif
