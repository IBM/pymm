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


#ifndef MCAS_HSTORE_HEAP_RC_H
#define MCAS_HSTORE_HEAP_RC_H

#include "hstore_config.h"

#include "as_extend.h"
#include "as_pin.h"
#include "histogram_log2.h"
#include "hop_hash_log.h"
#include "persistent.h"
#include "persister_nupm.h"
#include "pin_control.h"
#include "rc_alloc_wrapper_lb.h"
#include "trace_flags.h"
#include "tracked_header.h"

#include <boost/icl/interval_set.hpp>
#include <common/byte_span.h>
#include <common/exceptions.h> /* General_exception */
#include <common/string_view.h>
#include <nupm/region_descriptor.h>

#include <sys/uio.h> /* iovec */

#include <algorithm>
#include <array>
#include <cstddef> /* size_t, ptrdiff_t */
#include <memory> /* unique_ptr */
#include <vector>

struct dax_manager;

namespace impl
{
	struct allocation_state_combined;
}

struct heap_rc_ephemeral;

struct heap_rc
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
	std::unique_ptr<heap_rc_ephemeral> _eph;
	pin_control<heap_rc> _pin_data;
	pin_control<heap_rc> _pin_key;

	void pin_data_arm(cptr &cptr) const;
	void pin_key_arm(cptr &cptr) const;
	char *pin_data_get_cptr() const;
	char *pin_key_get_cptr() const;
	void pin_data_disarm() const;
	void pin_key_disarm() const;

public:
	explicit heap_rc(
		unsigned debug_level
		, impl::allocation_state_emplace *ase
		, impl::allocation_state_pin *aspd
		, impl::allocation_state_pin *aspk
		, impl::allocation_state_extend *asx
		, byte_span pool0_full
		, byte_span pool0_heap
		, unsigned numa_node
		, string_view id_
		, string_view backing_file
	);
	explicit heap_rc(
		unsigned debug_level
		, const std::unique_ptr<dax_manager> &dax_manager
		, string_view id_
		, string_view backing_file
		, const byte_span *iov_addl_first_
		, const byte_span *iov_addl_last_
		, impl::allocation_state_emplace *ase
		, impl::allocation_state_pin *aspd
		, impl::allocation_state_pin *aspk
		, impl::allocation_state_extend *asx
	);
	/* allocation_state_combined offered, but not used */
	explicit heap_rc(
		const unsigned debug_level
		, const std::unique_ptr<dax_manager> &dax_manager
		, const string_view id
		, const string_view backing_file
		, impl::allocation_state_combined const *
		, const byte_span *iov_addl_first_
		, const byte_span *iov_addl_last_
		, impl::allocation_state_emplace *ase
		, impl::allocation_state_pin *aspd
		, impl::allocation_state_pin *aspk
		, impl::allocation_state_extend *asx
	)
		: heap_rc(debug_level, dax_manager, id, backing_file, iov_addl_first_, iov_addl_last_, ase, aspd, aspk, asx)
	{
	}

	heap_rc(const heap_rc &) = delete;
	heap_rc &operator=(const heap_rc &) = delete;

	~heap_rc();

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
	std::size_t free(persistent_t<void *> &p, std::size_t sz);
	void free_tracked(const void *p, std::size_t sz);

	void extend_arm() const {};
	void extend_disarm() const {};
	void emplace_arm() const {};
	void emplace_disarm() const {};

	impl::allocation_state_pin *aspd() const;
	impl::allocation_state_pin *aspk() const;

	const pin_control<heap_rc> &pin_control_data() const { return _pin_data; }
	const pin_control<heap_rc> &pin_control_key() const { return _pin_key; }

	unsigned percent_used() const;

	bool is_reconstituted(const void * p) const;

    nupm::region_descriptor regions() const;
	bool is_crash_consistent() const { return false; }
	bool can_reconstitute() const { return true; }
};

#endif
