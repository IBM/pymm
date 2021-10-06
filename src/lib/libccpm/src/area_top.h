/*
   Copyright [2019, 2020] [IBM Corporation]
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

#ifndef __CCPM_AREATOP_ALLOCATOR_H__
#define __CCPM_AREATOP_ALLOCATOR_H__

#include "atomic_word.h"
#include "list_item.h"
#include <ccpm/interfaces.h>
#include <common/byte_span.h>
#include <gsl/pointers>
#include <array>
#include <cstddef>
#include <ios> // ios_base::fmtflags, ostream
#include <vector>

namespace ccpm
{
	struct level_hints
	{
	private:
		/* Each level has a list for every possible number of contiguous free elements.
		 * of contiguous free elements.
		 * Since "contiguous" free elements do not span a word, that is one list
		 * for every possible contiguous size in a word.
		 * The list at element n locates an area_ctl with has maximal runs of exactly
		 * n+1 free elements,
		 */

		using free_ctls_t = std::array<list_item, alloc_states_per_word>;
		free_ctls_t _free_ctls;
		free_ctls_t::size_type find_free_ctl_ix(unsigned min_run_length) const;
		free_ctls_t::size_type tier_ix_from_run_length(unsigned run_length) const;
	public:
		unsigned _ct_alloc_probe_success;
		unsigned _ct_alloc_probe_failure;
		unsigned _ct_subdivision;

	public:
		static constexpr auto size() { return alloc_states_per_word; }

		/* return smallest tier index which has an area with
		 *   run length >= min_run_length
		 * If no such tier, return sub_states_per_word.
		 */
		auto find_free_ctl(unsigned min_run_length_) const -> const list_item *
		{
			return & _free_ctls[find_free_ctl_ix(min_run_length_)];
		}

		auto find_free_ctl(unsigned min_run_length_) -> list_item *
		{
			return & _free_ctls[find_free_ctl_ix(min_run_length_)];
		}

		const auto *tier_from_run_length(unsigned run_length_) const
		{
			return & _free_ctls[tier_ix_from_run_length(run_length_)];
		}

		auto *tier_from_run_length(unsigned run_length_)
		{
			return & _free_ctls[tier_ix_from_run_length(run_length_)];
		}

		const auto *tier_end() const
		{
			return &_free_ctls[alloc_states_per_word];
		}

		/* Find the longest run length less than mac_run_length.
		 * Used for tracing, when find_free_ctl_ix has failed to
		 * find a long enough run
		 */
		free_ctls_t::size_type find_mac_ctl_ix(unsigned mac_run_length) const;

		level_hints()
			: _free_ctls()
			, _ct_alloc_probe_success(0)
			, _ct_alloc_probe_failure(0)
			, _ct_subdivision(0)
		{}

		using level_ix_t = std::uint8_t;
		void print(
			std::ostream &o_
			, level_ix_t level_
			, std::ios_base::fmtflags // size_format_
		) const;
	};

	struct area_ctl;

	/*
	 * Location of non-persisted items for a single "region" managed by the crash-consistent allocator.
	 * Persisted items are keps in an area_ctl, which is in persistent memory.
	 */
	struct area_top
	{
	private:
		using level_ix_t = std::uint8_t;
		using byte_span = common::byte_span;
		using persist_type = gsl::not_null<ccpm::persister *>;
		area_ctl *_ctl;
		std::size_t _bytes_free;
		bool _all_restored;
		unsigned _trace_level;
		std::ostream *_o;
		using level_hints_vec = std::vector<level_hints>;
		level_hints_vec _level;
		unsigned _ct_allocation;
		byte_span _region; /* for get_region only */

		area_top(
			// persist_type persist
			area_ctl *ctl
			, unsigned trace_level
			, byte_span iov
			, std::ostream &o
		);
		area_top(const area_top &) = delete;
		area_top &operator=(const area_top &) = delete;

		void allocate_strategy_1(
			persist_type persist_
			, void * & ptr_
			, std::size_t bytes
			, std::size_t alignment
			, level_hints_vec::iterator level_
			, unsigned run_length
		);

		bool allocate_recovery_1();
		bool allocate_recovery_2(persist_type persist, level_hints_vec::iterator level);
		bool trace_coarse() const { return 0 < _trace_level; }
		bool trace_fine() const { return 1 < _trace_level; }

	public:
		/* Initial area_ctl */
		explicit area_top(
			persist_type persist
			, byte_span iov
			, unsigned trace_level
			, std::ostream &o
		);
		/* Restored area_ctl */
		explicit area_top(
			persist_type persist
			, byte_span iov
			, const ownership_callback_t &resolver
			, unsigned trace_level
			, std::ostream &o
		);
		~area_top();

		bool includes(const void *addr) const;

		/* Free byte count. Required by users */
		std::size_t bytes_free() const;

		byte_span get_region() const { return _region; }

		void allocate(
			persist_type persist
			, void * & ptr, std::size_t bytes
			, std::size_t alignment
		);

		void deallocate(
			persist_type persist
			, void * & ptr, std::size_t bytes
		);

		void print(
			std::ostream &o
			, std::ios_base::fmtflags size_format
		) const;

		void print_ctls(
			std::ostream *o_
			, std::ios_base::fmtflags format_
		) const;

    level_ix_t height() const { return level_ix_t(_level.size()); }

    void set_root(const byte_span & iov, persist_type persist);
    byte_span get_root() const;

		/*
		 * called by area_ctl to add area_ctl a, at level level_ix, with a longest
		 * free run (consecutive free elements) of free_run, to _level, which is the
		 * non-persistent catalog of area_ctl items.
		 */
		void remove_from_chain(area_ctl *a, level_ix_t level_ix, unsigned longest_run);
		void restore_to_chain(area_ctl *a, level_ix_t level_ix, unsigned run_length);

		bool contains(const void *p) const;

		bool is_in_chain(
			const area_ctl *a
			, level_ix_t level_ix
			, unsigned run_length
		) const;
	};
}

#endif
