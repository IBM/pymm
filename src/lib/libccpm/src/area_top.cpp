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

#include "area_top.h"

#include "area_ctl.h"
#include <common/utils.h>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <ostream>

/* scan the chains of free pointers, starting with the least acceptable length
 * (min_run_length_) and ending at the greatest possible length (alloc_states_per_word)
 * to find the first non-empty list. This would be quicker If the non-empty tiers
 * were chained, or a bitmask of non-empty tiers were kept.
 */
auto ccpm::level_hints::find_free_ctl_ix(
	const unsigned min_run_length_
) const -> free_ctls_t::size_type
{
	auto ix = tier_ix_from_run_length(min_run_length_);
	for ( ; ix != alloc_states_per_word && _free_ctls[ix].empty() ; ++ix )
	{
	}
	return ix;
}

auto ccpm::level_hints::find_mac_ctl_ix(
	const unsigned mac_run_length_
) const -> free_ctls_t::size_type
{
	auto ix = tier_ix_from_run_length(mac_run_length_);
	for ( ; ix-1 != 0 && _free_ctls[ix-1].empty(); --ix )
	{
	}
	return ix;
}

auto ccpm::level_hints::tier_ix_from_run_length(
	unsigned run_length
) const -> free_ctls_t::size_type
{
	assert(run_length != 0 && run_length <= alloc_states_per_word);
	return run_length - 1;
}

auto ccpm::area_top::is_in_chain(
	const area_ctl *a
	, level_ix_t level_ix
	, unsigned run_length
) const -> bool
{
	assert(level_ix < _level.size());
	const auto &level = _level[level_ix];
	return run_length != 0 && level.tier_from_run_length(run_length)->contains(a);
}

bool ccpm::area_top::includes(const void *addr) const
{
	return _ctl && _ctl->includes(addr);
}

ccpm::area_top::area_top(
	area_ctl *top_ctl_
	, const unsigned trace_level_
	, const byte_span region_
	, std::ostream &o_
)
	: _ctl(top_ctl_)
	/* bytes_free does a full search of the existing tree.
	 * Consider doing a full chain restoration at the same time.
	 */
	, _bytes_free(_ctl->bytes_free())
	, _all_restored(false)
	, _trace_level(trace_level_)
	, _o(&o_)
	, _level(_ctl->level()+1)
	, _ct_allocation(0)
	, _region(region_)
{
	/* TODO: add _ctl to appropriate free_ctl chain */
}

ccpm::area_top::~area_top()
{
	if ( bool(std::getenv("CCA_SUBDIVISION_REPORT")) )
	{
		try {
			std::cout << "Allocations " << _ct_allocation << "\n";
			std::size_t sz = 8;
			for ( const auto &lv : _level )
			{
				try
				{
					std::cout << "probes at size " << sz
						<< " probe success " << lv._ct_alloc_probe_success
						<< " probe failures " << lv._ct_alloc_probe_failure
						<< " subdivisions " << lv._ct_subdivision
						<< "\n";
				}
				catch ( const std::exception & )
				{}
				sz *= alloc_states_per_word;
			}
		}
		catch ( const std::exception & )
		{}
	}
}

/* Initial area_ctl */
ccpm::area_top::area_top(persist_type persist_, const byte_span iov_, const unsigned trace_level_, std::ostream &o_)
	: area_top(
	//	persist_,
		area_ctl::commission(persist_, iov_), trace_level_, iov_, o_
	)
{}

/* Restored area_ctl */
ccpm::area_top::area_top(
	persist_type persist_
	, const byte_span iov_
	, const ownership_callback_t &resolver_
	, const unsigned trace_level_
	, std::ostream &o_
)
	: area_top(&area_ctl::root(::base(iov_))->restore(persist_, resolver_), trace_level_, iov_, o_)
{
}

auto ccpm::area_top::bytes_free() const -> std::size_t
{
	return _ctl ? _ctl->bytes_free() : 0;
}

void ccpm::area_top::remove_from_chain(
	area_ctl *a_
	, level_ix_t level_ix_
	, unsigned longest_run_
)
{
	if ( longest_run_ != 0 )
	{
		assert(level_ix_ < _level.size());
		auto &level = _level[level_ix_];
		auto free_ctl = level.tier_from_run_length(longest_run_);
		if ( free_ctl->contains(a_) )
		{
			a_->remove();
		}
		else
		{
			a_->force_reset();
		}
	}
}

void ccpm::area_top::restore_to_chain(
	area_ctl *a_
	, level_ix_t level_ix_
	, unsigned longest_run_
)
{
	if ( longest_run_ != 0 )
	{
		assert(level_ix_ < _level.size());
		auto &level = _level[level_ix_];
		auto free_ctl = level.tier_from_run_length(longest_run_);
		free_ctl->insert_after(a_);
	}
}

/*
 * Levels: areas are placed in levels according to the log of their size.
 * Levels form a hierarchy, with level 0 containing the smallest elements,
 * area_ctl::min_alloc_size bytes.
 *
 * Tier: within each level, areas are sorted according to the longest run
 * of free elements in the area. (A free run must be contained entirely
 * within a single atomic_word.) Tiers run from 0 to 32. Tier 0 will always
 * be empty, as there is no point in tracking fully allocated areas.
 * Tier 32 could also be "always empty", if we chose to undo subdivision
 * of entirely free areas. We do not do that yet.
 *
 * Allocate strategy:
 * Determine level L at which the allocation should occur
 *    (
 *      _sub_size(L) <= bytes_ <= _sub_size(L)*(32-1)
 *      ||
 *      _sub_size == area_ctl::min_alloc_size
 *    )
 *
 * 1. allocate from existing chains in area_top which are at L and have free
 *    runs at least long enough to hold bytes. Move the subdivided area from
 *    its current area_top chain to its new area_top chain or, of the area is
 *    entirely allocated, to no area_top chain.
 * (if none found)
 * 2. Starting after the rightmost location encountered in the previous search
 *    of this kind at L, conduct an exhaustive left-to-right search for subdivided
 *    areas, adding each area to its proper area_top chain if not already in the
 *    chain.
 *    If a suitable subdivided area is found, stop the search and allocate at that
 *    area. Add the subdivided area to its appropriate area_top chain.
 ( (if none found ...)
 * 3. Create a new subdivided region from L+1 (if any free at L+1), allocate from
 *    that region. If none free at L+1, first create subdivided region at L+2,
 *    then L+1, etc.) Add each new subdivided region to its appropriate area_top
 *    chain.
 * 4.
 *
 *
 * 1. Allocate from a chained area the appropriate level, and from any tier
 *    containing a long enough run of free elements. (The area from which
 *    elements are allocated may now have a shorter "longest run". Remove that
 *    area from its current tier. If the area still has a non-zero run, add it to
 *    a new tier.
 *
 * If the allocation fails, there are two possibilities: (a) subdivide a chained
 * area to produce an area at the right level, or (b) search for areas to add to
 * the chain.
 *
 * Attempting (a) first has this advantage: The new areas subdivided, if any, will
 * be new subdivisions and not an pre-existing subdivisions (which may or may not
 * be already present in a chain).
 * Attempting (a) first has this disadvantage: additional fragmentation.
 *
 * Attempting (b) first has this advantage: smaller, already-fragmented areas might
 * be discovered and used.
 * Attempting (b) first has this disadvantage: Any subdivision found would have
 * links of indeterminate state: we would not know whether the links were part
 * of a current chain, or left over from a previous generation of (non-persistent)
 * chains.
 *
 * 2. (restricted version of (b)): Move an unchained area to the chain and retry
 * the allocation. If we find such an area we will know that it is not already in
 * the chain; if it were it would have been found and used in Step 1.
 *
 * 3. Subdivide a chained area 2. (Chooses speed at the cost of fragmentation.)
 */

/* Part 1: allocate from an existing chain */
void ccpm::area_top::allocate_strategy_1(
	persist_type persist_
	, void * & ptr_
	, const size_t bytes_
	, const size_t alignment_
	, const level_hints_vec::iterator level_
	, const unsigned run_length_
)
{
	/* One each level there are separate ctl chains for the longest free run
	 * at each level (from 1 to substates_per_word). The index of each chain
	 * is one less then the longest free run.
	 */
	assert(run_length_ < level_->size());
	auto tier_ptr = level_->find_free_ctl(run_length_);
	if ( tier_ptr != level_->tier_end() )
	{
		++level_->_ct_alloc_probe_success;
		auto &free_ctl = *tier_ptr;

		/* A viable allocation exists at free_ctl. Use it. */
		auto viable = static_cast<area_ctl *>(free_ctl.next());
		viable->allocate(persist_, _ctl->get_doubt(), ptr_, bytes_, alignment_, run_length_);
		/* The ctl may have a new, shorter longest run.
		 * If so, move it to a new chain within the level object */
		auto longest_run = viable->el_max_free_run();
		viable->remove();
		if ( longest_run != 0 )
		{
			level_->tier_from_run_length(longest_run)->insert_after(viable);
		}
	}
	else
	{
		++level_->_ct_alloc_probe_failure;
		if ( trace_coarse() )
		{
			/* No viable allocation was found. If tracing, report the max
			 * which was available.
			 */
			auto level_ix = level_ix_t(level_ - _level.begin());
			auto ix = level_->find_mac_ctl_ix(run_length_);

			if ( ix == 0 )
			{
				PLOG("cache level %d, needed run length %u, no runs available", level_ix, run_length_);
			}
			else
			{
				PLOG("cache level %d, needed run length %u, max available was %zu", level_ix, run_length_, ix-1);
			}
			print_ctls(&std::cerr, std::ios_base::hex);
		}
	}
}

/* Recovery 1: rechain all existing (not chained) area_ctls. Such a
 * area_ctls could only exist after a crash or restart, as eligible area_ctls
 * are normally chained.
 */
bool ccpm::area_top::allocate_recovery_1()
{
	if ( _o && trace_fine() )
	{
		*_o << __func__ << " for " << common::p_fmt(this) << ", ctl at " << common::p_fmt(_ctl) << "\n";
	}
	auto ct =
		_all_restored
		? 0U
		: _ctl->restore_all(
				this
				, trace_fine() ? _o : nullptr
				, level_ix_t(_level.size()-1)
			)
		;
	_all_restored = true;
	return ct != 0;
}

/* Recovery 2: allocate from a child in a new subdivision.
 * We would prefer to allocate only at the immediate parent of level_ix (level_ix+1).
 * But if no free space is
 * known at that level, we have to move higher, and allocate twice, or more.
 * Go up from (level_ix+1) until we find a run of ct_atomic_words free elements.
 * Allocate that run as a subdivision, and iteratively allocate new subdivision
 * until we have allocated a subdivision for the target level.
 */
bool ccpm::area_top::allocate_recovery_2(
	persist_type persist_
	, const level_hints_vec::iterator level_
)
{
	auto parent_level = level_ + 1;
	list_item *subdivide_tier_ptr = nullptr;
	for (
		;
			(
				parent_level != _level.end()
				&&
				(
					subdivide_tier_ptr =
						parent_level->find_free_ctl(area_ctl::ct_atomic_words)
				)
				==
				parent_level->tier_end()
			)
		; ++parent_level )
	{
	}

	if (
		parent_level != _level.end()
		&&
		subdivide_tier_ptr != parent_level->tier_end()
	)
	{
		auto subdivide_level = parent_level;
		do
		{
			auto level = subdivide_level;
			auto &free_ctl = *subdivide_tier_ptr;
			/* A viable allocation exists at free_ctl. Use it. */
			auto parent = static_cast<area_ctl *>(free_ctl.next());
			/* The first parent will exist because find_free_ctl found it,
			 * and subsequent parents will exist because new_subdivision
			 * just created them.
			 */
			assert(parent);
			/* carve out a new area_ptr from viable */
			auto child = parent->new_subdivision(persist_, 1U);
			++level->_ct_subdivision;
			/* The parent may have a new, shorter longest run.
			 * If so, move it to a new chain within the level object */
			auto parent_longest_run = parent->el_max_free_run();
			parent->remove();
			if ( parent_longest_run != 0 )
			{
				level->
					tier_from_run_length(parent_longest_run)->
						insert_after(parent);
			}
			/* Link in the new area */
			--subdivide_level;
			auto child_level = subdivide_level;
			subdivide_tier_ptr =
				child_level->tier_from_run_length(child->el_max_free_run());
			subdivide_tier_ptr->insert_after(child);
			if ( trace_fine() )
			{
				print(*_o, std::ios_base::hex);
			}
		} while ( subdivide_level != level_ );

		/* A child with the maximum possible number of free elements now exists
		 * at the necessary level. Retry the allocation, which should now succeed.
		 */
		return true;
	}
	else
	{
		return false;
	}
}

void ccpm::area_top::allocate(
	persist_type persist_
	, void * & ptr_
	, const std::size_t bytes_
	, const std::size_t alignment_
)
{
	if ( _ctl )
	{
		++_ct_allocation;
		const auto bytes = std::max(bytes_, area_ctl::min_alloc_size);
		/* If there is no chained area suitable for allocation we will try to add
		 * areas to the lists of chained areas. If any areas are added, we then
		 * restart the search here:
		 */
		auto level_ix = area_ctl::size_to_level(bytes);
RETRY:
		/* If _level is a feasible level ... */
		if ( level_ix < _level.size() )
		{
			/* TODO: consider whether it is possible to reduce the required
			 * run length by combining the run_length_for_use and
			 * run_length_for_alignment penalties.
			 */
			/* Number of elements necessary to contain bytes */
			const auto run_length_for_use =
				unsigned(
					div_round_up(bytes, area_ctl::level_to_element_size(level_ix))
				);

			/* Number of consecutive free elements sufficent to ensure run_length
			 * elements aligned to the alignment specification.
			 */
			const auto run_length_for_alignment =
				run_length_for_use
				+
				unsigned(
					std::max(
						/* Slack needed because the alignment request exceeds the
						 * natural alignment of the level */
						int(alignment_ / area_ctl::level_to_element_size(level_ix)) - 1,
						/* Slack needed because the whole region is poorly aligned */
						int(alignment_ != 0 && reinterpret_cast<uintptr_t>(_ctl) % alignment_ != 0)
					)
				)
				;

			if ( trace_coarse() )
			{
				if ( run_length_for_use < run_length_for_alignment )
				{
					PWRN(
						"byte count %zu, run length increased fron %u to %u erase 'natural' slack %zu/%zu == %u 'region' slack %u"
						, bytes
						, run_length_for_use
						, run_length_for_alignment
						, alignment_
						, area_ctl::level_to_element_size(level_ix)
						, unsigned(alignment_ / area_ctl::level_to_element_size(level_ix))
						, unsigned( bool(alignment_ != 0 && reinterpret_cast<uintptr_t>(_ctl) % alignment_ != 0))
					);
				}
			}

#if 0
/* removed becauuse it does not account for small sizes combined with large alignments */
			/* Should have at most doubled the number of elements */
			assert(run_length_for_alignment <= run_length_for_use * 2);
#endif

			auto level_it = _level.begin() + level_ix;
			/* Additional run length may have pushed the allocation to a higher level.
			 * If so, move up one level and retry.
			 */
			if ( level_it->size() <= run_length_for_alignment )
			{
				if ( trace_coarse() )
				{
					PLOG(
						"bytes: %zu, increment level %u because size %u <= run_length_for_alignment %u" //
						, bytes //
						, level_ix //
						, level_it->size() //
						, run_length_for_alignment //
					);
				}
				++level_ix;
				goto RETRY;
			}

			/* The elements used for headers may make it impossible for a level
			 * to have any area_ctl, even a newly-created one with enough free elements.
			 * This can happen if the word_count is 1.
			 * If this is the case, move up one level and retry.
			 */
			if ( area_ctl::max_free_element_count(level_ix) < run_length_for_alignment )
			{
				if ( trace_coarse() )
				{
					PLOG(
						"bytes: %zu, increment level %u because max_free_element_count %u < run_length_for_alignment %u" //
						, bytes //
						, level_ix //
						, area_ctl::max_free_element_count(level_ix) //
						, run_length_for_alignment //
					);
				}
				++level_ix;
				goto RETRY;
			}

			allocate_strategy_1(
				persist_
				, ptr_
				, bytes
				, alignment_
				, level_it
				, run_length_for_alignment
			);

			if ( ptr_ == nullptr && trace_fine() )
			{
				PLOG("ctl %p no cached element available", common::p_fmt(_ctl));
			}

			while ( ptr_ == nullptr
				&&
				(
					/* Recovery 1: rechain an existing but unchained area_ctl
					 */
					allocate_recovery_1()
					||
					/* Recovery 2: make a new subdivision */
					allocate_recovery_2(persist_, level_it)
				)
			)
			{
				allocate_strategy_1(
					persist_
					, ptr_
					, bytes
					, alignment_
					, level_it
					, run_length_for_alignment
				);
				if ( ptr_ == nullptr && trace_fine() )
				{
					PLOG("ctl %p rechain/subdivision failed to make cached element available", common::p_fmt(_ctl));
				}
			}
		}
		else
		{
			if ( trace_fine() )
			{
				PLOG("have %zu mac levels, need level %u for %zu requested, %zu rounded bytes", _level.size()-1, level_ix, bytes_, bytes);
			}
		}
	}
}

void ccpm::area_top::deallocate(
	persist_type persist_
	, void * & ptr_, std::size_t bytes_
)
{
	if ( _ctl )
	{
		auto bytes = std::max(bytes_, area_ctl::min_alloc_size);
		_ctl->deallocate(persist_, this, ptr_, bytes);
	}

}

void ccpm::area_top::print(
	std::ostream &o_
	, std::ios_base::fmtflags format_
) const
{
	auto level_ = 0;
	const auto si = std::string(static_cast<std::size_t>(level_ * 2), ' ');
	if ( _ctl )
	{
		o_ << si << "area_top:\n";
		_ctl->print(o_, level_ix_t(level_ + 1), format_);
		o_ << si << "area_top end\n";
	}
	else
	{
		o_ << si << "area_top: not initialized\n";
	}
}

void ccpm::area_top::print_ctls(std::ostream *o_, std::ios_base::fmtflags format_) const
{
	if ( o_ )
	{
		auto indent = 0;
		const auto si = std::string(static_cast<std::size_t>(indent * 2), ' ');

		auto ix = 0;
		*o_ << si << "area_top levels:\n";
		for ( const auto & lv : _level )
		{
			*o_ << si << " level " << ix << ": ";
			lv.print(*o_, level_ix_t(indent + 1), format_);
			++ix;
		}
		*o_ << si << "area_top levels end\n";
	}
}

bool ccpm::area_top::contains(const void *p) const
{
	return _ctl && _ctl->contains(p);
}

void ccpm::area_top::set_root(const byte_span& iov, persist_type persist_)
{
  _ctl->set_root(iov, persist_);
}

auto ccpm::area_top::get_root() const -> byte_span
{
  return _ctl->get_root();
}

void ccpm::level_hints::print(
	std::ostream &o_
	, level_ix_t level_
	, std::ios_base::fmtflags // size_format_
) const
{
	const auto si = std::string(level_*2, ' ');
	o_ << si;
	for ( const auto &fc : _free_ctls )
	{
		auto ct = fc.count();
		o_ << ( ct == 0 ? '-' : ct < 10 ? char('0'+ct) : '!' );
	}
	o_ << "\n";
}
