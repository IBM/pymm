/*
   Copyright [2018-2021] [IBM Corporation]
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

/*
 * Hopscotch hash table - template Key, Value, and allocators
 */

#include "alloc_key.h" /* AK_ACTUAL */
#include "bits_to_ints.h"
#include "hop_hash_exceptions.h"
#include "hop_hash_log.h"
#include "key_not_found.h"
#include "perishable.h"
#include "perishable_expiry.h"
#include "persistent.h"
#include "test_flags.h"

#include <common/perf/tm.h>
#include <boost/iterator/transform_iterator.hpp>

#include <algorithm>
#include <cassert>
#include <exception>
#include <sstream> /* ostringstream */
#include <thread> /* this_thread */
#include <utility> /* move */


/*
 * ===== hop_hash_base =====
 */

namespace
{
	const char *hstore_consistency_check() { return std::getenv("HSTORE_CONSISTENCY_CHECK"); }
}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::hop_hash_base(
		AK_ACTUAL
		persist_data_type *pc_
		, construction_mode mode_
		, const Allocator &av_
	)
		: hop_hash_allocator<Allocator>{av_}
		, persist_map_controller_t(AK_REF av_, pc_, mode_)
		, _hasher{}
		, _auto_resize{true}
		, _consistency_check(hstore_consistency_check() ? atoi(hstore_consistency_check()) : 0)
	{
		const auto bp_src = this->persist_map_controller_t::bp_src();
		const auto bc_dst =
			boost::make_transform_iterator(_bc, std::mem_fn(&bucket_control_t::_buckets));
		/*
		 * Copy bucket pointers from pmem to DRAM.
		 */
		std::transform(
			bp_src
			, bp_src + _segment_capacity
			, bc_dst
			, [] (const auto &c) {
				return c ? &*c : nullptr;
			}
		);

		{
			segment_layout::six_t ix = 0U;
			_bc[ix].extend(_bc[ix].buckets(), &_bc[0], &_bc[0], ix);

			if ( mode_ == construction_mode::reconstitute )
			{
				_bc[ix].reconstitute(av_);
			}
		}

		for ( segment_layout::six_t ix = 1U; ix != this->persist_map_controller_t::segment_count_actual().value_not_stable(); ++ix )
		{
			_bc[ix-1]._next = &_bc[ix];
			_bc[0]._prev = &_bc[ix];
			_bc[ix].extend(_bc[ix].buckets(), &_bc[ix-1], &_bc[0], ix);

			if ( mode_ == construction_mode::reconstitute )
			{
				_bc[ix].reconstitute(av_);
			}
		}
		hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION, " segment_count ", this->persist_map_controller_t::segment_count_actual().value_not_stable()
			, " segment_count_specified ", this->persist_map_controller_t::segment_count_specified());
		/* If table allocation incomplete (perhaps in the middle of a resize op), resize until large enough. */

		hop_hash_log<TEST_HSTORE_PERISHABLE>::write(LOG_LOCATION, "HopHash base constructor: "
			, (this->is_size_stable() ? "stable" : "unstable"), " segment_count ", this->segment_count_actual().value_not_stable());

		if ( ! this->persist_map_controller_t::segment_count_actual().is_stable() )
		{
			/* ERROR: reconstruction code for resize state not written. */
			const auto ix = this->persist_map_controller_t::segment_count_actual().value_not_stable();
			bucket_control_t &junior_bucket_control = _bc[ix];

			junior_bucket_control.extend(
				this->persist_map_controller_t::resize_restart_prolog()
				, &_bc[ix-1]
				, &_bc[0]
				, ix
			);

			junior_bucket_control.reconstitute(av_);

			hop_hash_log<HSTORE_TRACE_RESIZE>::write(LOG_LOCATION
				, " finishing resize in constructor"
			);
			resize_pass2();
			this->persist_map_controller_t::resize_epilog();
		}

		hop_hash_log<TEST_HSTORE_PERISHABLE>::write(LOG_LOCATION, "HopHash base constructor: "
			, (this->persist_map_controller_t::is_size_stable() ? "stable" : "unstable"), " size ", this->persist_map_controller_t::size_unstable());

		if ( ! this->persist_map_controller_t::is_size_stable() )
		{
			/* The size is unknown.
			 * Also, the content state (FREE or IN_USE) of each element is unknown.
			 * Scan all elements to determine the size and the content state.
			 */
			size_type s = 0U;
			/* In order to develop a mask of "used" (non-free) locations,
			 * examine the members of every owner starting with the leftmost
			 * bucket which can include bi in its owner.
			 */
			auto sb = make_segment_and_bucket_for_iterator(0);
			const auto sb_end =
				make_segment_and_bucket_at_end();
			/* Develop the mask as if we have not yet read the ownership for element 0 */
			auto owned_mask = ( in_use_by_owner_mask(sb) << 1U ) & owner::ownership_bit_mask();
			for ( ; sb != sb_end ; sb.incr_without_wrap() )
			{
				auto owner_lk = make_owner_unique_lock(sb);
				owned_mask = (owned_mask >> 1U) | locate_owner(sb).ownership_bits(owner_lk);
				auto content_lk = make_content_unique_lock(sb);
				bool in_use = (owned_mask & 1);
				s += in_use;
				/* conditional - to suppress unnecessary persists */
				if ( content_lk.owner_ref().is_adjacent_content_in_use() != in_use )
				{
					hop_hash_log<TEST_HSTORE_PERISHABLE>::write(LOG_LOCATION, "Element ", sb, " changed to ", (in_use ? "in_use" : "free"));

					content_lk.owner_ref().set_adjacent_content_in_use(in_use);

					/*
					 * Persists the entire owner, although only the in_use bit has changed.
					 * Persisting only the in_uze bit would require an additional function
					 * in persist_map_controller_t.
					 */
					this->persist_map_controller_t::persist_owner(content_lk.owner_ref(), "summary content state from owner");
				}
			}
			this->persist_map_controller_t::size_set(s);

			hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION, "Restored size ", size());

		}
		check_consistency();
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::~hop_hash_base()
	{
		perishable::report();
	}

/*
 * Return a bit mask describing which contents are owned by the owner at a_.
 * bit 0 (LSB) is 1 iff the owner at a_ owns the content at a_, bit n is 1 iff
 * the owner at a_ owns the content at a_+n (modulo size of the hash table).
 *
 * Note: Whether content at a location a is "free" or "in use" can be determined
 * in two ways:
 * 1) examination of the "ownership" bits of all possible owners of the content,
 *    which are the the owners at locations [a .. a-owner::size).
 * 2) a summary bit at location a.
 */
template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<
		Key, T, Hash, Pred, Allocator, SharedMutex
	>::in_use_by_owner_mask(
		const segment_and_bucket_t &a_
	) const -> owner::value_type
	{
		auto c = in_use_by_owner_pre_mask(a_);
		return finish_owner_mask(c, a_);
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::is_in_use(
		const segment_and_bucket_t &a_
	) const -> bool
	{
		auto &b_src = a_.deref();
		return b_src.is_adjacent_content_in_use();
	}

/* From a hash, return the index of the bucket (in seg_entry) to which it maps */
template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::bucket_ix(
		const hash_result_t h_
	) const -> bix_t
	{
		hop_hash_log<HSTORE_TRACE_BUCKET_IX>::write(LOG_LOCATION, "h_ ", h_
			, " mask ", mask(), " -> ", ( h_ & mask() ));
		return h_ & mask();
	}

/* From a hash, return the index of the bucket (in seg_entry) to which it maps,
 * including a single new segment not in count.
 * For use when "resizeing" the hash table to double its pervious size.
 */
template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<
		Key, T, Hash, Pred, Allocator, SharedMutex
	>::bucket_expanded_ix(
		const hash_result_t h_
	) const -> bix_t
	{
		auto mask_expanded = (mask() << 1U) + 1;
		hop_hash_log<HSTORE_TRACE_BUCKET_IX>::write(LOG_LOCATION, "h_ ", h_
			, " mask ", mask_expanded, " -> ", ( h_ & mask_expanded ));
		return h_ & mask_expanded;
	}

/*
 * Starting at bucket bi_, scan forward to find the next free bucket.
 8 Throws hop_hash_full is there in free bucket.
 *
 * Precondition: hold owner unique lock on bi.
 * Exit with content unique lock on the free bucket (while retaining owner
 * unique lock on bi).
 */
template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<
		Key, T, Hash, Pred, Allocator, SharedMutex
	>::nearest_free_bucket(
		segment_and_bucket_t bi_
	) -> content_unique_lock_t
	{
		/* acquire content_lock(cursor) by virtue of owning owner_lock(cursor) */
		const auto start = bi_;
		auto content_lk = make_content_unique_lock(bi_);

		while ( is_in_use(content_lk.sb()) )
		{
			bi_.incr_with_wrap();
			content_lk = make_content_unique_lock(bi_);
			if ( content_lk.sb() == start )
			{
				throw hop_hash_full{bi_.index(), bucket_count()};
			}
		}

		hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION
			, "(", start.index(), ") => ", content_lk.index());

		content_lk.assert_clear(true, *this);
		return content_lk;
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<
		Key, T, Hash, Pred, Allocator, SharedMutex
	>::distance_wrapped(
		bix_t first, bix_t last
	) const -> unsigned
	{
		return
			unsigned
			(
				(
					last < first
					? last + bucket_count()
					: last
				) - first
			);
	}

/* Precondition: holds owner_unique_lock on bi_.
 *
 * Receives an owner index bi_, and a content unique lock
 * on the free content at b_dst_lock_
 *
 * Returns a free content index within range of bi_, and a lock
 * on that free content index
 */
template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<
		Key, T, Hash, Pred, Allocator, SharedMutex
	>::make_space_for_insert(
		bix_t bi_
		, content_unique_lock_t b_dst_lock_
	) -> content_unique_lock_t
	{
		auto free_distance = distance_wrapped(bi_, b_dst_lock_.index());
		while ( owner::size <= free_distance )
		{
			hop_hash_log<HSTORE_TRACE_LOCK>::write(LOG_LOCATION
				, "owner ", bi_
				, " nearest free location ", b_dst_lock_.index()
				, " owner size ", owner::size
				, " < free distance ", free_distance
			);
			hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION
				, "owner ", bi_
				, " nearest free location ", b_dst_lock_.index()
				, " owner size ", owner::size
				, " < free distance ", free_distance
				, "\n"
				, dump<HSTORE_TRACE_MANY>::make_hop_hash_dump(*this)
			);

			/*
			 * Relocate an element in some possible owner of the free element (b_dst_lock_)
			 * to somewhere else in that owner's range.
			 *
			 * The owners to check are, in order,
			 *   b_dst_lock_-owner::size-1 through b_dst_lock__-1.
			 * That is (currently)
			 *  b_dst_lock_-owner::size-1 through b_dst_lock__-1
			 */

			/* A bucket containing an item to be relocated.
			 * The first bucket to consider is at the destination *minus* (owner size - 1).
			 */
			auto owner_lock = make_owner_unique_lock(b_dst_lock_, owner::size - 1U);
			/* Every item in bucket owner_lock precedes b_dst_lock_,
			 * and is eligible for move. Every time we move forward the next owner_lock
			 * the mask loses a high 1 bit.
			 */
			owner::value_type eligible_items = owner::ownership_bit_mask();

			hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION
				, "owner_lock ", owner_lock.sb(), " ", owner_lock.index()
				, " b_dst_lock_ ", b_dst_lock_.sb(), " ", b_dst_lock_.index()
				, " ne ", owner_lock.sb() != b_dst_lock_.sb()
				, " owner_lock.ref().value ", owner_lock.ref().ownership_bits(owner_lock)
				, " eligible items ", eligible_items
				, " eq ", (owner_lock.ref().ownership_bits(owner_lock) & eligible_items) == 0
			);
			while (
				owner_lock.sb() != b_dst_lock_.sb()
				&&
				(owner_lock.ref().ownership_bits(owner_lock) & eligible_items) == 0
			)
			{
				auto sb = owner_lock.sb();
				sb.incr_with_wrap();
				owner_lock = make_owner_unique_lock(sb);
				/* The leftmost eligible item is no longer eligible; remove it */
				eligible_items >>= 1U;
				hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION
					, "owner_lock ", owner_lock.sb(), " ", owner_lock.index()
					, " b_dst_lock_ ", b_dst_lock_.sb(), " ", b_dst_lock_.index()
					, " owner_lock.ref().value ", owner_lock.ref().ownership_bits(owner_lock)
					, " eligible items ", eligible_items
				);
			}

			/* postconditions
			 *   owner_lock.index() == b_dst_lock_ : we ran out of items to examine
			 *  or
			 *   (_buckets[lock.index()]._owner & eligible_items) != 0 : at least one item
			 *     in owner is eliglbe for the move; the best item is the 1 at the
			 *     smallest index in _buckets[lock.index()]._owner & eligible_items.
			 */
			if ( owner_lock.sb() == b_dst_lock_.sb() )
			{
				/* If no bucket was found which owned possible elements to move,
				 * we are stuck, no element can be moved.
				 */
				throw move_stuck(bi_, bucket_count());
			}

			hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION
				, "owner_lock ", owner_lock.sb()
				, " can own ", bi_
			);

			const auto c = owner_lock.ref().ownership_bits(owner_lock) & eligible_items;
			assert(c != 0);

			/* find index of first (rightmost) one in c */
			const auto p = owner::nearest_owned_content_offset(c);

			hop_hash_log<HSTORE_TRACE_LOCK>::write(LOG_LOCATION
				, "intend to transfer content owned by ", owner_lock.index()
				, " from ", owner_lock.index(), "+", p
				, " to ", b_dst_lock_.index()
			);

			/* content to relocate */
			auto b_src_lock = make_content_unique_lock(owner_lock, p);
			hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION
				, "intend to transfer content owned by ", owner_lock.sb()
				, "  from ", b_src_lock.sb()
				, " to ", b_dst_lock_.sb()
			);

			b_src_lock.assert_clear(false, *this);
			b_dst_lock_.assert_clear(true, *this);
#if TRACK_OWNER
			b_src_lock.ref().owner_verify(owner_lock.index());
#endif
			std::ostringstream c_old;
#if HSTORE_TRACE_MANY
			{
				c_old << dump<HSTORE_TRACE_MANY>::make_owner_print(this->bucket_count(), owner_lock);
			}
#endif

			/* The owner will
			 *  a) lose at element at position p and
			 *  b) gain the element at position b_dst_lock_ (relative to lock.index())
			 */
			const auto q = distance_small(owner_lock.sb(), b_dst_lock_.sb());
			/*
			 * ownership is moving from bf to owner
			 * Mark the size "unstable" to indicate that content state must be rebuilt
			 * in case of a crash.
			 *
			 * 1. mark src EXITING and dst ENTERING
			 *   flush
			 * 2. update owner (atomic nove of in_use bit)
			 *   flush
			 * 3. mark src FREE and dst IN_USE
			 *   flush
			 */
			{
				persist_size_change<Allocator, size_no_change> s(*this);
				assert(is_in_use(b_src_lock.sb()));
				assert(!is_in_use(b_dst_lock_.sb()));

				b_dst_lock_.ref().content_share(b_src_lock.ref());
				b_dst_lock_.owner_ref().set_adjacent_content_in_use(true);

				this->persist_map_controller_t::persist_content(b_src_lock.ref(), "content free");
				this->persist_map_controller_t::persist_content(b_dst_lock_.ref(), "content in use");

				owner_lock.ref().move(q, p, owner_lock);
				this->persist_map_controller_t::persist_owner(owner_lock.ref(), "owner update");

				b_src_lock.ref().content_erase();
				b_src_lock.owner_ref().set_adjacent_content_in_use(false);

				b_src_lock.assert_clear(true, *this);
				b_dst_lock_.assert_clear(false, *this);
			}
			/* New free location */

			hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION
				, " bucket ", owner_lock.index()
				, " move ", b_src_lock.index(), "->", b_dst_lock_.index()
				, " "
				, c_old.str()
				, "->"
				, dump<HSTORE_TRACE_MANY>::make_owner_print(this->bucket_count(), owner_lock)
			);

			b_dst_lock_ = std::move(b_src_lock);

			free_distance = distance_wrapped(bi_, b_dst_lock_.index());
		}
		/* postcondition:
		 *   free_distance < owner::size
		 *
		 * Enter the new item in b_dst_lock_ and update the ownership at bi_
		 */

		hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION, " exit, free distance ", free_distance
			, "\n", dump<HSTORE_TRACE_MANY>::make_hop_hash_dump(*this));

		b_dst_lock_.assert_clear(true, *this);
		return b_dst_lock_;
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	void impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::ownership_check() const
	{
		for ( auto i = bix_t(); i != bucket_count(); ++i )
		{
			auto sbw = make_segment_and_bucket(i);
			auto owner_lk = make_owner_shared_lock(sbw);
			auto sbo = sbw;
			bix_t owner_ix = std::numeric_limits<bix_t>::max();
			if ( owner_lk.ref().ownership_bits(owner_lk) || owner_lk.ref().is_adjacent_content_in_use() )
			{
				hop_hash_log<true>::write(LOG_LOCATION, i, " ownership mask ", std::hex, ints_to_string(bits_to_ints(owner_lk.ref().ownership_bits(owner_lk), owner_lk.index())), " ", (owner_lk.ref().is_adjacent_content_in_use() ? "in_use" : "free"));
			}
			if ( (owner_lk.ref().ownership_bits(owner_lk) >> (owner::size-1U)) & 1 )
			{
				owner_ix = owner_lk.index();
			}
			for ( owner::index_type j = 1; j != owner::size; ++j )
			{
				sbo.incr_with_wrap();
				auto other_lk = make_owner_shared_lock(sbo);
				if ( (owner_lk.ref().ownership_bits(owner_lk) >> j) & other_lk.ref().ownership_bits(other_lk) )
				{
					hop_hash_log<true>::write(LOG_LOCATION, "XX conflicting owner masks "
						, std::dec, owner_lk.index(), "=", std::hex, owner_lk.ref().ownership_bits(owner_lk)
						, std::dec, other_lk.index(), "=", std::hex, other_lk.ref().ownership_bits(other_lk)
					);
				}
				if ( (other_lk.ref().ownership_bits(other_lk) >> ((owner::size-1U)-j)) & 1 )
				{
					if ( owner_ix == std::numeric_limits<bix_t>::max() )
					{
						owner_ix = j;
					}
					else
					{
						hop_hash_log<true>::write(LOG_LOCATION, "XX two owners: ", owner_ix, " and ", other_lk.index(), " own ", owner_ix + (owner::size-1U));
					}
				}
			}
			auto other_lk = make_owner_shared_lock(sbo);
			if ( other_lk.ref().is_adjacent_content_in_use() && owner_ix == std::numeric_limits<bix_t>::max() )
			{
				hop_hash_log<true>::write(LOG_LOCATION, "XX content at ", other_lk.index(), " in use without owner");
			}
			if ( ! other_lk.ref().is_adjacent_content_in_use() && owner_ix != std::numeric_limits<bix_t>::max() )
			{
				hop_hash_log<true>::write(LOG_LOCATION, "XX content at ", other_lk.index(), " not in use but owned by ", owner_lk.index(), "+", owner_ix);
			}
		}
	}

template <typename T>
	struct consistency_guard
	{
	private:
		const T *_hh;
	public:
		explicit consistency_guard(const T *hh_)
			: _hh(hh_)
		{
			_hh->check_consistency();
		}
		consistency_guard(consistency_guard &) = delete;
		consistency_guard &operator=(consistency_guard &) = delete;
		~consistency_guard()
		{
			_hh->check_consistency();
		}
	};

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	template <typename Lock, typename K>
		auto impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::locate_key(
			TM_ACTUAL
			Lock &bi_
			, const K &k_
		) const -> segment_and_bucket_t
		{
			return locate_key_inner(TM_REF bi_.ref().ownership_bits(bi_), bi_.sb(), common::string_view(k_.data(), k_.size()));
		}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::locate_key_inner(
		TM_ACTUAL
		owner::value_type ownership_bits_
		, segment_and_bucket_t sb_
		, const common::string_view k_
	) const -> segment_and_bucket_t
	{
		TM_SCOPE()
		/* Use the ownership bits to filter key checks, a performance aid to reduce the number of key compares. */
		auto distance_to_end = owner::size;

		while ( ownership_bits_ )
		{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
			auto distance = unsigned(__builtin_ctzl(ownership_bits_));
#pragma GCC diagnostic pop
			sb_.add_small(distance);
			distance_to_end -= distance;
			if ( key_equal()(sb_.deref().key(), k_) )
			{
				hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION, " returns (success) ", sb_.index());
				return sb_;
			}
			ownership_bits_ = (ownership_bits_ >> distance) & ~owner::value_type(1);
		}

		sb_.add_small(distance_to_end);
		return sb_;
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	template <typename ... Args>
		auto impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::emplace(
			AK_ACTUAL
			TM_ACTUAL
			Args && ... args
		) -> std::pair<iterator, bool>
		try
		{
			TM_SCOPE()
			consistency_guard<impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>> g(this);
			hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION, " BEGIN LIST\n"
				, dump<HSTORE_TRACE_MANY>::make_hop_hash_dump(*this)
				, LOG_LOCATION, " END LIST"
			);

		RETRY:
			/* convert the args to a value_type */
			value_type v(std::forward<Args>(args)...);

			/* The bucket in which to place the new entry */
			auto sbw = make_segment_and_bucket(bucket(v.first));
			auto owner_lk = make_owner_unique_lock(sbw);

			TM_SCOPE(check_exists)
			auto key_bi = locate_key(TM_REF owner_lk, v.first);
			/* If the key is found, refuse to emplace */
			{
				auto content_offset = distance_small(sbw, key_bi);
				if ( content_offset != owner::size )
				{
					return {iterator{sbw, content_offset}, false};
				}
			}

			/* the nearest free bucket */
			try
			{
				TM_SCOPE(nearest_free)
				auto b_dst = nearest_free_bucket(sbw);
				TM_SCOPE(make_space)
				b_dst = make_space_for_insert(owner_lk.index(), std::move(b_dst));

				b_dst.assert_clear(true, *this);
				const auto content_index = distance_small(owner_lk.sb(), b_dst.sb());

				/* 4-step change to owner:
				 *  1. mark the size "unstable"
				 *   flush (8 bytes)
				 *    (When size is unstable, content state must be reconstructed from
				 *    the ownership: owned content is IN_USE; unowned content is FREE.)
				 *  2. the new content (already entered)
				 *   flush (56 bytes. Could be 64, as there is no harm in flushing the adjacent but unrelated owner field.)
				 *  3. atomically update the owner
				 *   flush (8 bytes)
				 *  4. mark the size as "stable"
				 *   flush (8 bytes)
				 */
				{
					TM_SCOPE(insert)
					persist_size_change<Allocator, size_incr> s(*this);
					b_dst.ref().content_construct(owner_lk.index(), std::move(v));
					if ( owner_lk.index() == b_dst.index() )
					{
						owner_lk.ref().set_adjacent_content_in_use();
					}
					else
					{
						auto adjacent_owner_lk = make_owner_unique_lock(b_dst.sb());
						adjacent_owner_lk.ref().set_adjacent_content_in_use();
					}
					this->persist_map_controller_t::persist_content(b_dst.ref(), "content in use");
					owner_lk.ref().insert(
						owner_lk.index()
						, content_index
						, owner_lk
						, gsl::not_null<persist_map_controller_t *>(this)
					);
					this->persist_map_controller_t::persist_owner(owner_lk.ref(), "owner emplace");
					hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION, " bucket ", owner_lk.index()
						, " store at ", b_dst.index(), " "
						, dump<HSTORE_TRACE_MANY>::make_owner_print(this->bucket_count(), owner_lk)
						, " ", b_dst.ref());
				}
				/* persist_size_change may have failed to due to perishable counter, but the exception
				 * could not be propagated as an exception because it happened in a destructor.
				 * Throw the exception here.
				 */
				perishable::test();
				return {iterator{sbw, content_index}, true};
			}
			catch ( const no_near_empty_bucket &e )
			{
				TM_SCOPE(resize)
				if ( _auto_resize )
				{
					owner_lk.unlock();

					hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION, "1. before resize\n", dump<HSTORE_TRACE_MANY>::make_hop_hash_dump(*this));

					if ( segment_count() < _segment_capacity )
					{
						resize(AK_REF0);
						hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION, "2. after resize\n", dump<HSTORE_TRACE_MANY>::make_hop_hash_dump(*this));
						goto RETRY;
					}
				}
				throw;
			}
		}
		catch ( const perishable_expiry & )
		{

			hop_hash_log<trace_perishable_expiry>::write(LOG_LOCATION, "perishable expiry dump\n"
				, dump<trace_perishable_expiry>::make_hop_hash_dump(*this)
			);

			throw;
		}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::insert(
		TM_ACTUAL
		const value_type &v_
	) -> std::pair<iterator, bool>
	{
		TM_SCOPE()
		return emplace(TM_REF v_);
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	void impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::resize(AK_ACTUAL0)
	{
		hop_hash_log<HSTORE_TRACE_RESIZE>::write(LOG_LOCATION
			, " capacity ", bucket_count()
			, " size ", size()
		);
		{
#if 0
			monitor_extend<Allocator> m{bucket_allocator_t(av)};
#endif
			_bc[segment_count()].extend(
				this->persist_map_controller_t::resize_prolog(AK_REF0)
				, &_bc[segment_count()-1]
				, &_bc[0]
				, segment_count()
			);
		}

		/* adjust count and everything which depends on it (size, mask) */

		/* PASS 1: copy content */

		hop_hash_log<HSTORE_TRACE_RESIZE>::write(LOG_LOCATION, "before pass 1"
			, "\n", dump<HSTORE_TRACE_RESIZE>::make_hop_hash_dump(*this));

		resize_pass1();

		hop_hash_log<HSTORE_TRACE_RESIZE>::write(LOG_LOCATION, "after pass 1"
			, "\n", dump<HSTORE_TRACE_RESIZE>::make_hop_hash_dump(*this));

		/*
		 * Crash-consistency notes.
		 *
		 * Until the start of pass 2, there is no need (for crash-consistency purposes)
		 * to remember that we are in a resize operation. The need for resize can be
		 * rediscovered and the operations through pass1 can be re-executed.
		 *
		 * Pass2 will update owners, so during that pass we must recognize that
		 * a restart must not re-excute pass1.
		 *
		 * Starting with Pass2, the "junior content" (the allocated but not live segment)
		 * must be scanned and its contents reconstituted by the allocator.
		 *
		 * Exactly when the "actual size" changes to encompass the junior content,
		 * the junior content scan must no longer be performed. Therefore the "actual size"
		 * and the "need to scan junior content bit" must be updated together.
		 */

		this->persist_map_controller_t::resize_interlog();

		/* PASS 2: remove old content, update owners. Some old content mave have been
		 * removed if pass 2 was restarted, so use the new (junior) content to drive
		 * the operations.
		 */
		resize_pass2();

		this->persist_map_controller_t::resize_epilog();
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	void impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::resize_pass1()
	{
		/* PASS 1: copy content */

		bix_t ix_senior = 0U;
		const auto sb_senior_end =
			make_segment_and_bucket_at_end();

			hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION
				, " bucket_count ", bucket_count()
				, " sb_senior_end ", sb_senior_end.si(), ",", sb_senior_end.bi()
			);

		for (
			auto sb_senior = make_segment_and_bucket(0U)
			; sb_senior != sb_senior_end
			; sb_senior.incr_without_wrap(), ++ix_senior
		)
		{
			auto senior_content_lk = make_content_unique_lock(sb_senior);

			/* special locate, used to access junior new buckets */
			content<value_type> &junior_content = _bc[segment_count()].buckets()[ix_senior];
			owner &junior_owner = _bc[segment_count()]._buckets[ix_senior];
			if ( is_in_use(senior_content_lk.sb()) )
			{
				/* examine hash(key) to determine whether to copy content */
				auto hash = _hasher.hf(senior_content_lk.ref().key());
				auto ix_owner = bucket_expanded_ix(hash);
				/*
				 * [ix_owner, ix_owner + owner::size) is permissible range for content
				 */
				if ( ix_owner <= ix_senior && ix_senior < ix_owner + owner::size )
				{
					/*
					 * content can stay where it is because bucket index index MSB is 0
					 */

					hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION
						, " ", ix_senior, " 1a no-relocate, owner ", bucket_ix(hash)
						, " -> ", ix_owner, ": content ", ix_senior
					);

				}
				else if (
					ix_senior < owner::size
					&&
					bucket_count()*2U < ix_owner + owner::size
				)
				{
					/* content can stay where it is because the owner wraps
					 * NOTE: this test is not exact, but is close enough if owner::size
					 * is equal to or less than half the minimum table size.
					 */

					hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION
						, " ", ix_senior, " 1b no-relocate, owner ", bucket_ix(hash)
						, " -> ", ix_owner, ": content ", ix_senior);

				}
				else
				{
					/* content must move */
					junior_content.content_share(senior_content_lk.ref(), ix_owner);
					junior_owner.set_adjacent_content_in_use();

					hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION
						, " ", ix_senior, " 1c relocate, owner ", bucket_ix(hash), " -> ", ix_owner
						, ": content ", ix_senior, " -> "
						, ix_senior + bucket_count());

				}
			}
			else
			{
				hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION, " ", ix_senior, " 1d empty");
			}
		}

		/* persist new content */
		this->persist_map_controller_t::persist_new_segment("pass 1 copied content");
	}

/* Returns true iff ownership wraps the table, i.e. the owner is near the end of the table
 * and the content index is near the beginning.
 *
 * ix_senior: index of content before move
 * junior_bucket_control: access to the new "junior" segment, to which content may move
 * populated_content_lk: lock providing access to the new location for the content (which
 *   may be the same as the old location).
 *
 * If the owning location changes, the old and new owner masks need to be updated.
 */
template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	bool impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::resize_pass2_adjust_owner(
		bix_t ix_senior
		, bucket_control_t &junior_bucket_control
		, content_unique_lock_t &populated_content_lk
	)
	{
		/* examine the key to locate old and new owners (owners) */
		auto hash = _hasher.hf(populated_content_lk.ref().key());
		/* Note: the owner needs to be changed iff the ownership range
		 * of ix_senior_owner + bucket_count() includes populated_content_lk.
		 */
		auto ix_senior_owner = bucket_ix(hash);
		if ( ! ( distance_wrapped(ix_senior_owner, ix_senior) < owner::size) )
		{
			hop_hash_log<true>::write(LOG_LOCATION, "senior owner ", ix_senior_owner
				, " cannot reach senior content ", ix_senior
				, ", which is more than ", owner::size, " entries away. "
				, "Buchet bucket count is ", bucket_count());
		}
		assert(distance_wrapped(ix_senior_owner, ix_senior) < owner::size);
		auto ix_junior_owner = bucket_expanded_ix(hash);

		hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION, ".2 content "
			, ix_senior, " -> ? "
			, " owner ", ix_senior_owner, " -> ", ix_junior_owner
		);

		if ( ix_senior_owner != ix_junior_owner )
		{
			auto senior_owner_sb = make_segment_and_bucket(ix_senior_owner);
			auto senior_owner_lk = make_owner_unique_lock(senior_owner_sb);
			owner_unique_lock_t
				junior_owner_lk(
					junior_bucket_control.buckets()[ix_senior_owner]
					, segment_and_bucket_t(&junior_bucket_control, ix_senior_owner)
					, junior_bucket_control._bucket_mutexes[ix_senior_owner]._m_owner
				);

			/*
			 * special locate, used before size has been updated,
			 * to access junior buckets
			 */
			auto &junior_owner = junior_bucket_control.buckets()[ix_senior_owner];
			auto owner_pos = distance_wrapped(ix_senior_owner, ix_senior);
			if ( ! ( owner_pos < owner::size) )
			{
				hop_hash_log<true>::write(LOG_LOCATION, "senior owner ", ix_senior_owner
					, " at invalid distance from senior content ", ix_senior
					, " with bucket count ", bucket_count());
			}
			assert(owner_pos < owner::size);
			junior_owner.insert(
				ix_junior_owner
				, owner_pos
				, junior_owner_lk
				, gsl::not_null<persist_map_controller_t *>(this)
			);
			hop_hash_log<false>::write(LOG_LOCATION, "resize moves content at +", owner_pos, " from senior ", senior_owner_lk.index(), " to junior ", junior_owner_lk.index());
			senior_owner_lk.ref().erase(
				owner_pos
				, senior_owner_lk
				, gsl::not_null<persist_map_controller_t *>(this)
			);
			this->persist_map_controller_t::persist_owner(junior_owner_lk.ref(), "pass 2 junior owner");
			this->persist_map_controller_t::persist_owner(senior_owner_lk.ref(), "pass 2 senior owner");
			/*
			 * If the owner index exceeds the content index (which can only happen due to wrap)
			 * the owner needs to change (be incremented by the bucket count).
			 */
			return ix_senior < ix_junior_owner;
		}
		return false;
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	void impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::resize_pass2()
	{

		hop_hash_log<HSTORE_TRACE_RESIZE>::write(LOG_LOCATION, "entering pass 2"
			, "\n", dump<HSTORE_TRACE_RESIZE>::make_hop_hash_dump(*this));
		/* PASS 2: remove old content, update owners. Some old content mave have been
		 * removed if pass 2 was restarted, so use the new (junior) content to drive
		 * the operations.
		 */
		const auto old_segment_count = this->persist_map_controller_t::segment_count_actual().value_not_stable();
		bucket_control_t &junior_bucket_control = _bc[old_segment_count];

		bix_t ix_senior = 0U;
		const auto sb_senior_end = make_segment_and_bucket_at_end();
		for (
			auto sb_senior = make_segment_and_bucket(0U)
			; sb_senior != sb_senior_end
			; sb_senior.incr_without_wrap(), ++ix_senior
		)
		{
			/* get segment count before it was made unstable */
			/* special locate, used before size has been updated
			 * to access junior buckets
			 */
			content_unique_lock_t
				junior_content_lk(
					junior_bucket_control.buckets()[ix_senior]
					, segment_and_bucket_t(&junior_bucket_control, ix_senior)
					, junior_bucket_control._bucket_mutexes[ix_senior]._m_content
				);

			const bool is_junior_in_use =
				owner_shared_lock_t(
					junior_bucket_control._buckets[ix_senior]
					, segment_and_bucket_t(&junior_bucket_control, ix_senior)
					, junior_bucket_control._bucket_mutexes[ix_senior]._m_owner
				).ref().is_adjacent_content_in_use();

			auto senior_content_lk = make_content_unique_lock(sb_senior);

			if ( is_junior_in_use )
			{
				/* The content has moved. */
				resize_pass2_adjust_owner(ix_senior, junior_bucket_control, junior_content_lk);
				senior_content_lk.ref().content_erase();
				make_owner_unique_lock(sb_senior).ref().set_adjacent_content_in_use(false);
			}
			else if ( make_owner_shared_lock(sb_senior).ref().is_adjacent_content_in_use() )
			{
				/* The content has not moved, but it might need a new owner */
				auto wrapped_owner = resize_pass2_adjust_owner(ix_senior, junior_bucket_control, senior_content_lk);
				if ( wrapped_owner )
				{
					hop_hash_log<TRACK_OWNER>::write(LOG_LOCATION, ".2b content at "
						, ix_senior, " wrapped owner adjustment");
#if TRACK_OWNER
					senior_content_lk.ref().owner_update(bucket_count());
#endif
				}
			}
			else
			{
			}
		}

		/* flush for state_set bucket_t::FREE in loop above. */
		this->persist_map_controller_t::persist_existing_segments("pass 2 senior content");
		/* flush for state_set owner::LIVE in loop above. */
		this->persist_map_controller_t::persist_new_segment("pass 2 junior owner");

		/* link in new segment in non-persistent circular list of segments */
		_bc[old_segment_count-1]._next = &junior_bucket_control;
		_bc[0]._prev = &junior_bucket_control;
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<
		Key, T, Hash, Pred, Allocator, SharedMutex
	>::make_segment_and_bucket_unsafe(
		bix_t ix_
	) const -> segment_and_bucket_t
	{
/* Similar to make_segment_and_bucket_for_iterator, but uses unsafe segment_count
 * Since this may refrences buckets entering through a "junior" segment, the
 * bucekt index is up to twice the bucket count.
 */
		assert( ix_ < bucket_count() * 2 );

		auto si =
			__builtin_expect((segment_layout::ix_high(ix_) == 0),false)
			? 0
			: segment_layout::log2(ix_high(ix_))
			;
		auto bi =
			(
				__builtin_expect((segment_layout::ix_high(ix_) == 0),false)
				? 0
				: ix_high(ix_) % (bix_t(1U) << (si-1))
			)
			*
			segment_layout::base_segment_size + segment_layout::ix_low(ix_)
			;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
		return
			si == segment_count_not_stable()
			? segment_and_bucket_t(&_bc[si-1], _bc[si-1].segment_size() ) /* end iterator */
			: segment_and_bucket_t(&_bc[si], bi)
			;
#pragma GCC diagnostic pop
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<
		Key, T, Hash, Pred, Allocator, SharedMutex
	>::make_segment_and_bucket(
		bix_t ix_
	) const -> segment_and_bucket_t
	{
		assert( ix_ < bucket_count() );
		/* Same as in make_segment_and_bucket_for_iterator, but with the knowledge that ix_ != bucket_count() */
#if 0
		return make_segment_and_bucket_for_iterator(ix_);
#else
		auto si =
			__builtin_expect((segment_layout::ix_high(ix_) == 0),false)
			? 0
			: segment_layout::log2(ix_high(ix_))
			;
		auto bi =
			(
				__builtin_expect((segment_layout::ix_high(ix_) == 0),false)
				? 0
				: ix_high(ix_) % (bix_t(1U) << (si-1))
			)
			*
			segment_layout::base_segment_size + segment_layout::ix_low(ix_)
			;

		return segment_and_bucket_t(&_bc[si], bi);
#endif
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<
		Key, T, Hash, Pred, Allocator, SharedMutex
	>::make_segment_and_bucket_prev(
		segment_and_bucket_t a
		, unsigned bkwd
	) const -> segment_and_bucket_t
	{
		a.subtract_small(bkwd);
		return a;
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<
		Key, T, Hash, Pred, Allocator, SharedMutex
	>::make_segment_and_bucket_for_iterator(
		bix_t ix_
	) const -> segment_and_bucket_t
	{
		assert( ix_ <= bucket_count() );

		auto si =
			__builtin_expect((segment_layout::ix_high(ix_) == 0),false)
			? 0
			: segment_layout::log2(ix_high(ix_))
			;
		auto bi =
			(
				__builtin_expect((segment_layout::ix_high(ix_) == 0),false)
				? 0
				: ix_high(ix_) % (bix_t(1U) << (si-1))
			)
			*
			segment_layout::base_segment_size + segment_layout::ix_low(ix_)
			;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
		return
			si == segment_count()
			? segment_and_bucket_t(&_bc[si-1], _bc[si-1].segment_size() ) /* end iterator */
			: segment_and_bucket_t(&_bc[si], bi)
			;
#pragma GCC diagnostic pop
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<
		Key, T, Hash, Pred, Allocator, SharedMutex
	>::make_segment_and_bucket_at_begin() const -> segment_and_bucket_t
	{
		return segment_and_bucket_t(&_bc[0], 0); /* begin iterator */
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<
		Key, T, Hash, Pred, Allocator, SharedMutex
	>::make_segment_and_bucket_at_end() const -> segment_and_bucket_t
	{
		auto ix = bucket_count();

		auto si =
			__builtin_expect((segment_layout::ix_high(ix) == 0),false)
			? 0
			: segment_layout::log2(ix_high(ix))
			;
		return segment_and_bucket_t(&_bc[si-1], _bc[si-1].segment_size() ); /* end iterator */
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<
		Key, T, Hash, Pred, Allocator, SharedMutex
	>::locate_bucket_mutexes(
		const segment_and_bucket_t &a_
	) const -> bucket_mutexes_t &
	{
		return _bc[a_.si()]._bucket_mutexes[a_.bi()];
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::locate_owner(
		const segment_and_bucket_t &a_
	) -> const owner &
	{
		return static_cast<const owner &>(a_.deref());
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<
		Key, T, Hash, Pred, Allocator, SharedMutex
	>::make_owner_unique_lock(
		const segment_and_bucket_t &a_
	) const -> owner_unique_lock_t
	{
		return owner_unique_lock_t(a_.deref(), a_, locate_bucket_mutexes(a_)._m_owner);
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<
		Key, T, Hash, Pred, Allocator, SharedMutex
	>::make_owner_unique_lock(
		const content_unique_lock_t &cl_
		, unsigned bkwd
	) const -> owner_unique_lock_t
	{
		auto a = cl_.sb();
		a.subtract_small(bkwd);
		return owner_unique_lock_t(a.deref(), a, locate_bucket_mutexes(a)._m_owner);
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	template < typename K >
		auto impl::hop_hash_base<
			Key, T, Hash, Pred, Allocator, SharedMutex
		>::make_owner_shared_lock(
			const K &k_
		) const -> owner_shared_lock_t
		{
			return make_owner_shared_lock(make_segment_and_bucket(bucket(k_)));
		}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<
		Key, T, Hash, Pred, Allocator, SharedMutex
	>::make_owner_shared_lock(
		const segment_and_bucket_t &a_
	) const -> owner_shared_lock_t
	{
		return owner_shared_lock_t(a_.deref(), a_, locate_bucket_mutexes(a_)._m_owner);
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<
		Key, T, Hash, Pred, Allocator, SharedMutex
	>::make_content_unique_lock(
		const segment_and_bucket_t &a_
	) const -> content_unique_lock_t
	{
		return
			content_unique_lock_t(
				a_.deref()
				, a_
				, locate_bucket_mutexes(a_)._m_content
			);
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<
		Key, T, Hash, Pred, Allocator, SharedMutex
	>::make_content_unique_lock(
		const owner_unique_lock_t &wl_
		, unsigned fwd
	) const -> content_unique_lock_t
	{
		auto a = wl_.sb();
		a.add_small(fwd);
		return make_content_unique_lock(a);
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	template <typename K>
		auto impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::find(
			TM_ACTUAL
			const K &k_
		) -> iterator
		{
			TM_SCOPE()
			auto bi_lk = make_owner_shared_lock(k_);
			auto key_bi = locate_key(TM_REF bi_lk, k_);
			auto content_ix = distance_small(bi_lk.sb(), key_bi);
			return content_ix == owner::size ? end() : iterator{bi_lk.sb(), content_ix};
		}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	template <typename K>
		auto impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::find(
			TM_ACTUAL
			const K &k_
		) const -> const_iterator
		{
			TM_SCOPE()
			auto bi_lk = make_owner_shared_lock(k_);
			auto key_bi = locate_key(TM_REF bi_lk, k_);
			return distance_small(bi_lk.si(), key_bi.si()) == owner::size ? end() : key_bi;
		}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::erase(
		iterator it_
	) -> iterator
	try
	{
		consistency_guard<impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>> g(this);
		/* The bucket which owns the entry */
		auto owner_lk = make_owner_unique_lock(it_.sb_owner());

		auto erase_src_lk = make_content_unique_lock(it_.sb_content());
		/* 4-step owner erase:
		 *
		 * 1. mark size unstable
		 *  persist
		 * 2. disclaim owner ownership atomically
		 *  persist
		 * 3. mark content FREE (in erase)
		 *  persist
		 * 4. mark size stable
		 *  persist
		 */

		this->persist_map_controller_t::persist_content(erase_src_lk.ref(), "content erase exiting");
		{
			persist_size_change<Allocator, size_decr> s(*this);
			owner_lk.ref().erase(
				static_cast<unsigned>(erase_src_lk.index()-owner_lk.index())
				, owner_lk
				, gsl::not_null<persist_map_controller_t *>(this)
			);
			this->persist_map_controller_t::persist_owner(owner_lk.ref(), "owner erase");
			erase_src_lk.ref().content_erase();
			/* write a "FREE" mark for the content */
			erase_src_lk.owner_ref().set_adjacent_content_in_use(false);
		}
		/* persist_size_change may have failed to due to perishable counter, but the exception
		 * could not be propagated as an exception because it happened in a destructor.
		 * Throw the exception here.
		 */
		perishable::test();
		return ++it_;
	}
	catch ( const perishable_expiry & )
	{
		hop_hash_log<trace_perishable_expiry>::write(LOG_LOCATION, "perishable expiry dump (erase)\n"
			, dump<trace_perishable_expiry>::make_hop_hash_dump(*this));
		throw;
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	template <typename K>
		auto impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::erase(
			TM_ACTUAL
			const K &k_
		) -> size_type
		try
		{
			consistency_guard<impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>> g(this);
			auto it = find(TM_REF k_);
			return
				it == end()
				? 0U
				: (erase(it), 1U)
				;
		}
		catch ( const perishable_expiry & )
		{
			hop_hash_log<trace_perishable_expiry>::write(LOG_LOCATION, "perishable expiry dump (erase)\n"
				, dump<trace_perishable_expiry>::make_hop_hash_dump(*this));
			throw;
		}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	template < typename K >
		auto impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::count(
			TM_ACTUAL
			const K &k_
		) const -> size_type
		{
			auto bi_lk = make_owner_shared_lock(k_);
			auto key_bi = locate_key(TM_REF bi_lk, k_);
			return distance_small(bi_lk.sb(), key_bi.sb()) == owner::size ? 0U : 1U;
		}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	template <typename K>
		auto impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::at(
			TM_ACTUAL
			const K &k_
		) const -> const mapped_type &
		{
			TM_SCOPE()
			/* The bucket which owns the entry */
			auto bi_lk = make_owner_shared_lock(k_);
			const auto key_bi = locate_key(TM_REF bi_lk, k_);
			if ( distance_small(bi_lk.sb(), key_bi) == owner::size )
			{
				/* no such element */
				throw impl::key_not_found{};
			}
			/* element found at bf */
			return static_cast<bucket_t &>(key_bi.deref()).mapped();
		}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	template < typename K >
		auto impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::at(
			TM_ACTUAL
			const K &k_
		) -> mapped_type &
		{
			TM_SCOPE()
			/* Lock the entry owner */
			auto bi_lk = make_owner_shared_lock(k_);
			const auto key_bi = locate_key(TM_REF bi_lk, k_);
			if ( distance_small(bi_lk.sb(), key_bi) == owner::size )
			{
				/* no such element */
				throw impl::key_not_found{};
			}
			/* element found at bf */
			return static_cast<bucket_t &>(key_bi.deref()).mapped();
		}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	template < typename K >
		auto impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::bucket(
			const K &k_
		) const -> size_type
		{
			return bucket_ix(_hasher.hf(k_));
		}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::bucket_size(
		size_type n_
	) const -> size_type
	{
		auto a = make_segment_and_bucket(n_);
		auto g = owner_shared_lock_t(a.deref(), a, locate_bucket_mutexes(a)._m_owner);
		size_type s = 0;
		for ( auto v = g.ref().get_value(g); v; v >>= 1U )
		{
			s += ( v && 1u );
		}
		return s;
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::size(
	) const -> size_type
	{
		consistency_guard<impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>> g(this);
		return this->persist_map_controller_t::size();
	}

template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<
		Key, T, Hash, Pred, Allocator, SharedMutex
	>::finish_owner_mask(
		owner::value_type owner_mask_
		, segment_and_bucket_t sb_
	) const -> owner::value_type
	{
		owner_mask_ >>= 1U;
		/*
		 * The "used" flag of a bucket is held by at most one owner.
		 * Therefore, no corresponding bits shall be 1 in both owner_mask_
		 * (the sum of previous owners) and _buckets[owner_lk.index()]._owner
		 * (the current owner to be included).
		 */

		auto owner_lk = make_owner_shared_lock(sb_);
		const auto v = locate_owner(sb_).ownership_bits(owner_lk);
		/* If the previous partial ownership mask and v have any bits in common,
		 * then both some previous slot and sb_ claim to own the content
		 * at the common bit. Two owners is an error.
		 */
		const auto disagree_mask = (owner_mask_ & v);
		if ( disagree_mask != 0 )
		{
			hop_hash_log<trace_perishable_expiry>::write(LOG_LOCATION, "ownership disagreement in range ["
				, bucket_ix(sb_.index()-owner::size), "..", sb_.index()
				, "]");
		}

		assert( trace_perishable_expiry || disagree_mask == 0 );

		/* returns:
		 *  - in the 0 bit, the "used" aspect of _buckets[sb_+1] : 1 if the bucket is marked owned by an owner field, else 0
		 *  - in the nth bit, the partially-developed "used" aspect of _buckets[sb_+1+n] : 1 if if the bucket is marked owned by an owner field at or preceding bi, else 0
		 */
		return owner_mask_ | v;
	}

/*
 * returns the partial ownership mask up to but not including bucket a_:
 *  - in its 0 bit, the partially-deveoped "used" aspect of _buckets[a_] : 1 if the bucket is marked owned by an owner field, else 0
 *    The "used" aspect is complete except for the ownership information kept in bucket a_ itself.
 *  - in its nth bit, the partially-developed "used" aspect of _buckets[a_+n] : 1 if if the bucket is marked owned by an owner field at or preceding bi, else 0
 *    The "used" aspects are complete except for the ownership information kept in buckets a_ .. a_+n inclusive.
 */
template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	auto impl::hop_hash_base<
		Key, T, Hash, Pred, Allocator, SharedMutex
	>::in_use_by_owner_pre_mask(
		const segment_and_bucket_t &a_
	) const -> owner::value_type
	{
		/* In order to develop a mask of "used" (non-free) locations,
		 * examine the members of every owner starting with the leftmost
		 * bucket which can include bi in its owner.
		 */
		owner::value_type c = 0U;
		for ( auto sbw = make_segment_and_bucket_prev(a_, owner::size-1)
			; sbw != a_
			; sbw.incr_with_wrap()
		)
		{
			auto owner_lk = make_owner_shared_lock(sbw);
			/*
			 * The "used" flag of a bucket is held by at most one owner.
			 * Therefore, no corresponding bits shall be 1 in both c
			 * (the sum of previous owners) and _buckets[owner_lk.index()]._owner
			 * (the current owner to be included).
			 */

			c >>= 1U;
			const auto v = locate_owner(sbw).ownership_bits(owner_lk);
			const auto disagree_mask = (c & v);
			if ( disagree_mask != 0 )
			{
				hop_hash_log<trace_perishable_expiry>::write(
					LOG_LOCATION, "ownership disagreement in range ["
					, bucket_ix(a_.index()-owner::size), "..", sbw.index()
					, "]"
				);
			}

			assert( trace_perishable_expiry || disagree_mask == 0 );

			c |= v;
		}
		/* c now contains:
		 *  - in its 0 bit, the "used" aspect of _buckets[bi_] : 1 if the bucket is marked owned by an owner field, else 0
		 *  - in its nth bit, the partially-developed "used" aspect of _buckets[bi_+n] : 1 if if the bucket is marked owned by an owner field at or preceding bi, else 0
		 */
		return c;
	}

/* check that ownership bits match content bits. */
template <
	typename Key, typename T, typename Hash, typename Pred
	, typename Allocator, typename SharedMutex
>
	void impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>::check_consistency() const
	{
		if ( 0 < _consistency_check)
		{
			std::ostringstream o;
			o << std::setbase(16) << std::this_thread::get_id();
			auto sb = make_segment_and_bucket(0U);
			const auto sb_end =
				make_segment_and_bucket_at_end();
			hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION
				, " bucket_count ", bucket_count()
				, " sb_end ", sb_end.si(), ",", sb_end.bi()
			);

			/* Start with the partially-developed ownership mask prededing position zero */
			owner::value_type ownership_mask = in_use_by_owner_pre_mask(sb);

			for (
				; sb != sb_end
				; sb.incr_without_wrap()
			)
			{
				ownership_mask = finish_owner_mask(ownership_mask, sb);
				if ( 1 < _consistency_check)
				{
					if ( ownership_mask )
					{
						PLOG("%s[%s]: %zu owner mask is %" PRIx64, __func__, o.str().c_str(), sb.index(), ownership_mask);
					}
					if ( ownership_mask & 1ULL )
					{
						PLOG("%s[%s]: %zu is owned", __func__, o.str().c_str(), sb.index());
					}
					if ( is_in_use(sb) )
					{
						PLOG("%s[%s]: %zu is occupied", __func__, o.str().c_str(), sb.index());
					}
				}
				bool in_use = is_in_use(sb);
				bool owned = ownership_mask & 1ULL;
				if ( owned != in_use )
				{
					PLOG("%s[%s]: bucket %zu (seg %zu (%p) offset %zu) ownership does not match occupancy: %s %s"
						, __func__
						, o.str().c_str()
						, sb.index()
						, sb.si()
						, sb.sp()
						, sb.bi()
						, (owned ? "owned" : "unowned")
						, (in_use ? "in_use" : "free")
					);
					/* Panic: ownership mask does not agree with content "in use" status */
					abort();
				}
			}
		}
	}
