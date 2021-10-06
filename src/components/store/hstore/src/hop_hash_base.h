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

#ifndef _MCAS_HSTORE_HOP_HASH_BASE_H
#define _MCAS_HSTORE_HOP_HASH_BASE_H

#include "hop_hash_allocator.h"
#include "segment_layout.h"
#include "persist_map_controller.h"

#include "alloc_key.h" /* AK_FORMAL */
#include "bucket_control.h"
#include "bucket_shared_lock.h"
#include "bucket_unique_lock.h"
#include "construction_mode.h"
#include "hop_hash_iterators.h"
#include "trace_flags.h"
#include <common/string_view.h>

#include <cstddef> /* size_t */
#include <memory> /* allocator_traits */
#include <tuple>
#include <utility> /* pair */

/* Inteded to implement Hopscotch hashing
 * http://mcg.cs.tau.ac.il/papers/disc2008-hopscotch.pdf
 */

#include "hop_hash_debug.h"

#if TRACED_TABLE
template <
	typename Key
	, typename T
	, typename Hash
	, typename Pred
	, typename Allocator
	, typename SharedMutex
>
	struct hop_hash;
#endif

namespace impl
{
	template <typename Value>
		struct hash_bucket;

	template <typename Bucket>
		struct segment_and_bucket;

	template <typename HopHash>
		struct hop_hash_iterator;

	template <typename HopHash>
		struct hop_hash_const_iterator;

	template <typename HopHash>
		struct hop_hash_local_iterator;

	template <typename HopHash>
		struct hop_hash_const_local_iterator;

	template <typename Bucket, typename Referent, typename SharedMutex>
		struct bucket_unique_lock;

	template <typename Bucket, typename Referent, typename SharedMutex>
		struct bucket_shared_lock;

	template <typename Mutex>
		struct bucket_mutexes;

	template <typename HopHash>
		struct hop_hash_local_iterator_impl;

	template <typename HopHash>
		struct hop_hash_iterator_impl;

	template <
		typename Key
		, typename T
		, typename Hash
		, typename Pred
		, typename Allocator
		, typename SharedMutex
	>
		struct hop_hash_base
			: private hop_hash_allocator<Allocator>
			, private segment_layout
			, private
				persist_map_controller<
					typename std::allocator_traits<Allocator>::template rebind_alloc<
						std::pair<const Key, T>
					>
				>
		{
		private:
			using allocator_traits_type = std::allocator_traits<Allocator>;
		public:
#if HSTORE_TRACE_RESIZE
			using persist_map_controller<
				typename std::allocator_traits<Allocator>::template rebind_alloc
				<
					std::pair<const Key, T>
				>
			>::segment_count_actual;
#endif
			using key_type        = Key;
			using mapped_type     = T;
			using value_type      = std::pair<const key_type, mapped_type>;
			using persist_map_controller_t =
				persist_map_controller<
					typename allocator_traits_type::template rebind_alloc<value_type>
				>;
			using size_type       = typename persist_map_controller_t::size_type;
			using hasher          = Hash;
			using key_equal       = Pred;
			using allocator_type  = Allocator;
			using pointer         = typename allocator_traits_type::pointer;
			using const_pointer   = typename allocator_traits_type::const_pointer;
			using reference       = value_type &;
			using const_reference = const value_type &;
			using iterator        = hop_hash_iterator<hop_hash_base>;
			using const_iterator  = hop_hash_const_iterator<hop_hash_base>;
			using local_iterator  = hop_hash_local_iterator<hop_hash_base>;
			using const_local_iterator = hop_hash_const_local_iterator<hop_hash_base>;
			using persist_data_type =
				persist_map<typename allocator_traits_type::template rebind_alloc<value_type>>;
#if ! HSTORE_TRACE_RESIZE
		private:
#endif
			using bix_t = size_type; /* sufficient for all bucket indexes */
		private:
			using hash_result_t = typename hasher::result_type;
#if HSTORE_TRACE_RESIZE
		public:
#endif
			using bucket_t = hash_bucket<value_type>;
			using content_t = content<value_type>;
		private:
			using bucket_mutexes_t = bucket_mutexes<SharedMutex>;
			using bucket_control_t = bucket_control<bucket_t, SharedMutex>;
			using bucket_aligned_t = typename bucket_control_t::bucket_aligned_t;
#if ! TRACED_OWNER && ! TRACED_CONTENT
			static_assert(sizeof(bucket_aligned_t) <= 64, "Bucket size exceeds presumed cache line size (64)");
#endif
			using bucket_allocator_t =
				typename allocator_traits_type::template rebind_alloc<bucket_aligned_t>;
			using owner_unique_lock_t = bucket_unique_lock<bucket_t, owner, SharedMutex>;
			using owner_shared_lock_t = bucket_shared_lock<bucket_t, owner, SharedMutex>;
			using content_unique_lock_t = bucket_unique_lock<bucket_t, content_t, SharedMutex>;
			using content_shared_lock_t = bucket_shared_lock<bucket_t, content_t, SharedMutex>;
#if HSTORE_TRACE_RESIZE
		public:
#endif
			using segment_and_bucket_t = segment_and_bucket<bucket_t>;
			void ownership_check() const;
		private:
			static constexpr auto _segment_capacity =
				persist_map_controller_t::_segment_capacity;
			/* Need to adjust hash and bucket_ix interpretations in more places
			* before this can becom non-zero
			*/
			static constexpr unsigned log2_base_segment_size =
				persist_map_controller_t::log2_base_segment_size;
			static constexpr bix_t base_segment_size =
				persist_map_controller_t::base_segment_size;
			static_assert(
				owner::size < base_segment_size
				, "Base segment size must exceeed owner size"
			);
			using six_t = std::size_t; /* segment indexes (but uint8_t would do) */

			/* The number of non-null elements of _b may exceed _count
			* only if a crash occurred between the assignemnt of a non-null
			* value to _b[_count] and the increment of _count.
			* In that case the non-null value at _count points to lost memory.
			* That memory can be recovered the next time _b is extended.
			*/
			hasher _hasher;

			bool _auto_resize;

			bucket_control_t _bc[_segment_capacity];

			six_t segment_count() const override
			{
				return persist_map_controller_t::segment_count_actual().value();
			}

			six_t segment_count_not_stable() const
			{
				return persist_map_controller_t::segment_count_actual().value_not_stable();
			}

			auto bucket_ix(const hash_result_t h) const -> bix_t;
			auto bucket_expanded_ix(const hash_result_t h) const -> bix_t;

			auto nearest_free_bucket(segment_and_bucket_t bi) -> content_unique_lock_t;

			auto make_space_for_insert(
				bix_t bi
				, content_unique_lock_t bf
			) -> content_unique_lock_t;

			/*
			 * Note in use:
			 * locate key is called with two flavors of arguments:
			 *   For emplace: Lock is owner_unique_lock and K is persist_fixed_string
			 *   For other uses (find, count, at): Lock is owner_shared_lock and K is std::string
			 */
			template <typename Lock, typename K>
				auto locate_key(
					TM_FORMAL
					Lock &bi
					, const K &k
				) const -> segment_and_bucket_t;

			auto locate_key_inner(
				TM_FORMAL
				owner::value_type ownership_bits_
				, segment_and_bucket_t sb_
				, common::string_view k_
			) const -> segment_and_bucket_t;

			void resize(AK_FORMAL0);
			void resize_pass1();
			void resize_pass2();
			bool resize_pass2_adjust_owner(
				bix_t ix_senior
				, bucket_control_t &junior_bucket_control
				, content_unique_lock_t &populated_content_lk
			);
			auto locate_bucket_mutexes(
				const segment_and_bucket_t &
			) const -> bucket_mutexes_t &;

			auto make_owner_unique_lock(
				const segment_and_bucket_t &a
			) const -> owner_unique_lock_t;

			/* lock an owner which precedes content */
			auto make_owner_unique_lock(
				const content_unique_lock_t &
				, unsigned bkwd
			) const -> owner_unique_lock_t;

			template <typename K>
				auto make_owner_shared_lock(const K &k) const -> owner_shared_lock_t;
			auto make_owner_shared_lock(
				const segment_and_bucket_t &
			) const -> owner_shared_lock_t;

			auto make_content_unique_lock(
				const segment_and_bucket_t &
			) const -> content_unique_lock_t;
			/* lock content which follows an owner */
			auto make_content_unique_lock(
				const owner_unique_lock_t &
				, unsigned fwd
			) const -> content_unique_lock_t;

			using persist_map_controller_t::mask;

			auto owner_value_at(owner_unique_lock_t &bi) const -> owner::value_type;
			auto owner_value_at(owner_shared_lock_t &bi) const -> owner::value_type;

			auto make_segment_and_bucket(bix_t ix) const -> segment_and_bucket_t;
			auto make_segment_and_bucket_unsafe(bix_t ix) const -> segment_and_bucket_t;
			auto make_segment_and_bucket_for_iterator(
				bix_t ix
			) const -> segment_and_bucket_t;
			auto make_segment_and_bucket_at_begin() const -> segment_and_bucket_t;
			auto make_segment_and_bucket_at_end() const -> segment_and_bucket_t;
			auto make_segment_and_bucket_prev(
				segment_and_bucket_t a
				, unsigned bkwd
			) const -> segment_and_bucket_t;

			static auto locate_owner(const segment_and_bucket_t &a) -> const owner &;

			template <typename K>
				auto bucket(const K &) const -> size_type;
			auto bucket_size(const size_type n) const -> size_type;

			auto in_use_by_owner_mask(const segment_and_bucket_t &a) const -> owner::value_type;
			bool is_in_use(const segment_and_bucket_t &a) const;

			/* computed distance from first to last, accounting for the possibility that
			 * last is smaller than first due to wrapping.
			 */
			auto distance_wrapped(bix_t first, bix_t last) const -> unsigned;

			/* begin an owner mask (all owners prededing sb) */
			auto in_use_by_owner_pre_mask(const segment_and_bucket_t &sb) const -> owner::value_type;
			/* finish the owner mask (or in ownership by sb) */
			auto finish_owner_mask(owner::value_type owner_mask, segment_and_bucket_t sb) const -> owner::value_type;

			int _consistency_check;

		public:
			explicit hop_hash_base(
				AK_FORMAL
				persist_data_type *pc
				, construction_mode mode
				, const Allocator &av = Allocator()
			);
			hop_hash_base(const hop_hash_base &) = delete;
			hop_hash_base(hop_hash_base &&) noexcept = default;
		protected:
			virtual ~hop_hash_base();
		public:
			hop_hash_base &operator=(const hop_hash_base &) = delete;
			void check_consistency() const;
			allocator_type get_allocator() const noexcept
			{
				return static_cast<const hop_hash_allocator<Allocator> &>(*this);
			}

			template <typename ... Args>
				auto emplace(
					AK_FORMAL
					TM_FORMAL
					Args && ... args
				) -> std::pair<iterator, bool>;
			auto insert(
					TM_FORMAL
					const value_type &value
			) -> std::pair<iterator, bool>;

			template <typename K>
				auto erase(
					TM_FORMAL
					const K &key
				) -> size_type;

			auto erase(iterator it) -> iterator;

			template <typename K>
				auto find(
					TM_FORMAL
					const K &key
				) -> iterator;
			template <typename K>
				auto find(
					TM_FORMAL
					const K &key
				) const -> const_iterator;

			template <typename K>
				auto at(
					TM_FORMAL
					const K &key
				) -> mapped_type &;
			template <typename K>
				auto at(
					TM_FORMAL
					const K &key
				) const -> const mapped_type &;

			template <typename K>
				auto count(
					TM_FORMAL
					const K &k
				) const -> size_type;
			auto begin() -> iterator
			{
				return iterator(make_segment_and_bucket_at_begin(), 0U);
			}
			auto end() -> iterator
			{
				return iterator(make_segment_and_bucket_at_end(), 0U);
			}
			auto begin() const -> const_iterator
			{
				return cbegin();
			}
			auto end() const -> const_iterator
			{
				return cend();
			}
			auto cbegin() const -> const_iterator
			{
				return const_iterator(make_segment_and_bucket_at_begin(), 0U);
			}
			auto cend() const -> const_iterator
			{
				return const_iterator(make_segment_and_bucket_at_end(), 0U);
			}

			using persist_map_controller_t::bucket_count;
			using persist_map_controller_t::max_bucket_count;

			auto begin(size_type n) -> local_iterator
			{
				auto sb = make_segment_and_bucket_for_iterator(n);
				auto owner_lk = make_owner_shared_lock(sb);
				return local_iterator(*this, sb, locate_owner(sb).value(owner_lk));
			}
			auto end(size_type n) -> local_iterator
			{
				auto sb = make_segment_and_bucket_for_iterator(n);
				return local_iterator(*this, sb, owner::value_type(0));
			}
			auto begin(size_type n) const -> const_local_iterator
			{
				return cbegin(n);
			}
			auto end(size_type n) const -> const_local_iterator
			{
				return cend(n);
			}
			auto cbegin(size_type n) const -> const_local_iterator
			{
				auto sb = make_segment_and_bucket_for_iterator(n);
				auto owner_lk = make_owner_shared_lock(sb);
				return const_local_iterator(sb, locate_owner(sb).value(owner_lk));
			}
			auto cend(size_type n) const -> const_local_iterator
			{
				auto sb = make_segment_and_bucket_for_iterator(n);
				return const_local_iterator(sb, owner::value_type(0));
			}
			auto size() const -> size_type;

			bool set_auto_resize(bool v1) { auto v0 = _auto_resize; _auto_resize = v1; return v0; }
			bool get_auto_resize() const { return _auto_resize; }

#if TRACED_TABLE
			friend
				auto operator<< <>(
					std::ostream &
					, const impl::hop_hash_print<
						hop_hash<Key, T, Hash, Pred, Allocator, SharedMutex>
					> &
				) -> std::ostream &;

			friend
				auto operator<< <>(
					std::ostream &
					, const impl::dump<true>::hop_hash_dump<hop_hash_base> &
				) -> std::ostream &;
#endif
#if TRACED_OWNER
			template <typename Lock>
				friend
					auto operator<<(
						std::ostream &o_
						, const impl::owner_print<Lock> &
					) -> std::ostream &;

			template <typename HopHashBase>
				friend
					auto operator<<(
						std::ostream &o_
						, const owner_print<
							bypass_lock<typename HopHashBase::bucket_t, const owner>
						> &
					) -> std::ostream &;
#endif
#if TRACED_BUCKET
			template<
				typename HopHashBase
				, typename LockOwner
				, typename LockContent
			>
				friend
					auto operator<<(
						std::ostream &o_
						, const impl::bucket_print<
							LockOwner
							, LockContent
						> &
					) -> std::ostream &;

			template <typename HopHash>
				friend
				auto operator<<(
					std::ostream &o_
					, const bucket_print
					<
						bypass_lock<typename HopHash::bucket_t, const owner>
						, bypass_lock<
							typename HopHash::bucket_t
							, const content<typename HopHash::value_type>
						>
					> &
				) -> std::ostream &;
#endif
			template <typename HopHash>
				friend struct impl::hop_hash_local_iterator_impl;
			template <typename HopHash>
				friend struct impl::hop_hash_iterator_impl;
			template <typename HopHash>
				friend struct impl::hop_hash_local_iterator;
			template <typename HopHash>
				friend struct impl::hop_hash_const_local_iterator;
			template <typename HopHash>
				friend struct impl::hop_hash_iterator;
			template <typename HopHash>
				friend struct impl::hop_hash_const_iterator;
		};
}

#if TRACK_OWNER
#include "owner.tcc"
#endif
#include "hop_hash_base.tcc"

#if TRACED_CONTENT || TRACED_OWNER || TRACED_BUCKET
#include "hop_hash_debug.tcc"
#endif

#endif
