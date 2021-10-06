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

#ifndef _MCAS_HSTORE_HOP_HASH_ITERATORS_H
#define _MCAS_HSTORE_HOP_HASH_ITERATORS_H

#include "owner.h"
#include <iterator> /* itereator, forward_iterator_tag */

namespace impl
{
	template <typename HopHash>
		struct hop_hash_iterator;

	template <typename HopHash>
		struct hop_hash_const_iterator;

	template <typename HopHash>
		struct hop_hash_local_iterator;

	template <typename HopHash>
		struct hop_hash_const_local_iterator;

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
		struct hop_hash_base;
}

namespace impl
{
	template <typename HopHash>
		bool operator!=(
			const hop_hash_local_iterator_impl<HopHash> a
			, const hop_hash_local_iterator_impl<HopHash> b
		);

	template <typename HopHash>
		bool operator==(
			const hop_hash_local_iterator_impl<HopHash> a
			, const hop_hash_local_iterator_impl<HopHash> b
		);

	template <typename HopHash>
		bool operator!=(
			const hop_hash_iterator_impl<HopHash> a
			, const hop_hash_iterator_impl<HopHash> b
		);

	template <typename HopHash>
		bool operator==(
			const hop_hash_iterator_impl<HopHash> a
			, const hop_hash_iterator_impl<HopHash> b
		);

	template <typename HopHash>
		struct hop_hash_local_iterator_impl
			: public std::iterator
				<
					std::forward_iterator_tag
					, typename HopHash::value_type
				>
		{
		private:
			using segment_and_bucket_t = typename HopHash::segment_and_bucket_t;
			segment_and_bucket_t _sb_owner;
			owner::index_type _content_index;
			void advance_to_in_use()
			{
				while ( _content_index != owner::size && ! _sb_owner.deref().is_in_use(_content_index) )
				{
					++_content_index;
				}
				if ( _content_index == owner::size )
				{
					_content_index = 0;
					_sb_owner.incr_with_wrap();
				}
			}
			segment_and_bucket_t sb_content() const
			{
				segment_and_bucket_t sbc(_sb_owner);
				sbc.add_small(_content_index);
				return sbc;
			}

		protected:
			using base =
				std::iterator<std::forward_iterator_tag, typename HopHash::value_type>;

			/* HopHash 106 (Iterator) */
			auto deref() const -> typename base::reference
			{
				return sb_content().deref().content<typename HopHash::value_type>::value();
			}
			void incr()
			{
				++_content_index;
				advance_to_in_use();
			}
		public:
			/* HopHash 17 (EqualityComparable) */
			friend
				bool operator== <>(
					hop_hash_local_iterator_impl<HopHash> a
					, hop_hash_local_iterator_impl<HopHash> b
				);
			/* HopHash 107 (InputIterator) */
			friend
				bool operator!= <>(
					hop_hash_local_iterator_impl<HopHash> a
					, hop_hash_local_iterator_impl<HopHash> b
				);
			/* HopHash 109 (ForwardIterator) - handled by 107 */
		public:
			hop_hash_local_iterator_impl(const segment_and_bucket_t &sb_owner_)
				: _sb_owner(sb_owner_)
				, _content_index(0)
			{
				advance_to_in_use();
			}
		};

	template <typename HopHash>
		struct hop_hash_iterator_impl
			: public std::iterator
				<
					std::forward_iterator_tag
					, typename HopHash::value_type
				>
		{
		private:
			using segment_and_bucket_t = typename HopHash::segment_and_bucket_t;
			/* ERROR: iterator runs through non-empty content.
			 * It should run through owners, indexing the content within each owner
			 * that is necessary to allow erase(), which needs both to locate both owner bucket
			 * and content bucket, to take an iterator.
			 */
			segment_and_bucket_t _sb_owner;
			unsigned _content_index;
			/* The "end" value is the past the last _sb_owner, that is (last segment, last bucket+1) */
			void advance_to_in_use()
			{
				while ( ! _sb_owner.at_end() && ! _sb_owner.deref().is_in_use(_content_index) )
				{
					++_content_index;
					if ( _content_index == owner::size )
					{
						_content_index = 0;
						_sb_owner.incr_without_wrap();
					}
				}
			}

		protected:
			using base =
				std::iterator<std::forward_iterator_tag, typename HopHash::value_type>;

			segment_and_bucket_t sb_content() const
			{
				segment_and_bucket_t sbc(_sb_owner);
				sbc.add_small(_content_index);
				return sbc;
			}

			/* HopHash 106 (Iterator) */
			auto deref() const -> typename base::reference
			{
				return sb_content().deref().content<typename HopHash::value_type>::value();
			}
			void incr()
			{
				++_content_index;
				if ( _content_index == owner::size )
				{
					_content_index = 0;
					_sb_owner.incr_without_wrap();
				}
				advance_to_in_use();
			}
		public:
			/* HopHash 17 (EqualityComparable) */
			friend
				bool operator== <>(
					hop_hash_iterator_impl<HopHash> a
					, hop_hash_iterator_impl<HopHash> b
				);
			/* HopHash 107 (InputIterator) */
			friend
				bool operator!= <>(
					hop_hash_iterator_impl<HopHash> a
					, hop_hash_iterator_impl<HopHash> b
				);
			/* HopHash 109 (ForwardIterator) - handled by 107 */
		public:
			hop_hash_iterator_impl(const segment_and_bucket_t &sb_owner_, const owner::index_type ix_)
				: _sb_owner(sb_owner_)
				, _content_index(ix_)
			{
				advance_to_in_use();
			}
			friend struct impl::hop_hash_iterator<HopHash>;
		};

	template <typename HopHash>
		struct hop_hash_local_iterator
			: public hop_hash_local_iterator_impl<HopHash>
		{
		private:
			using segment_and_bucket_t = typename HopHash::segment_and_bucket_t;
			using typename hop_hash_local_iterator_impl<HopHash>::base;
		public:
			hop_hash_local_iterator(
				const HopHash & // t_
				, const segment_and_bucket_t & sb_
				, owner::value_type mask_
			)
				: hop_hash_local_iterator_impl<HopHash>(sb_, mask_)
			{}
			/* HopHash 106 (Iterator) */
			auto operator*() const -> typename base::reference
			{
				return this->deref();
			}
			auto operator++() -> hop_hash_local_iterator &
			{
				this->incr();
				return *this;
			}
			/* HopHash 17 (EqualityComparable) */
			/* HopHash 107 (InputIterator) */
			auto operator->() const -> typename base::pointer
			{
				return &this->deref();
			}
			auto operator++(int) -> hop_hash_local_iterator
			{
				auto ti = *this;
				this->incr();
				return ti;
			}
			/* HopHash 109 (ForwardIterator) - handled by 107 */
		};

	template <typename HopHash>
		struct hop_hash_const_local_iterator
			: public hop_hash_local_iterator_impl<HopHash>
		{
		private:
			using segment_and_bucket_t = typename HopHash::segment_and_bucket_t;
			using typename hop_hash_local_iterator_impl<HopHash>::base;
		public:
			hop_hash_const_local_iterator(const segment_and_bucket_t & sb_, owner::value_type mask_)
				: hop_hash_local_iterator_impl<HopHash>(sb_, mask_)
			{}
			/* HopHash 106 (Iterator) */
			auto operator*() const -> const typename base::reference
			{
				return this->deref();
			}
			auto operator++() -> hop_hash_const_local_iterator &
			{
				this->incr();
				return *this;
			}
			/* HopHash 17 (EqualityComparable) */
			/* HopHash 107 (InputIterator) */
			auto operator->() const -> const typename base::pointer
			{
				return &this->deref();
			}
			auto operator++(int) -> hop_hash_const_local_iterator
			{
				auto ti = *this;
				this->incr();
				return ti;
			}
			/* HopHash 109 (ForwardIterator) - handled by 107 */
		};

	template <typename HopHash>
		struct hop_hash_iterator
			: public hop_hash_iterator_impl<HopHash>
		{
		private:
			using segment_and_bucket_t = typename HopHash::segment_and_bucket_t;
			using typename hop_hash_iterator_impl<HopHash>::base;
			const auto &sb_owner() const { return this->_sb_owner; }
			using hop_hash_iterator_impl<HopHash>::sb_content;
			auto content_index() const { return this->_content_index; }
		public:
			hop_hash_iterator(const segment_and_bucket_t & sb_, owner::index_type ix_)
				: hop_hash_iterator_impl<HopHash>(sb_, ix_)
			{}
			/* HopHash 106 (Iterator) */
			auto operator*() const -> typename base::reference
			{
				return this->deref();
			}
			auto operator++() -> hop_hash_iterator &
			{
				this->incr();
				return *this;
			}
			/* HopHash 17 (EqualityComparable) */
			/* HopHash 107 (InputIterator) */
			auto operator->() const -> typename base::pointer
			{
				return &this->deref();
			}
			hop_hash_iterator operator++(int)
			{
				auto ti = *this;
				this->incr();
				return ti;
			}
			/* HopHash 109 (ForwardIterator) - handled by 107 */
			template <
				typename Key
				, typename T
				, typename Hash
				, typename Pred
				, typename Allocator
				, typename SharedMutex
			>
				friend struct hop_hash_base;
		};

	template <typename HopHash>
		struct hop_hash_const_iterator
			: public hop_hash_iterator_impl<HopHash>
		{
		private:
			using segment_and_bucket_t = typename HopHash::segment_and_bucket_t;
			using typename hop_hash_iterator_impl<HopHash>::base;
		public:
			hop_hash_const_iterator(const segment_and_bucket_t & sb_, owner::index_type ix_)
				: hop_hash_iterator_impl<HopHash>(sb_, ix_)
			{}
			hop_hash_const_iterator(typename HopHash::size_type i)
				: hop_hash_iterator_impl<HopHash>(i)
			{}
			/* HopHash 106 (Iterator) */
			auto operator*() const -> const typename base::reference
			{
				return this->deref();
			}
			auto operator++() -> hop_hash_const_iterator &
			{
				this->incr();
				return *this;
			}
			/* HopHash 17 (EqualityComparable) */
			/* HopHash 107 (InputIterator) */
			auto operator->() const -> const typename base::pointer
			{
				return &this->deref();
			}
			hop_hash_const_iterator operator++(int)
			{
				auto ti = *this;
				this->incr();
				return ti;
			}
			/* HopHash 109 (ForwardIterator) - handled by 107 */
		};

	template <typename HopHash>
		bool operator==(
			const hop_hash_local_iterator_impl<HopHash> a
			, const hop_hash_local_iterator_impl<HopHash> b
		)
		{
			return a._sb_owner == b._sb_owner && a._content_index == b._content_index;
		}

	template <typename HopHash>
		bool operator!=(
			const hop_hash_local_iterator_impl<HopHash> a
			, const hop_hash_local_iterator_impl<HopHash> b
		)
		{
			return !(a==b);
		}

	template <typename HopHash>
		bool operator==(
			const hop_hash_iterator_impl<HopHash> a
			, const hop_hash_iterator_impl<HopHash> b
		)
		{
			return a._sb_owner == b._sb_owner && a._content_index == b._content_index;
		}

	template <typename HopHash>
		bool operator!=(
			const hop_hash_iterator_impl<HopHash> a
			, const hop_hash_iterator_impl<HopHash> b
		)
		{
			return !(a==b);
		}
}

#endif
