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


#ifndef _MCAS_HSTORE_HOP_HASH_H
#define _MCAS_HSTORE_HOP_HASH_H

#include "hop_hash_base.h"

#include "alloc_key.h" /* AK_ACTUAL */
#include "construction_mode.h"

#include <common/perf/tm_fwd.h>

#include <cstddef> /* size_t */
#include <functional> /* equal_to */
#include <new> /* allocator */
#include <shared_mutex> /* shared_timed_mutex */
#include <utility> /* hash, pair */

/* Inteded to implement Hopscotch hashing
 * http://mcg.cs.tau.ac.il/papers/disc2008-hopscotch.pdf
 */

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
		struct hop_hash_iterator_impl;

	template <typename HopHash>
		struct hop_hash_local_iterator_impl;
}

template <
	typename Key
	, typename T
	, typename Hash = std::hash<Key>
	, typename Pred = std::equal_to<Key>
	, typename Allocator = std::allocator<std::pair<const Key, T>>
	, typename SharedMutex = std::shared_timed_mutex
>
	struct hop_hash
		: private impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>
	{
		using base = impl::hop_hash_base<Key, T, Hash, Pred, Allocator, SharedMutex>;
		using size_type      = std::size_t;
		using key_type       = Key;
		using mapped_type    = T;
		using value_type     = std::pair<const key_type, mapped_type>;
		/* base is private. Can we use it to specify public types? Yes. */
		using iterator       = impl::hop_hash_iterator<base>;
		using const_iterator = impl::hop_hash_const_iterator<base>;
		using typename base::persist_data_type;
		using typename base::allocator_type;

		/* contruct/destroy/copy */
		explicit hop_hash(
			AK_ACTUAL
			persist_data_type *pc_
			, construction_mode mode_
			, const Allocator &av_ = Allocator()
		)
			: base(AK_REF pc_, mode_, av_)
		{}
#if 0
		hop_hash(hop_hash &&) noexcept = default;
#endif
		using base::get_allocator;

		/* size and capacity */
		auto empty() const noexcept -> bool
		{
			return size() == 0;
		}
		using base::size;
		using base::get_auto_resize;
		using base::set_auto_resize;
		using base::bucket_count;
		auto max_size() const noexcept -> size_type
		{
			return (size_type(1U) << (base::_segment_capacity-1U));
		}
		/* iterators */

		using base::begin;
		using base::end;
		using base::cbegin;
		using base::cend;

		/* modifiers */

		using base::emplace;
		using base::insert;
		using base::erase;

		/* counting */

		using base::count;

		/* lookup */
		using base::find;
		using base::at;

		template <typename HopHash>
			friend struct impl::hop_hash_local_iterator_impl;

		template <typename HopHash>
			friend struct impl::hop_hash_iterator_impl;
	};

#endif
