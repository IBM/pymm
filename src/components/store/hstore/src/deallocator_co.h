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


#ifndef MCAS_HSTORE_DEALLOCATOR_CO_H
#define MCAS_HSTORE_DEALLOCATOR_CO_H

#include "heap_co.h"
#include "hop_hash_log.h"
#include "pointer_pobj.h"
#include "store_root.h"
#include "trace_flags.h"

#pragma GCC diagnostic push
#if defined __clang__
#pragma GCC diagnostic ignored "-Wnested-anon-types"
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <libpmemobj.h> /* pmemobj_free */
#pragma GCC diagnostic pop

#include <cstdlib> /* size_t, ptrdiff_t */

template <typename T, typename Persister>
	struct deallocator_co;

template <typename Persister>
	struct deallocator_co<void, Persister>
		: public Persister
	{
	public:
		using value_type = void;
		using difference_type = std::ptrdiff_t;
		using pointer = pointer_pobj<void, 0U>;
		using const_pointer = pointer_pobj<const void, 0U>;
	};

template <typename T, typename Persister>
	struct deallocator_co
		: public Persister
	{
	public:
		using value_type = T;
		using size_type = std::size_t;
		using difference_type = std::ptrdiff_t;
		using pointer = pointer_pobj<T, 0U>;
		using const_pointer = pointer_pobj<const T, 0U>;

		deallocator_co(const Persister = Persister()) noexcept
		{}

		deallocator_co(const deallocator_co &) noexcept = default;

		template <typename U>
			deallocator_co(const deallocator_co<U, Persister> &) noexcept
				: deallocator_co()
			{}

		deallocator_co &operator=(const deallocator_co &) = delete;

		void deallocate(
			pointer ptr
			, size_type sz_
		)
		{
			auto pool = ::pmemobj_pool_by_oid(ptr);

			TOID_DECLARE_ROOT(struct store_root_t);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
			TOID(struct store_root_t) root = POBJ_ROOT(pool, struct store_root_t);
			assert(!TOID_IS_NULL(root));
#pragma GCC diagnostic ignored "-Wpedantic"
			auto heap = static_cast<heap_co *>(pmemobj_direct((D_RO(root)->heap_oid)));
#pragma GCC diagnostic pop
#if HSTORE_TRACE_PALLOC
			{
				auto p = static_cast<char *>(pmemobj_direct(ptr));
				hop_hash_log<HSTORE_TRACE_PALLOC>::write(LOG_LOCATION
					, "[", p
					, "..", common::p_fmt(p + sz_ * sizeof(T))
					, ")"
				);
			}
#endif
			heap->free(ptr, sizeof(T) * sz_);
		}
		auto max_size() const
		{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
			return PMEMOBJ_MAX_ALLOC_SIZE;
#pragma GCC diagnostic pop
		}

		void persist(const void *ptr, size_type len, const char * = nullptr)
		{
			Persister::persist(ptr, len);
		}
	};

#endif
