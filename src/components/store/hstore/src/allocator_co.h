/*
   Copyright [2017-2019] [IBM Corporation]
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


#ifndef MCAS_HSTORE_ALLOCATOR_CO_H
#define MCAS_HSTORE_ALLOCATOR_CO_H

#include "deallocator_co.h"

#include "bad_alloc_cc.h"
#include "persister_cc.h"

#include <cstddef> /* size_t, ptrdiff_t */
#include <new> /* bad_alloc */

template <typename T, typename Persister>
	struct allocator_co;

template <>
	struct allocator_co<void, persister>
		: public deallocator_co<void, persister>
	{
		using deallocator_type = deallocator_co<void, persister>;
		using typename deallocator_type::pointer;
		using typename deallocator_type::const_pointer;
		using typename deallocator_type::value_type;
	};

template <typename Persister>
	struct allocator_co<void, Persister>
		: public deallocator_co<void, Persister>
	{
		using deallocator_type = deallocator_co<void, Persister>;
		using typename deallocator_type::pointer;
		using typename deallocator_type::const_pointer;
		using typename deallocator_type::value_type;
	};

template <typename T, typename Persister = persister>
	struct allocator_co
		: public deallocator_co<T, Persister>
	{
	private:
		heap_co *_heap;
	public:
		using deallocator_type = deallocator_co<T, Persister>;
		using typename deallocator_type::value_type;
		using typename deallocator_type::size_type;
		using typename deallocator_type::difference_type;
		using pointer = pointer_pobj<T, 0U>;
		using const_pointer = pointer_pobj<const T, 0U>;

		explicit allocator_co(heap_co &heap_, Persister p_ = Persister())
			: deallocator_co<T, Persister>(p_)
			, _heap(&heap_)
		{}

		allocator_co(const allocator_co &a_) noexcept = default;

		template <typename U, typename P>
			allocator_co(const allocator_co<U, P> &a_) noexcept
				: allocator_co(a_.heap())
			{}

		allocator_co &operator=(const allocator_co &a_) = delete;

		auto allocate(
			size_type s
			, size_type // alignment
					= alignof(T)

		) -> pointer
		{
			auto oid = heap().malloc(s * sizeof(T));
			if ( OID_IS_NULL(oid) )
			{
				throw bad_alloc_cc(0, s, sizeof(T));
			}
			return pointer(oid);
		}

		void persist(const void *ptr, size_type len, const char * = nullptr)
		{
			Persister::persist(ptr, len);
		}

		auto &heap() const
		{
			return *_heap;
		}
	};

#endif
