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

#ifndef COMANCHE_NUPM_DEALLOCATOR_RA_H
#define COMANCHE_NUPM_DEALLOCATOR_RA_H

#include "avl_malloc.h"

#include "mr_traits.h"
#include <cstddef> /* size_t, ptrdiff_t */

namespace nupm
{
template <typename T, typename MR>
class deallocator_adaptor;

template <typename MR>
class deallocator_adaptor<void, MR>
{
public:
	using pointer = void *;
	using const_pointer = const void *;
	using value_type = void;
	template <typename U>
	struct rebind
	{
		using other = deallocator_adaptor<U, MR>;
	};
};

template <typename T, typename MR>
class deallocator_adaptor
{
	/* Note: to distinguish copyable allocators from memory providers like
			 * AVL_range_allocator, C++17 pmr library refers to the latter as a
			 * "memory resources," not as a "allocators."
			 */
	MR *_pmr;

public:
	auto pmr() const
	{
		return _pmr;
	}

public:
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;
	using pointer = T *;
	using const_pointer = const T *;
	using reference = T &;
	using const_reference = const T &;
	using value_type = T;

	template <typename U>
	struct rebind
	{
		using other = deallocator_adaptor<U, MR>;
	};

	explicit deallocator_adaptor(MR &pmr_) noexcept
		: _pmr(&pmr_)
	{
	}

	explicit deallocator_adaptor(const deallocator_adaptor &) noexcept = default;

	template <typename U>
	explicit deallocator_adaptor(const deallocator_adaptor<U, MR> &d_) noexcept
		: deallocator_adaptor(d_.pmr())
	{
	}

	deallocator_adaptor &operator=(const deallocator_adaptor &e_) = delete;

	pointer address(reference x) const noexcept
	{
		return pointer(&x);
	}
	const_pointer address(const_reference x) const noexcept
	{
		return pointer(&x);
	}

	void deallocate(
		pointer p_, size_type sz_)
	{
		deallocate(p_,sz_, alignof(T));
	}

	void deallocate(
		pointer p_, size_type sz_, size_type alignment_ // alignment
	)
	{
		unsigned numa_node = 0;
		mr_traits<MR>::deallocate(this->pmr(), numa_node, p_, sz_ * sizeof(T), alignment_);
	}
};
} // namespace nupm

#endif
