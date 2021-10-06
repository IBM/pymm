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


#ifndef MCAS_HSTORE_DEALLOCATOR_RC_H
#define MCAS_HSTORE_DEALLOCATOR_RC_H

#include "hstore_config.h"
#include "heap_access.h"
#include "persistent.h"
#include "persister_cc.h"

#include <cstddef> /* size_t, ptrdiff_t */
#include <stdexcept> /* logic_error */

template <typename T, typename Heap, typename Persister>
	struct deallocator_rc;

template <typename Heap, typename Persister>
	struct deallocator_rc<void, Heap, Persister>
	{
		using value_type = void;
	};

template <typename T, typename Heap, typename Persister = persister>
	struct deallocator_rc
		: public Persister
	{
		using heap_type = Heap;
	private:
		heap_access<heap_type> _pool;
	public:
		using value_type = T;
		using persister_type = Persister;
		using size_type = std::size_t;
		using pointer_type = persistent_t<value_type *>;

		explicit deallocator_rc(const heap_access<heap_type> &pool_, Persister p_ = Persister()) noexcept
			: Persister(p_)
			, _pool(pool_)
		{}

		explicit deallocator_rc(const deallocator_rc &) noexcept = default;

		template <typename U, typename P>
			explicit deallocator_rc(const deallocator_rc<U, Heap, P> &d_) noexcept
				: deallocator_rc(d_.pool())
			{}

		deallocator_rc &operator=(const deallocator_rc &e_) = delete;

		/*
		 * Note: emplace_{arm,disarm} must be no-ops in pools which do not support
		 * crash-consistency.
		 */
		void emplace_arm()
		{
			_pool->emplace_arm();
		}

		void emplace_disarm()
		{
			_pool->emplace_disarm();
		}

		void deallocate(
			pointer_type & p_
			, size_type sz_
		)
		{
			_pool->free(reinterpret_cast<persistent_t<void *> &>(p_), sizeof(T) * sz_);
		}

		void deallocate(
			pointer_type & p_
		)
		{
			/* What we might like to say, if persistent_t had the intelligence:
			 * _pool->free(static_pointer_cast<void *>(&p));
			 */
			_pool->free(reinterpret_cast<persistent_t<void *> *>(&p_));
		}

		/* Deallocate a "tracked" allocation.
		 * The crash-consistent allocator remembers allocations, so hstore does not
		 * need to "track" them.
		 *
		 * The "reconstituting" allocator does not track memory allocations.
		 * The memory used by kvstore::alloc_memory anmd kvstore::free_memory
		 * must therefore be tracked by someone else. That job falls to hstore,
		 * even though it has nothing to do with the hash store.
		 */
		void deallocate_tracked(
			const void *p_
			, size_type sz_
		)
		{
			_pool->free_tracked(p_, sizeof(T) * sz_);
		}

		void persist(const void *ptr, size_type len, const char * = nullptr) const
		{
			persister_type::persist(ptr, len);
		}

		auto pool() const
		{
			return _pool;
		}
	};

#endif
