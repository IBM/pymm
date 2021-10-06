/*
   Copyright [2019-2020] [IBM Corporation]
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

#ifndef _MCAS_HSTORE_ALLOCATION_STATE_FIXED_H
#define _MCAS_HSTORE_ALLOCATION_STATE_FIXED_H

#include "hstore_config.h" /* Persister */

#include "allocator_cc.h"
#include "cptr.h"
#include "hstore_alloc_type.h"
#include "hstore_kv_types.h"
#include "hstore_nupm_types.h"
#include "owner.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>

/* Note: "pin", in this context, means to move key or mapped_type data from a
 * small inline location within the persist_fixed_string struct to an area
 * which will not move over its lifetime.
 */

/* Allocating space for a lockable value is tricky.
 *
 * The crash-consistent allocator uses the value of a persist_fixed_string::large_t::cptr
 * to determine whether an allocation should be rolled back or rolled forward.
 * The rule is that cptr == null rolls back and cptr != null rolls forward.
 * But in conversion from an inline value to a locakble value, cptr is not naturally null
 * before the store.
 * We pin that by
 *   1. Saving a copy of the original cptr before the allocation
 *   2. Setting cptr to null before the allocation
 *   3. After a crash, and if the allocation has not committed, restoring the saved value of cptr.
 *
 * To allocate space for a lockable value (when using Crash-consistent allocator), use these steps.
 * Allocation state "armed" means that the allocation_state object can properly handle a callback on restart to ask whether we own space at an address.
 *
 * An "allocation pointer" is a pointer which could be returned by the allocator. One characterisic of an "allocation pointer" is that it is 8-byte aligned.
 * V is the persist_fixed_string object to modify.
 * T is a temporary area, part of the allocation_state_pin object, of the size of V.
 *
 * (start state is S_unarmed)
 *   0. precondition: the "arm" pointer is null.
 *      (The allocation_state is not armed because its "arm" pointer is null.)
 *   1. Shallow copy V to T.
 *   2. Set the "arm" pointer to address of V
 *      (The allocation_state is now armed because the "arm" pointer is not null.
 *      The state may not and will not be queried in an allocator callback, though,
 *      until step 4 has begun.)
 * (S_unarmed->S_calling)
 *   3. Set V.cptr to null
 *   4. Allocate space from the allocator, to be stored at in V.cptr
 *      (The allocator records the cptr location as "in doubt" and, if a crash
 *      occurs, will call back to resolve the doubt.)
 * (S_calling->S_uncommitted)
 *      (The allocator write to V.cptr commits the operation.)
 * (S_uncommitted->S_comitted)
 *   5. Shallow copy the inline data area of T to *cptr, and set the "is_fixed" flag.
 *   6. Disarm the allocation_state_pin object (set the "arm" pointer to null.)
 * (S_committed->S_unarmed)
 *
 * Restart shall
 *   1. Handle any callback from the allocator, and note whether such a callback happened
 *   2. Determine the state (S0..S4)
 *   3. Perform an operation depending on the state
 *
 *   1. Handle and make note of a call back from the allocator.
 *     a. Note that the allocator has not yet made a callback
 *     b. Call allocator initialization.
 *        If callback occurs:
 *          i. Note that allocator has made a callback
 *          ii. return "owned" to the allocator iff the callback value matches arm_ptr->cptr, else "unowned"
 *
 *   2. Determine the state:
 *     S_unarmed: arm_ptr == NULL
 *     S_calling: arm_ptr != NULL and allocator has not made a callback
 *     S_uncommitted: arm_ptr != NULL and allocator has made a callback but the callback value did not match V.cptr (which was null)
 *     S_comitted: arm_ptr != NULL and allocator has made a callback and the callback value did match V.cptr
 *
 *   3. Perform an operation depending on the state:
 *     S_unarmed (unarmed): do nothing.
 *     S_calling (armed, allocator may have been called but had not yet marked the allocation "in doubt"): Roll back by shallow copy of T.cptr to V.cptr. Disarm the allocation_state_pin
 *     S_uncommitted (armed, allocator has been called but cptr not written): Roll back by shallow copy of T.cptr to V.cptr. Disarm the allocation_state_pin
 *     S_comitted (armed, allocator has written cptr): Roll forward by executing steps 5 and 6.
 *
 */
namespace impl
{
	/* Client-side persistent allocation state when using the crash-consistent
	 * allocator in an operator which will "pin" key or value data */
	struct allocation_state_pin
	{
	private:
		using alloc_t = typename hstore_alloc_type<Persister>::alloc_type;
		using dealloc_t = typename alloc_t::deallocator_type;

		cptr _old_cptr;
		persistent_t<cptr *> _arm_ptr;
		/* used only on recovery, to record whether the allocator tested the "in_use" function */
		bool _callback_tested;

	public:
		allocation_state_pin()
			: _old_cptr{}
			, _arm_ptr(nullptr)
			, _callback_tested(false)
		{}

		void reset()
		{
			_arm_ptr = nullptr;
			_callback_tested = false;
		}

		template <typename Persister>
			void arm(
				cptr &cptr_
				, Persister p_
			)
			{
				assert(_arm_ptr == nullptr);
				/* 2. set state to "armed" */
				_old_cptr = cptr_;
				p_.persist(&_old_cptr, sizeof _old_cptr);
				_arm_ptr = &cptr_;
				p_.persist(&_arm_ptr, sizeof _arm_ptr);
			}

		template <typename Persister>
			void disarm(Persister p_)
			{
				reset();
				p_.persist(&_arm_ptr, sizeof _arm_ptr);
			}

		persistent_t<::cptr *> arm_ptr() const
		{
			return _arm_ptr;
		}

		char *get_cptr()
		{
			assert(is_armed());
			return _old_cptr.P;
		}

		bool is_armed() const { return _arm_ptr != nullptr; }

		bool was_callback_tested() const
		{
			return _callback_tested;
		}

		bool is_in_use(const void *p_)
		{
			if ( is_armed() )
			{
				_callback_tested = true;
			}
			/* In use if armed and the cptr update from nullptr to the allocated space was persisted */
			return is_armed() && _arm_ptr->P == p_;
		}
	};
}

#endif
