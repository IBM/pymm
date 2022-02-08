/*
   Copyright [2019-2021] [IBM Corporation]
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

#include "hstore_config.h"
#include "alloc_key.h" /* AK_ACTUAL */
#include "construction_mode.h"
#include "monitor_extend.h"
#include "perishable.h"
#include "segment_layout.h"
#if 1
#include "logging.h"
#endif

#include <type_traits> /* is_base_of */

/*
 * ===== persist_map =====
 */

template <typename Allocator>
	impl::persist_map<Allocator>::persist_map(
		AK_ACTUAL
		std::size_t n, Allocator av_
		, allocation_state_emplace *ase_
		, allocation_state_pin *aspd_
		, allocation_state_pin *aspk_
		, allocation_state_extend *asx_
	)
		: _size_control()
		, _segment_count(
			/* The map tends to split when it is about 40% full.
			 * Triple the expected object count when computing a segment count.
			 */
			((n*3U)/base_segment_size == 0 ? 1U : segment_layout::log2((3U * n)/base_segment_size))
		)
		, _sc{}
		, _ase{ase_}
		, _aspd{aspd_}
		, _aspk{aspk_}
		, _asx{asx_}
	{
		/* do_initial_allocation now requires a persist_map_controller, to interpret the
		 * allocation_state_combined field. Construct a temporary one here.
		 * The permanent persist_map_controller will be constructed later.
		 * ERROR: See if we can use a single persist_map_controller, constructed at an
		 * appropriate time.
		 */
		persist_map_controller<Allocator> pc(AK_REF av_, this, construction_mode::create);
		do_initial_allocation(AK_REF &pc);
	}

template <typename Allocator>
	void impl::persist_map<Allocator>::do_initial_allocation(
		AK_ACTUAL
		gsl::not_null<persist_map_controller<Allocator> *> pc_)
	{
		auto &av = static_cast<Allocator &>(*pc_);
		if ( _segment_count.actual().is_stable() )
		{
			if ( _segment_count.actual().value() == 0 )
			{
				/*
				 * (1) save enough information to know when the allocated pointer is hardened. In this case, the address and new value of the length of the segment table
				 *
				 * inlcudes setting of doubt type to emplace
				 */

				/* ERROR: we can get at the pesistent state two ways: through the allocator (which
				 * knows about the allocation_state_extend member of persistent_state) and
				 * pc_, which is the persistent controller. This is one too many ways.
				 */
				monitor_extend<Allocator> m{bucket_allocator_t(av)};

				pc_->record_segment_count_addr_and_target_value(&_segment_count, _segment_count.actual().value() + 1);
				/* Run the allocation */
				bucket_allocator_t(av).allocate(
					AK_REF
					_sc[0].bp
					, base_segment_size
					, segment_align
				);

				/* (2) Someone needs to persist the pointer at _sc[0].bp. We let the "tentative_allocator" (part of the "tentative_allocation_state") do that. */
				new ( &*_sc[0].bp ) bucket_aligned_t[base_segment_size];
				/* (3) atomic increment */
				_segment_count.actual_incr();
				/* (4) presist the change */
				av.persist(&_segment_count, sizeof _segment_count);
				/* (5) destructor of monitor_extend will set the allocation_state_extend
				 *     state to idle, indicating that address at _sc[0].bp, which the
				 *     allocator will remember for a while, is definitely allocated and
				 *     not in doubt.
				 */
			}

			/* while not enough allocated segments to hold n elements */
			for ( auto ix = _segment_count.actual().value(); ix != _segment_count.specified(); ++ix )
			{
				auto segment_size = base_segment_size<<(ix-1U);

				monitor_extend<Allocator> m{bucket_allocator_t(av)};
				if ( pc_->pool()->is_crash_consistent() )
				{
					pc_->record_segment_count_addr_and_target_value(&_segment_count, _segment_count.actual().value() + 1);
				}

				bucket_allocator_t(av).allocate(
					AK_REF
					_sc[ix].bp
					, segment_size
					, segment_align
				);

				new (&*_sc[ix].bp) bucket_aligned_t[base_segment_size << (ix-1U)];
				_segment_count.actual_incr();
				av.persist(&_segment_count, sizeof _segment_count);
			}

			av.persist(&_size_control, sizeof _size_control);
		}
	}

template <typename Allocator>
	void impl::persist_map<Allocator>::reconstitute(Allocator av_)
	{
		/*
		 * Note: persist_map<Allocator>::reconstitute(Allocator) is not called
		 * if is_crash_consistent, so this code is not yet exercised.
		 */
		if ( av_.pool()->is_crash_consistent() )
		{
			/* */
			/* manifest constant 4 is number of possible emplace/erase deallocations (though 2 is the maximum expected) */
			for ( auto i = 0; i != 4; ++i )
			{
#if ! TEST_HSTORE_PERISHABLE // compile error when testing perishable
				if ( auto p = static_cast<typename Allocator::pointer_type>(ase()->er_disused_ptr(i)) )
				{
					PLOG(PREFIX "possibly incomplete deallocation at %p", LOCATION, static_cast<void *>(p));
#if 0
					/* error: no version of deallocate works without a size */
					av_.deallocate(p);
#endif
				}
#endif
			}
			/* emplace can be disarmed now. */
			av_.emplace_disarm();
			/* extend has only allocations (no deallocations), so it could have been
			 * disarmed when the allocator was reinstantiated. But it was not,
			 * so disarm it here.
			 */
			av_.extend_disarm();
		}
		else if ( av_.pool()->can_reconstitute() )
		{
			auto av = bucket_allocator_t(av_);
			if ( ! _segment_count.actual().is_stable() || _segment_count.actual().value() != 0 )
			{
				segment_layout::six_t ix = 0U;
				av.reconstitute(base_segment_size, _sc[ix].bp);
				++ix;
	
				/* restore segments beyond the first */
				for ( ; ix != _segment_count.actual().value_not_stable(); ++ix )
				{
					auto segment_size = base_segment_size<<(ix-1U);
					av.reconstitute(segment_size, _sc[ix].bp);
				}
				if ( ! _segment_count.actual().is_stable() )
				{
					/* restore the last, "junior" segment */
					auto segment_size = base_segment_size<<(ix-1U);
					av.reconstitute(segment_size, _sc[ix].bp);
				}
			}
		}
	}
