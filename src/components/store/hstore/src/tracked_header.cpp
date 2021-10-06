/*
   Copyright [2020-2021] [IBM Corporation]
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

#include "tracked_header.h"

#include "injectee.h"
#include <common/logging.h>
#include <common/pointer_cast.h>

tracked_header::tracked_header(unsigned debug_level_, tracked_header *prev_, tracked_header *next_, std::size_t size_, std::size_t align_)
	: _prev(prev_)
	, _next(next_)
	, _size(size_)
	, _align(align_)
{
	const char *p = common::pointer_cast<const char>(this);
	if ( 3 < debug_level_ )
	{
		PLOG("%s: begin %p this %p data %p end %p (size 0x%zx align 0x%zx)"
			, __func__
			, common::p_fmt(p+sizeof(*this)-_align)
			, common::p_fmt(p)
			, common::p_fmt(p + sizeof(*this))
			, common::p_fmt(p +sizeof(*this) -_align+_size)
			, _size
			, _align
		);
	}
}

void tracked_header::recover(unsigned debug_level_, injectee *eph_)
{
	/* _next ptrs must be a consistent circular list.
	 * Fix up _prev ptrs so that they are consistent with next.
	 * As _prev need not survive a crash, no need to flush.
	 */
	auto h = this;
	/* Fix h->next */
	h->_next->_prev = h;

	if ( 3 < debug_level_ )
	{
		PLOG(
			"%s: TH anchor %p prev %p next %p"
			, __func__
			, common::p_fmt(h)
			, common::p_fmt(h->_prev)
			, common::p_fmt(h->_next)
		);
	}

	for ( auto e = h->_next; e != h; e = e->_next )
	{
		e->_next->_prev = e;
		eph_->inject_allocation(common::pointer_cast<char>(e+1) - e->_align, e->_size);
		if ( 3 < debug_level_ )
		{
			PLOG(
				"%s: TH %p prev %p next %p size %zu align %zu"
				, __func__
				, common::p_fmt(e)
				, common::p_fmt(e->_prev)
				, common::p_fmt(e->_next)
				, e->_size
				, e->_align
			);
		}
	}
}
