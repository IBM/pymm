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


#ifndef _MCAS_HSTORE_SEGMENT_COUNT_H
#define _MCAS_HSTORE_SEGMENT_COUNT_H

// #include "as_combined.h"
// #include "bucket_aligned.h"
// #include "hash_bucket.h"
// #include "persist_fixed_string.h"
// #include "persistent.h"
#include "persist_atomic.h" /* persistent_atomic_t */
#include "segment_layout.h" /* segment_layout::six_t */
#include "value_unstable.h" /* value_unstable */

namespace impl
{
	using segment_count_actual_t = value_unstable<segment_layout::six_t, 1>;

	struct segment_count
	{
	private:
		/* current segment count */
		segment_count_actual_t _actual;
		/* desired segment count */
		persistent_atomic_t<segment_layout::six_t> _specified;
	public:
		segment_count(segment_layout::six_t specified_)
			: _actual(0)
			, _specified(specified_)
		{}
		segment_count_actual_t actual() const { return _actual; }
		auto specified() const { return _specified; }
		void actual_destabilize() { _actual.destabilize(); }
		void actual_incr() { _actual.incr(); }
		void actual_value_set_stable(segment_layout::six_t v) { _actual.value_set_stable(v); }
	};
}

#endif
