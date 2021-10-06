/*
   Copyright [2019] [IBM Corporation]
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

#include "atomic_word.h"

unsigned ccpm::aw_count_max_free_run(atomic_word aw)
{
	/* A bit is free if its value is zero.
	 */

	/* Algorithm of Hacker's Delight 6-3; modified to count runs of 0s, not 1s */
	if ( aw == 0 ) { return 64; }
	aw = ~aw;

	unsigned k = 0;
	for ( ; aw != 0x0; ++k )
	{
		aw = aw & (aw << 1);
	}
	return k;
}

namespace
{
	/* find n free bits. The loop is O(n). Good for small n */
	inline auto find_n_free_order_n(ccpm::atomic_word aw, const unsigned n) -> unsigned
	{
		aw = ~aw;
		for ( unsigned i = 0; i != n - 1 && aw != 0; ++i )
		{
			aw &= aw >> 1;
		}
		return aw == 0 ? ccpm::alloc_states_per_word-n+1 : unsigned(__builtin_ctzll(aw));
	}

#if 0 // ununsed
	/* find n free bits. The loop is O(bit_count(aw) - n). Good for large n */
	inline auto find_n_free_order_w(ccpm::atomic_word aw, const unsigned n) -> unsigned
	{
		/* mask if 1's marks area to check */
		ccpm::atomic_word mask = (ccpm::atomic_word(1U) << n) - 1U;
		ccpm::atomic_word desired = ccpm::atomic_word(0);
		unsigned pos = 0;
		while ( pos + n <= ccpm::alloc_states_per_word && (aw & mask) != desired )
		{
			aw >>= 1;
			++pos;
		}
		return pos;
	}
#endif
}

/* in the atomic word aw, return the index of a start of a run of n free elements
* (or alloc_states_per_word-n+1, if there is no such run)
*/
auto ccpm::aw_find_n_free(atomic_word aw, const unsigned n) -> unsigned
{
	return find_n_free_order_n(aw, n);
}
