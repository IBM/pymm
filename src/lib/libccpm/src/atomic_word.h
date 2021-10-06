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

#ifndef CCPM_ATOMIC_WORD_H__
#define CCPM_ATOMIC_WORD_H__

#include <cstdint>
#include <limits>

namespace ccpm
{
	/* Unit which can be stored and persisted atomically.
	 * (128 bits can also be stored and persisted atomically, but
	 * we do not use it. Yet.)
	 */
	using atomic_word = std::uint64_t;

	constexpr unsigned log2(unsigned x)
	{
		return x <= 1 ? 0U : log2(x>>1)+1;
	}

	static constexpr unsigned alloc_states_per_word =
		std::numeric_limits<atomic_word>::digits;

	static constexpr unsigned log2_alloc_states_per_word =
		log2(alloc_states_per_word);

	/* Our atomic_words require a power of 2 number of bits, or else
	 * we have to rewrite the area_ctl::sub_size() function.
	 */
	static_assert(unsigned(1) << log2_alloc_states_per_word == alloc_states_per_word, "alloc_states_per_word not a power of 2");

	/* in the atomic word at ix, return the index of a start of a run of n free
	 * elements (or sub_states_per_word-n, if there is no such run)
	 */

	unsigned aw_find_n_free(atomic_word aw, unsigned n);
	unsigned aw_count_max_free_run(atomic_word aw);
}

#endif
