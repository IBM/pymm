/*
   Copyright [2019, 2020] [IBM Corporation]
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

#ifndef __CCPM_AREACTL_ALLOCATOR_H__
#define __CCPM_AREACTL_ALLOCATOR_H__

#include <ccpm/interfaces.h>
#include "doubt.h"
#include "list_item.h"

#include "atomic_word.h"
#include "element_state.h"
#include <common/byte_span.h>
#include <common/pointer_cast.h>
#include <gsl/pointers>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <ios> // ios_base::fmtflags
#include <limits>
#include <stdexcept>
#include <common/cpu.h>
#include <iosfwd>

#define USE_MAGIC 1

namespace ccpm
{
	struct area_top;

	/*
	 * |<- origin of the area_ctl element map                   |<- this area_ctl
	 *  ---------------------------------------------------------==========-----------
	 * | ... | area_ctl (1st grandchild) | area_ctl (1st child) | area_ctl | elements |
	 *  ---------------------------------------------------------==========-----------
	 *
	 * area_ctl:
	 *  ----------------------------------------------------------------------------------------------------
	 * | list_item | _full_height | _level | _element_count | _doubt | _alloc_bits | element_state | _magic |
	 *  ----------------------------------------------------------------------------------------------------
	 *
	 *   Note: The "elements" orgin is the beginning of the *first descendant*
	 *   of an area_ctl. But the array of area_ctls (from the most distant first
	 *   descendant of an area_ctl to the area_ctl itself) is overhead, and is not
	 *   available for allocation.
	 *
	 *   The _alloc_bits layout is one bit per element, and includes bits for the
	 *   overhead area. (These overhead bits are 1, meaning "allocated", and never
	 *   change.)
	 *
	 *   The element_state is one byte per element, and also includes bytes for the
	 *   overhead elements. An element state state is meaningful only if its
	 *   corresponding alloc bit is true (allocated, not free).
	 *
	 * element_alloc_map:
	 *  ------------------------------------------------------------------------
	 * |front_pad_count @ '1'b | (_element_count - front_pad_count) @ alloc_bit |
	 *  ------------------------------------------------------------------------
	 *   Note: @ is the bit replication operator: "count @ pattern"
	 *
	 *   Note: The element_alloc_map considers the records the allocation state of
	 *   all "elements." The first element is located at the start of the area_ctl.
	 *   The first front_pad_count elements are occupied by area_ctl and element_map,
	 *   and are never truly available for allocation. They are marked "allocated"
	 *   element_alloc_map.
	 *
	 * element_state:
	 *  -----------------------------------------------------------------------------
	 * |front_pad_count @ dont_care | (_element_count - front_pad_count) @ substate |
	 *  -----------------------------------------------------------------------------
	 *
	 *   Note: The element_state is one byte per element. There are only 3 or 4
	 *   states, we use an entire byte to simplify addressing. Bytes whose
	 *   corresponding alloc_bit is "free" are ignored; a free element has no
	 *   additional state.
	 *
	 *   Note: We probably do not need to include the front_pad_count bytes elements
	 *   in the element_state; their inclusion is left over from the previous
	 *   design. When this design is complete, remove those bytes.
	 *
	 * elements:
	 *  ------------------------------------------------------
	 * | (_element_count - front_pad_count) @ sub_size() bytes |
	 *  ------------------------------------------------------
	 *
	 * Non-persistent state
	 *
	 * The persistent state is enough to make the allocator crash-consistent, but not
	 * enough to make it perform well. For that, add some helping logic, and anchors
	 * in area_top, which holds non-persistent state:
	 *
	 * area_ctl chains (for those area_ctls with any free elements):
	 *
	 *   During normal execution, area_ctls of the same sub_size() *and* maximum free
	 *   element run are double linked. The links do not persist across a crash; the
	 *   chains must be rebuilt, possibly over time.
	 *   area_top pointer to each doubly-linked list. The pointers are in a
	 *   vector<array>, as the number of inner elements (alloc_states_per_word-1) is
	 *   compile bound, and the number of outer elements (dependent on area size) is
	 *   runtime bound.
	 *
	 *   Assuming smallest size 8 bytes (just large enough for a doubly-linked list),
	 *   and alloc_states_per_word==2^6,
	 *
	 *   2^3-byte (8 B) area_ctl
	 *     with maximum run length 1
	 *     with maximum run length 2
	 *      ...
	 *     with maximum run length 64 (alloc_states_per_word)
	 *   2^9-byte (512 B) area_ctl
	 *   2^15-byte (32 KiB) area_ctl
	 *   2^21-byte (2 MiB) area_ctl
	 *   2^27-byte (128 MiB) area_ctl
	 *   2^33-byte (8 GiB) area_ctl
	 *   2^39-byte (512 GiB) area_ctl
	 *   2^45-byte (32 TiB) area_ctl
	 *
	 *   (current Apache Pass memory limit: 6 TiB)
	 *
	 *   In order to restore the linked lists after a crash, allocation lookup should
	 *   be able to perform an exhaustive search for an allocation of any given size.
	 *   The information from that exhaustive search should be used to rebuild the
	 *   area_ctl chains.
	 */

	struct alignas(uint64_t) area_ctl
		: public list_item
	{
	private:
		using index_t = unsigned;
		using level_ix_t = std::uint8_t;
		using byte_span = common::byte_span;
		using persist_type = gsl::not_null<ccpm::persister *>;
	public:
		/* arbitrary number of atomic words (containing allocation status) which
		 * cover an area) */
#if 1
		static constexpr index_t ct_atomic_words = 1;
#define USE_PADDING 56
#elif 0
		static constexpr index_t ct_atomic_words = 2;
#define USE_PADDING 48
#else
		static constexpr index_t ct_atomic_words = 4;
#define USE_PADDING 32
#endif
		static constexpr std::size_t min_alloc_size = 8;

    void set_root(const byte_span & iov, persist_type persist);
    byte_span get_root() const;

	private:
		static constexpr index_t alloc_states_per_word = ccpm::alloc_states_per_word;
		static constexpr index_t max_elements = ct_atomic_words * alloc_states_per_word;
		static constexpr std::uint64_t magic = 0x0123456789abcdef;
		/* Every area_ctl has the height of the full tree, which is also the count
		 * of area_ctl elements at the start of the space.
		 */
		level_ix_t _full_height;
		level_ix_t _level;
		index_t    _element_count;

		/* _dt is used (non-zero) only in the top level area. It appears in all
		 * other areas only for consistency. Some day it might be used in more
		 * than one area for multi-thread allocation.
		 */
		doubt _dt;

		/* allocation bits, 1 per element. 1b = allocated, 0 = free */
		std::array<atomic_word, ct_atomic_words> _alloc_bits;
		/* status bytes, 1 per element. Further state for allocated elements.
		 * Contain no information for free elements.
		 */
		std::array<sub_state, max_elements> _element_state;
		std::uint64_t _magic;

    /* space for the root pointer */
    byte_span _root;

#if USE_PADDING
		char _padding[USE_PADDING];
#endif

		/* functions */

		/*
		 * level: Level of this control. 0 is the lowest level, containing elements if size 8.
		 * element_count: Count of elements which will fit in this ctl.
		 * header_ct: ??
		 * full_height: The number of levels in the full tree.
		 */
		area_ctl(
			persist_type persist
			, level_ix_t level
			, index_t element_count
			, level_ix_t header_ct
			, level_ix_t full_height
		);
		/* Simple constructor. Used, if necessary, to create area_ctl at offset 0
		 * in the region
		 */
		area_ctl(
			persist_type persist
			, level_ix_t full_height
		);

		area_ctl *area_prev() { return static_cast<area_ctl *>(this->prev()); }
		area_ctl *area_next() { return static_cast<area_ctl *>(this->next()); }

		static std::size_t sub_size(level_ix_t level);
		std::size_t sub_size() const;

		/* functions prefixed "el" operate on one or more of all elements in the
		 * element state array
		 */
		void el_set_alloc(
			persist_type persist
			, index_t ix, bool alloc_state
		);

		index_t el_fill_state_range(
			persist_type persist
			, index_t first, index_t last, sub_state s
		);
		index_t el_fill_alloc_range(
			persist_type persist
			, index_t first, index_t last, bool alloc
		);

		bool el_alloc_value(index_t ix) const;
		sub_state el_state_value(index_t ix) const;

		bool el_is_free(index_t ix) const
		{
			return el_alloc_value(ix) == false;
		}

		bool el_is_subdivision(index_t ix) const
		{
			return el_alloc_value(ix) && el_state_value(ix) == sub_state::subdivision;
		}

		bool el_is_reserved(index_t ix) const
		{
			return el_alloc_value(ix) && el_state_value(ix) == sub_state::reserved;
		}

		bool el_is_continued(index_t ix) const
		{
			return el_alloc_value(ix) && el_state_value(ix) == sub_state::continued;
		}

		bool el_is_client_start(index_t ix) const
		{
			return
				el_alloc_value(ix)
				&&
				(
					el_state_value(ix) == sub_state::client_aligned
					||
					el_state_value(ix) == sub_state::client_unaligned
				)
				;
		}

		bool el_is_client_start_aligned(index_t ix) const
		{
			return
				el_alloc_value(ix)
				&&
				el_state_value(ix) == sub_state::client_aligned
				;
		}

		/* functions prefixed "aw" operate on a single atomic word of an
		 * element state array
		 */

		/* in the atomic word at ix, return the index of a start of a run of
		 * n free elements(or alloc_states_per_word-n, if there is no such run)
		 */
		index_t el_find_n_free(index_t n) const;

		/* in elements starting at ix, find the longest substring (client, continued*)
		 * which does not cross a word boundary. (And a client_continue should never
		 * be the first element in a word.)
		 */
		index_t el_run_at(index_t ix) const;
		index_t el_client_run_at(index_t ix) const;
		index_t el_subdivision_run_at(index_t ix) const;
		index_t el_reserved_run_at(index_t ix) const;

		/* in the atomic word at ix, replace 0s with the the client allocated indicators
		*/
		auto el_allocate_n(
			persist_type persist
			, index_t ix, index_t n
			, sub_state s
		) -> atomic_word &;
		auto el_deallocate_n(index_t ix , index_t n) -> atomic_word &;
		void el_reserve_range(
			persist_type persist
			, index_t first_, index_t last
		);

		sub_state &element_state(index_t ix_)
		{
			return _element_state[ix_];
		}

		const sub_state &element_state(index_t ix_) const
		{
			return _element_state[ix_];
		}

		/* locate start of elements */
		char *area_byte()
		{
			return common::pointer_cast<char>(this - this->_level);
		}

		const char *area_byte() const
		{
			return common::pointer_cast<const char>(this - this->_level);
		}

		char *element_byte(index_t ix)
		{
			return static_cast<char *>(element_void(ix));
		}

		const char *element_byte(index_t ix) const
		{
			return static_cast<const char *>(element_void(ix));
		}

		void *element_void(index_t ix)
		{
			return area_byte() + ix * sub_size();
		}

		const void *element_void(index_t ix) const
		{
			return area_byte() + ix * sub_size();
		}

		area_ctl *area_child(index_t ix)
		{
			return static_cast<area_ctl *>(element_void(ix)) + (_level-1);
		}

		const area_ctl *area_child(index_t ix) const
		{
			return static_cast<const area_ctl *>(element_void(ix)) + (_level-1);
		}

		index_t front_pad_count(level_ix_t header_ct) const;

		/* element count needed to hold header_ct area_ctl objects with elements
		 * of size sub_size
		 */
		static index_t front_pad_count(level_ix_t header_ct, std::size_t sub_size);

		/* Returns pointer to the area_ctl, the index of the element, and the number
		 * of levels below the top at which the element was found
		 */
		struct element_location
		{
			area_ctl *ctl;
			index_t element_ix;
			bool is_aligned;
		};

		auto locate_element(void *ptr_)
			-> element_location;

		void deallocate_local(
			persist_type persist
			, area_top *top
			, doubt &dt
			, void * & ptr
			, index_t element_ix
			, std::size_t bytes
		);

		void set_allocated_local(persist_type persist, index_t ix, std::size_t bytes, bool aligned);
		void set_deallocated_local(persist_type persist, index_t ix, std::size_t bytes);

	public:
		level_ix_t full_height() const { return _full_height; }
		bool is_valid() const { return _magic == magic; }
		static level_ix_t height(std::size_t bytes);
		static auto commission(
			persist_type persist
			, byte_span span_
		) -> area_ctl *;
		bool includes(const void *ptr) const;

		std::size_t bytes_free() const;
		std::size_t bytes_free_local() const;
		index_t elements_free_local() const;
		std::size_t bytes_free_sub() const;

		bool contains(const void *p) const
		{
			auto pc = static_cast<const char *>(p);
			return element_byte(0) <= pc && pc < element_byte(_element_count);
		}

		void deallocate(
			persist_type persist
			, area_top *top
			, void * & ptr, std::size_t bytes
		);

		void allocate(
			persist_type persist
			, doubt &dt
			, void * & ptr
			, std::size_t bytes
			, std::size_t alignment
			, index_t run_length
		);

		auto new_subdivision(persist_type persist, level_ix_t header_ct) -> area_ctl *;

		/* Restore this area_ctl and all subdivisions to their area_top chains.
		 *
		 * Returns: number of area_ctls restored
		 */
		auto restore_all(
			area_top *top
            , std::ostream *o
			, level_ix_t current_level_ix
		) -> std::size_t;

		void print(std::ostream &o_, level_ix_t level_, std::ios_base::fmtflags size_format_) const;

		void set_allocated(persist_type persist, byte_span span);
		void set_deallocated(persist_type persist, byte_span span);
		auto restore(persist_type persist, const ownership_callback_t &resolver_) -> area_ctl &;
		level_ix_t level() const;

		index_t el_max_free_run() const
		{
			index_t m = 0;
			for ( unsigned ix = 0U; ix != ct_atomic_words && m != 64; ++ix )
			{
				m = std::max(m, aw_count_max_free_run(_alloc_bits[ix]));
			}
			return m;
		}

		doubt &get_doubt()
		{
			return _dt;
		}

		/* Convert allocation size to the level at which it should occur
		 * Note that this function happens to be independent of alignment,
		 * because the worst case (a region covering half the elements covered
		 * by an alloc word) can be accomodated somwehere within an alloc word.
		 */
		static level_ix_t size_to_level(std::size_t bytes_)
		{
			level_ix_t level = 0U;

			/* level 0 is area_ctl::min_alloc_size * (alloc_states_per_word-1) or fewer */
			std::size_t limit = min_alloc_size * (alloc_states_per_word-1);
			while ( limit < bytes_ )
			{
				limit *= alloc_states_per_word;
				++level;
			}

			return level;
		}

		/* convert level to element size */
		static std::size_t level_to_element_size(level_ix_t level_)
		{
			std::size_t element_size = min_alloc_size;
			for ( ; level_ != 0; --level_ )
			{
				element_size *= alloc_states_per_word;
			}
			return element_size;
			/*
			 * roughly exquvalent to
			 * return area_ctl::min_alloc_size * exp(subdivision_size, substates_per_word);
			 */
		}
		static index_t max_free_element_count(level_ix_t level);
		static auto root(void *ptr) -> area_ctl *;
	};
}


static_assert(sizeof(ccpm::area_ctl) % CACHE_LINE_SIZE == 0,
              "area_ctl size should be of integral cache lines");
#endif
