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

#include "area_ctl.h"

#include "area_top.h"
#include "logging.h"
#include <common/pointer_cast.h>
#include <common/utils.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include <boost/io/ios_state.hpp>
#include <iomanip>
#include <ostream>
#include <sstream> /* ostringstream */
#include <stdexcept> /* range_error */

/*
 * To avoid excess allocation to ensure alignment, sizeof(ccpm::area_ctl) should be
 * a multiple of the expected common-case alignment. For hop_hash "segments" this
 * is 64 (cache line size) and for hop_hash values this is 8 (the default alignment
 * for a hop_hash value). There is no portable way to ensure the size (without also
 * ensuring alignment, which could be 64 for segments but should not be so large
 * for values, so we use a padding field at the end of ccpm::area_ctl and check its
 * effect here.
 */

static_assert(sizeof(ccpm::area_ctl) % 64 == 0, "area_ctl is not 64-byte aligned");
static_assert(sizeof(ccpm::area_ctl) % 32 == 0, "area_ctl is not 32-byte aligned");
static_assert(sizeof(ccpm::area_ctl) % 16 == 0, "area_ctl is not 16-byte aligned");
static_assert(sizeof(ccpm::area_ctl) % 8 == 0, "area_ctl is not 8-byte aligned");

/*
 * TODO:
 */

#define PERSIST(pf, x) ((pf)->persist(common::make_byte_span(&(x), sizeof (x))))
#define PERSIST_N(pf, p, ct) ((pf)->persist(common::make_byte_span((p), (sizeof *(p)) * (ct))))

struct verifier
{
private:
	const ccpm::area_ctl *_a;
public:
	verifier(const ccpm::area_ctl *a_)
		: _a(a_)
	{
		assert(_a->is_valid());
	}
	verifier(const verifier &) = delete;
	verifier& operator=(const verifier &) = delete;
	~verifier()
	{
		assert(_a->is_valid());
	}
};

constexpr ccpm::area_ctl::index_t ccpm::area_ctl::ct_atomic_words;
constexpr std::size_t ccpm::area_ctl::min_alloc_size;

/* Constructor for just enough of an area_ctl to hold _full_height
 * Also used as the simplified area_ctl constructior for the possibly most-common case: level 0, header_ct_ 0, element_count max_elements
 */
ccpm::area_ctl::area_ctl(
	persist_type persist_
	, level_ix_t full_height_
)
	: _full_height(full_height_)
	, _level(0)
	, _element_count(max_elements)
	, _dt()
	, _alloc_bits{{0}}
	, _element_state()
#if USE_MAGIC
	, _magic(magic)
#endif
  , _root()
#if USE_PADDING
	, _padding()
#endif
{
	assert(reinterpret_cast<uintptr_t>(this) % alignof(area_ctl) == 0);
#if USE_PADDING
	(void)_padding; // unused
#endif
	PERSIST(persist_, *this);
}

ccpm::area_ctl::area_ctl(
	persist_type persist_
	, level_ix_t level_
	, index_t element_count_
	, level_ix_t header_ct_
	, level_ix_t full_height_
)
	: _full_height(full_height_)
	, _level(level_)
	, _element_count(element_count_)
	, _dt()
	, _alloc_bits{{0}}
	, _element_state()
#if USE_MAGIC
	, _magic(magic)
#endif
  , _root()
#if USE_PADDING
	, _padding()
#endif
{
#if USE_PADDING
	(void)_padding; // unused
#endif

	assert(reinterpret_cast<uintptr_t>(this) % alignof(area_ctl) == 0);
	/* *** element_bitmap *** */

	/* These are available for allocation */

	/* Elements in the range [element_count_ .. max_elements) are permanently
	 * marked "allocated, reserved" because they are past end of storage
	 */
	el_reserve_range(persist_, element_count_, max_elements);

	/* If we are at the lowest level, or allocation at a lower level would cause
	 * the space for area_ctls to exceed one atomic word, mark the bitmap elements
	 * covering area_ctl and the bitmap itself as allocated, to prevent their use
	 * by the allocator.
	 * Otherwise, add subdivision beneath.
	*/
	if (
		_level == 0
		||
		alloc_states_per_word < front_pad_count(level_ix_t(header_ct_ + 1), sub_size() / alloc_states_per_word)
	)
	{
		/* These are permanently "allocated" to instances of area_ctl */
		el_reserve_range(persist_, 0, front_pad_count(level_ix_t(header_ct_ + _level)));

		/* If not at lowest level, and in case we are constructing the initial
		 * area_ctl, the origin needs an area_ctl with at least _full_height field
		 * so that recovery can find the root area_ctl.
		 */
		if ( _level != 0 )
		{
			new (element_void(0)) area_ctl(persist_, full_height_);
			PERSIST(persist_, *static_cast<area_ctl *>(element_void(0)));
		}
	}
	else
	{
		new_subdivision(persist_, level_ix_t(header_ct_ + 1));
	}

	/* elements are uninitialized. The allocator perhaps ought to zero them on
	 * allacation
	 */
#if USE_MAGIC
	_magic = magic;
#endif
	PERSIST(persist_, *this);
}

auto ccpm::area_ctl::sub_size(level_ix_t level) -> std::size_t
{
	return area_ctl::min_alloc_size * (std::size_t(1) << log2_alloc_states_per_word * level);
}

auto ccpm::area_ctl::sub_size() const -> std::size_t
{
	return sub_size(_level);
}

auto ccpm::area_ctl::commission(
	persist_type persist_
	, byte_span span_
) -> area_ctl *
{
	auto h = height(::size(span_));
	auto top_level = level_ix_t(h - 1);
	auto pos = static_cast<area_ctl *>(::base(span_)) + top_level;
	return
		new
			(pos)
			area_ctl(
				persist_
				, top_level
				, index_t(::size(span_) / sub_size(top_level))
				, 1
				, h
			)
		;
}

bool ccpm::area_ctl::includes(const void *ptr) const
{
	return
		this <= ptr
		&&
		static_cast<const char *>(ptr) < common::pointer_cast<const char>(this) + _element_count * sub_size()
		;
}

/* Precondition: this area has a run of ct_atomic_words empty slots
 *
 */
auto ccpm::area_ctl::new_subdivision(
	persist_type persist_
	, level_ix_t header_ct_
) -> area_ctl *
{
	verifier v(this);
	const auto ix = el_find_n_free(ct_atomic_words);
	/* caller should have checked that the area had at least ct_atomic_words
	 * empty slots
	 */
	assert(ix != _element_count);
	const index_t element_count = max_elements;
	/* construct area */
	assert(sub_size() > element_count);
	/* Okay to construct the area before acquiring ownership, because the code
	 * is not multithreaded. A multithreaded version should add a "doubt" element
	 * for subdivision allocation, and modify state only *after* obtaining
	 * ownership by setting alloc bits, i.e.
	 *   1. persist doubt
	 *   2. set allocate bits (may fail if another thread exists)
	 *   3. set state bytes, commission (any order)
	 *   4. clear doubt
	 * Furthermore, recovery in a a multithreaded version needs to check the
	 * subdivision doubt(s), and clear the allocation bits where doubt exists
	 * (because the subdivision did not complete).
	 */

	const auto pac =
		( _level == 1 && header_ct_ == 0 )
		/* possibly-common case fast path */
		? new (element_byte(ix)) area_ctl(persist_, _full_height)
		/* slow path */
		: new
			/* An area_ctl at _level is (_level * sizeof *this) bytes past
			 * the start of the first parent element
			 */
			(element_byte(ix) + (_level-1) * (sizeof *this))
			area_ctl(
				persist_
				, level_ix_t(_level - 1)
				, element_count
				, header_ct_
				, _full_height
			)
		;
	auto &aw = el_allocate_n(persist_, ix, ct_atomic_words, sub_state::subdivision);
	PERSIST(persist_, aw);

	return pac;
}

auto ccpm::area_ctl::height(std::size_t bytes_) -> level_ix_t
{
	level_ix_t h = 1U;

	std::size_t subdivision_size = min_alloc_size;
	while ( subdivision_size * ccpm::area_ctl::max_elements < bytes_ )
	{
		if ( subdivision_size * alloc_states_per_word <= subdivision_size )
		{
			std::ostringstream s;
			s << "Area size " << std::hex << std::showbase << bytes_ << " bytes exceeds maximum " << subdivision_size << " bytes";
			throw std::range_error(s.str());
		}
		subdivision_size *= ccpm::alloc_states_per_word;
		++h;
	}
	return h;
}

/* functions prefixed "el" operate on one or more of all elements in the element
 * state array
 */
void ccpm::area_ctl::el_set_alloc(
	persist_type persist_
	, index_t ix
	, bool alloc_state
)
{
	verifier v(this);
	const auto outer_offset = ix / alloc_states_per_word;
	const auto inner_offset = ix % alloc_states_per_word;
	auto &aw = _alloc_bits[outer_offset];
	aw =
		( aw & ~(atomic_word(1U) << inner_offset) )
		| (atomic_word(alloc_state) << inner_offset)
		;
	PERSIST(persist_, aw);
}

auto ccpm::area_ctl::el_fill_state_range(
	persist_type persist_
	, const index_t first_, const index_t last_, const sub_state s
) -> index_t
{
	verifier v(this);
	std::fill(&_element_state[first_], &_element_state[last_], s);
	PERSIST_N(persist_, &_element_state[first_], last_ - first_);
	return last_;
}

auto ccpm::area_ctl::el_fill_alloc_range(persist_type persist_, index_t first_, index_t last_, bool alloc)
	-> index_t
{
	verifier v(this);
	for ( ; first_ != last_; ++first_ )
	{
		el_set_alloc(persist_, first_, alloc);
	}
	return last_;
}

auto ccpm::area_ctl::el_alloc_value(index_t ix) const -> bool
{
	verifier v(this);
	const auto outer_offset = ix / alloc_states_per_word;
	const auto inner_offset = ix % alloc_states_per_word;
	const atomic_word aw = _alloc_bits[outer_offset];
	return (aw >> inner_offset) & 1;
}

auto ccpm::area_ctl::el_state_value(index_t ix) const -> sub_state
{
	verifier v(this);
	/* bad form to call this if not allocated, as the value is a dont care */
	assert(el_alloc_value(ix));
	return _element_state[ix];
}

auto ccpm::area_ctl::el_find_n_free(index_t n) const -> index_t
{
	verifier v(this);
	for ( index_t outer_offset = 0; outer_offset != _element_count; ++outer_offset )
	{
		const auto pos = aw_find_n_free(_alloc_bits[outer_offset], n);
		if ( pos + n <= alloc_states_per_word )
		{
			return outer_offset * alloc_states_per_word + pos;
		}
	}
	return _element_count;
}

auto ccpm::area_ctl::el_run_at(const index_t ix) const -> index_t
{
	verifier v(this);
	auto iy = ix + 1;
	for (
		; iy != _element_count && el_is_continued(iy)
		; ++iy
	)
	{
	}
	return index_t(iy - ix);
}

auto ccpm::area_ctl::el_client_run_at(const index_t ix) const -> index_t
{
	verifier v(this);
	assert(el_is_client_start(ix)); /* should have been verified by caller */
	auto n = el_run_at(ix);
	/* Error if the run crosses a word boundary */
	assert(ix/alloc_states_per_word == (ix+n-1)/alloc_states_per_word);
	return n;
}

auto ccpm::area_ctl::el_reserved_run_at(const index_t ix) const -> index_t
{
	verifier v(this);
	assert(el_is_reserved(ix)); /* should have been verified by caller */
	return el_run_at(ix);
}

auto ccpm::area_ctl::el_subdivision_run_at(const index_t ix) const -> index_t
{
	verifier v(this);
	assert(el_is_subdivision(ix)); /* should have been verified by caller */
	auto n = el_run_at(ix);
	/* Error if the run crosses a word boundary */
	assert(ix/alloc_states_per_word == (ix+n-1)/alloc_states_per_word);
	return n;
}

/* functions prefixed "aw" operate on only a single atomic_word, which is a subset
 * of the element state array
 */

/*
 * Precondition:
 *   (The process may crash
 *   *after* having modified the allocation bits and
 *   *before* the allocation has completed (element_state changed and the requestor
 *   having recorded the allocation).)
 *
 *   Therefore the caller must have persisted a record of "doubt" must have been
 *   persisted which recorded the address and length of the intended allocation.
 *
 *   The caller should invalidate the record of doubt after it has accepted
 *   ownership of the area.
 *
 *   Two callers use el_allocate_n:
 *    (1) Allocation by a client. In this case the caller will clear the doubt
 *        after it has (a) persisted the state bytes (in allocator-owned memory)
 *        and (b) persisted the allocation pointer in client-owned memory.
 *    (2) Allocation of a subdivision internally by area_ctl. In this case the
 *        subdivision builder will clear the doubt after it has persisted the
 *        state bytes (in allocator-owned memory).
 *
 * Starting with element ix, replace n 0s with the client allocated indicators
 * (Must affect not more than one single atomic_word.)
 */
auto ccpm::area_ctl::el_allocate_n(
	persist_type persist_
	, const index_t ix
	, const index_t n
	, const sub_state s
) -> atomic_word &
{
	verifier v(this);
	assert(0 < n);
	const auto outer_offset = ix / alloc_states_per_word;
	const auto inner_offset = ix % alloc_states_per_word;

	/* Since this allocation is single threaded, it is okay to write the states
	 * before changing alloc bits. If this were a thread-safe allocator, with
	 * alloc bits arbitrating ownership among threads, we would have to
	 * (1) mark the area "in doubt", and persist that
	 * (2) set and persist the alloc bits
	 * (3) set and persist the states
	 * (4) remove the "in doubt" marker.
	 */

	_element_state[ix] = s;
	el_fill_state_range(persist_, ix+1, ix+n, sub_state::continued);
	/* the bit mask to add */
	atomic_word res = (atomic_word(1U) << n) - 1U;

	auto &aw = _alloc_bits[outer_offset];
	aw |= (res << inner_offset);
	return aw;
}

void ccpm::area_ctl::el_reserve_range(
	persist_type persist_
	, index_t first_
	, const index_t last_
)
{
	verifier v(this);

	if ( first_ != last_ )
	{
		el_fill_state_range(persist_, first_, first_ + 1, sub_state::reserved);
		el_fill_state_range(persist_, first_ + 1, last_, sub_state::continued);
	}

	for ( ; first_ != last_; ++first_ )
	{
		const auto outer_offset = first_ / alloc_states_per_word;
		const auto inner_offset = first_ % alloc_states_per_word;

		/* the bit mask to add */
		atomic_word res = (atomic_word(1U) << inner_offset);

		auto &aw = _alloc_bits[outer_offset];
		aw |= res;
	}
	PERSIST(persist_, _alloc_bits);
}

auto ccpm::area_ctl::el_deallocate_n(
	const index_t ix
	, const index_t n
) -> atomic_word &
{
	verifier v(this);
	assert(0 < n);
	const auto outer_offset = ix / alloc_states_per_word;
	const auto inner_offset = ix % alloc_states_per_word;
	const auto mask = ((atomic_word(1U) << (n)) - 1U) << inner_offset;
	auto &aw = _alloc_bits[outer_offset];
	aw &= ~mask;
	/* ERROR: aw must persist before the element states, if deallocate chooses
	 * to change element stats.
	 */
	return aw;
}

/* pad count, in elements */
auto ccpm::area_ctl::front_pad_count(const level_ix_t header_ct_) const -> index_t
{
	verifier v(this);
	return front_pad_count(header_ct_, sub_size());
}

auto ccpm::area_ctl::front_pad_count(const level_ix_t header_ct_, std::size_t sub_size_) -> index_t
{
	return index_t(div_round_up(sizeof(area_ctl) * header_ct_, sub_size_));
}

auto ccpm::area_ctl::max_free_element_count(const level_ix_t level) -> index_t
{
	return
		alloc_states_per_word
		-
		( ct_atomic_words == 1 ? front_pad_count(1, sub_size(level)) : 0 )
		;
}

std::size_t ccpm::area_ctl::bytes_free() const
{
	verifier v(this);
	std::size_t r = 0;
	for ( index_t ix = 0; ix != _element_count; ++ix )
	{
		r +=
			el_is_free(ix) ? sub_size()
			: el_is_subdivision(ix) ? area_child(ix)->bytes_free()
			: std::size_t(0);
	}
	return r;
}

auto ccpm::area_ctl::elements_free_local() const -> index_t
{
	verifier v(this);
	index_t r = 0;
	for ( index_t ix = 0; ix != _element_count; ++ix )
	{
		r += el_is_free(ix);
	}
	return r;
}

std::size_t ccpm::area_ctl::bytes_free_local() const
{
	verifier v(this);
	return elements_free_local() * sub_size();
}

std::size_t ccpm::area_ctl::bytes_free_sub() const
{
	verifier v(this);
	std::size_t r = 0;
	for ( index_t ix = 0; ix != _element_count; ++ix )
	{
		r +=
			el_is_subdivision(ix) ? area_child(ix)->bytes_free()
			: std::size_t(0);
	}
	return r;
}

/* Returns
 *  0: pointer to the area_ctl
 *  1: the index of the element
 *  2: the number of levels below the top at which the element was found
 *  3: true iff the pointer was aligned with the start of an element allocation
 */
auto ccpm::area_ctl::locate_element(
	void *const ptr_
) -> element_location
{
	verifier v(this);
	assert( element_void(0) <= ptr_ );
	if ( ptr_ < element_void(0) )
	{
		std::ostringstream o;
		o << "locate_element: ptr " << ptr_ << " below allocation range ["
			<< element_void(0) << ".." << element_void(_element_count) << ")";
		throw std::runtime_error(o.str());
	}

	assert( ptr_ < element_void(_element_count) );
	if ( element_void(_element_count) <= ptr_ )
	{
		std::ostringstream o;
		o << "locate_element: ptr " << ptr_ << " above allocation range ["
			<< element_void(0) << ".." << element_void(_element_count) << ")";
		throw std::runtime_error(o.str());
	}

	const auto offset =
		std::size_t( static_cast<const char *>(ptr_) - element_byte(0) );
	auto ix = index_t(offset / sub_size());
	const auto ix_offset = offset % sub_size();
	if ( el_is_free(ix) )
	{
		assert( ! el_is_free(ix) );
		throw
			std::runtime_error("locate_element: deallocating free area (double free?)");
	}
	if ( el_is_client_start(ix) )
	{
		const bool aligned = el_is_client_start_aligned(ix);
		/* The start of a client, not a sub-allocation */
		if ( el_is_client_start_aligned(ix) )
		{
			if ( ix_offset != 0 )
			{
				throw
					std::runtime_error("locate_element: implausible ptr (not at start of aligned allocation)");
			}
		}
		else
		{
			if ( ix_offset == 0 )
			{
				throw
					std::runtime_error("locate_element: implausible ptr (not at start of misaligned allocation)");
			}
		}
		return element_location{this, ix, aligned};
	}
	else
	{
		/* not an exact hit, look for the sub-allocation */
		assert( sub_size() != min_alloc_size );
		if ( sub_size() == min_alloc_size )
		{
			throw
				std::runtime_error("locate_element: implausible ptr (not minimally aligned)");
		}

		/* Should be somewhere in a subdivision. If in a "continued" element, scan
		 * backwards to the start of the subdivision.
		 * A maximum of 4 tests, but could be read if each continued element of a
		 * subdivision had an offset encoded in its state.
		 */
		for ( ; el_is_continued(ix); --ix )
		{}

		assert( el_is_subdivision(ix) );
		if ( ! el_is_subdivision(ix) )
		{
			throw
				std::runtime_error(
					"locate_element: implausible ptr points inside an area not subdivision"
				);
		}
		return area_child(ix)->locate_element(ptr_);
	}
}

void ccpm::area_ctl::deallocate_local(
	persist_type persist_
	, area_top *const top_
	, doubt &dt_
	, void * & ptr_
	, const index_t element_ix_
	, const std::size_t bytes_
)
{
	verifier v(this);
	assert( el_is_client_start(element_ix_) );
	if ( ! el_is_client_start(element_ix_) )
	{
		throw
			std::runtime_error(
				"deallocate: implausible ptr (not start of an allocation)"
			);
	}

	/* remember the tier (within _level) at which this element should be found. */
	const auto old_run = el_max_free_run();
	const auto run_size = el_client_run_at(element_ix_);
	/* It is possible that the deallocation covers more elements than would be
	 * guessed by bytes_ due to aligment round-up.
	 * But the discovered run size always be at least enough to contain
	 * bytes_.
	 */
	assert(div_round_up(bytes_, sub_size()) <= run_size);
	/* two-step release:
	 * (1) reclaim space,
	 * (2) tell client that we have reclaimed the space
	 */
	dt_.set(persist_, __func__, ptr_, bytes_);
	auto &aw = el_deallocate_n(element_ix_, run_size);
	PERSIST(persist_, aw);
	/* Should not be necessary, as elements with alloc_state free and elements
	 * "in doubt" should have their state examined
	 */
#if 0
	fill_state(element_ix_, run_size, sub_state::free);
	PERSIST_N(&persist_, _element_state[element_ix_], run_size);
#endif
	ptr_ = nullptr;
	PERSIST(persist_, ptr_);
	dt_.clear(persist_, __func__);
	/* Need to move or add this area in the chains, but do not know whether the
	 * element is currently in a chain. Remove if it is in a chain, then add.
	 */
	auto new_run = el_max_free_run();
	if ( new_run != old_run )
	{
		/* remove from old list, if in one. */
		top_->remove_from_chain(this, _level, old_run);
		/* add to new list */
		top_->restore_to_chain(this, _level, new_run);
	}
}

void ccpm::area_ctl::deallocate(persist_type persist_, area_top *const top_, void * & ptr_, const std::size_t bytes_)
{
	verifier v(this);
	const auto loc = locate_element(ptr_);
	loc.ctl->deallocate_local(
		persist_
		, top_
		, _dt
		, ptr_
		, loc.element_ix
		, bytes_
	);
}

void ccpm::area_ctl::set_allocated(
	persist_type persist_
	, byte_span span_
)
{
	verifier v(this);
	const auto loc = locate_element(::base(span_));
	loc.ctl->set_allocated_local(persist_, loc.element_ix, ::size(span_), loc.is_aligned);
}

void ccpm::area_ctl::set_allocated_local(
	persist_type persist_
	, const index_t ix_, const std::size_t bytes_, const bool aligned_
)
{
	verifier v(this);
	el_allocate_n(
		persist_
		, ix_
		, index_t(div_round_up(bytes_, sub_size()))
		, aligned_ ? sub_state::client_aligned : sub_state::client_unaligned
	);
}

void ccpm::area_ctl::set_deallocated(
	persist_type persist_
	, const byte_span span_
)
{
	verifier v(this);
	const auto loc = locate_element(::base(span_));
	loc.ctl->set_deallocated_local(persist_, loc.element_ix, ::size(span_));
}

void ccpm::area_ctl::set_deallocated_local(
	persist_type persist_
	, const index_t ix_
	, const std::size_t bytes_
)
{
	verifier v(this);
	auto &aw = el_deallocate_n(ix_, index_t(div_round_up(bytes_, sub_size())));
	PERSIST(persist_, aw);
}

void ccpm::area_ctl::allocate(
	persist_type persist_
	, doubt &dt_
	, void * & ptr_
	, const std::size_t bytes_
	, const std::size_t alignment_
	, const index_t run_length_
)
{
	verifier v(this);
	/* allocate at this level if possible */
	assert( run_length_ <= alloc_states_per_word );

	const auto ix = el_find_n_free(run_length_);

	/* The caller already check that this area_ctl had a sufficient free run */
	assert( ix != _element_count );

	/* Somewhere in the elements [ix:run_length_] should be an alignment_-aligned
	 * area.
	 */

	/* byte ptr to first element in subdivision */
	const auto p0 = this->element_byte(0);
	/* byte ptr to start of first element in free run */
	const auto pe = this->element_byte(ix);
	/* amount needed to retreat to achieve alignment */
	const auto offset_bytes = reinterpret_cast<uintptr_t>(pe) % alignment_;
	/* byte ptr to range (may or may not align with the start of an element) */
	const auto pr = pe +
		( offset_bytes
			? alignment_ - offset_bytes
			: 0
		)
		;

	/* Allocation must fall within the area; logic bug if it did not. */
	assert(element_void(0) <= pr);
	assert(pr + bytes_ <= element_void(_element_count));

	/* index of first element to allocate */
	const index_t ix_begin = index_t((pr - p0) / sub_size());
	/* Note whether the allocation aligns with the start of an element.
	 * Remember that for deallocation sanity check.
	 */
	const auto is_element_aligned = index_t(pr - p0) % sub_size() == 0;
	/* end of allocation range */
	const auto pr_end = pr + bytes_;
	/* index past last element to allocate */
	const auto ix_end = index_t(div_round_up((pr_end - p0), sub_size()));
	/* under no circumstance should the end of the result go beyond the end of the
	 * run found by el_find_n_free
	 */
	assert(ix_end <= ix_begin + run_length_);
	const auto run_length_as_aligned = ix_end - ix_begin;

	/* two-step release: (1) tell client about the space, (2) release the space */
	dt_.set(persist_, __func__, pr, run_length_as_aligned * sub_size());
	ptr_ = pr;
	PERSIST(persist_, ptr_);
	auto &aw =
		el_allocate_n(
			persist_
			, ix_begin
			, run_length_as_aligned
			, is_element_aligned
				? sub_state::client_aligned
				: sub_state::client_unaligned
			);
	PERSIST(persist_, aw);
	dt_.clear(persist_, __func__);
}

/* restore_at and restore_to did not work too well, too many calls.
 * Instead, restore just once.
 *
 * (Could refine this by restoring depth-first until we found a suitable run for
 * the current allocation, and resuming the restore when we next needed a run.
 * To do this, restore_all would accept and return a vector of subdivision indices,
 * which would be used to sweep across the tree left to right until all area_ctls
 * present (at least at the start of the current run) had been restored.
 */

/* Relink all elements of this area with at least min_run elements to the
 * area_top chain */
auto ccpm::area_ctl::restore_all(
	area_top *const top_
	, std::ostream *o_
	, const level_ix_t current_level_ix_
) -> std::size_t
{
	top_->print_ctls(o_, std::ios_base::fmtflags{});

	verifier v(this);
	std::size_t count = 0;
	/*
	 * Restore only this level (no children), and if the area_ctl has a run of
	 * min_run or greater
	 */
	auto longest_run = el_max_free_run();
	if ( longest_run > 0 )
	{
		/* reset linked list element, which is left over from previous chain */
		top_->remove_from_chain(this, current_level_ix_, longest_run);
		/* place in chain */
		top_->restore_to_chain(this, current_level_ix_, longest_run);
		++count;
	}

	/* (First test is for performance only: elements at level 0 will never be
	 * subdivided.
	 */
	if ( current_level_ix_ != 0 )
	{
		for ( index_t e_ix = 0; e_ix != _element_count; ++e_ix )
		{
			if ( el_is_subdivision(e_ix) )
			{
				count +=
					area_child(e_ix)->restore_all(
						top_
						, o_
						, level_ix_t(current_level_ix_ - 1U)
					);
			}
		}
	}
	return count;
}

void ccpm::area_ctl::print(std::ostream &o_, level_ix_t indent_, std::ios_base::fmtflags format_) const
{
	verifier v(this);
	const auto si = std::string(indent_*2, ' ');
	o_ << si << "area_ctl " << this << " (" << element_void(0) << ".."
		<< element_void(_element_count)
		<< "] :\n";

	++indent_;
	const auto sj = std::string(indent_*2, ' ');

    {
		boost::io::ios_flags_saver s(o_);
		o_ << sj << "level " << unsigned(_level)
			<< ", " << _element_count << "(" << elements_free_local() << " free)"
			<< " x " << std::showbase;
		o_.setf(format_, std::ios_base::basefield);
		o_ << sub_size() << " bytes" << "\n";
    }
	for ( index_t ix = 0; ix != _element_count; ++ix )
	{
		if ( el_is_free(ix) )
		{
		}
		else
		{
			switch ( _element_state[ix] )
			{
			case sub_state::subdivision:
				o_ << sj << element_void(ix) << ": element " << ix << ":"
					<< el_subdivision_run_at(ix)
					<< " " << "subdivision\n";
				area_child(ix)->print(o_, level_ix_t(indent_ + 1U), format_);
				o_ << sj << "end subdivision\n";
				break;
			case sub_state::reserved:
				o_ << sj << element_void(ix) << ": element " << ix << ":"
					<< el_reserved_run_at(ix)
					<< " " << "reserved\n";
				break;
			case sub_state::client_aligned:
				o_ << sj << element_void(ix) << ": element " << ix << ":"
					<< el_client_run_at(ix)
					<< " " << "client.aligned\n";
				break;
			case sub_state::client_unaligned:
				o_ << sj << element_void(ix) << ": element " << ix << ":"
					<< el_client_run_at(ix)
					<< " " << "client.unaligned\n";
				break;
			case sub_state::continued:
				break;
			default:
				o_ << sj << element_void(ix) << ": element " << ix << ":"
					<< " (?unknown)\n";
				break;
			}
		}
	}
	o_ << si << "area_ctl end\n";
}

auto ccpm::area_ctl::restore(persist_type persist_, const ownership_callback_t &resolver_) -> area_ctl &
{
	PLOG(PREFIX "restore", LOCATION);
	verifier v(this);
	/* Address of the region transitioning to/from client allocated
	 * during a crash, if any
	 */
	if ( const auto p = _dt.get() )
	{
		/* Guessing that true means client-owned. Without a named function,
		 * could be either way.
		 */
		const bool client_owned = resolver_(p);
		if ( client_owned )
		{
			/* The client owns p, but the allocator may still indicate that
			 * p is free. Find the memory range for p and mark the range
			 * "client allocated."
			 */
			set_allocated(persist_, common::make_byte_span(p, _dt.bytes()));
			_dt.clear(persist_, __func__);
		}
		else
		{
			/* The client does not own p, but the allocator may still
			 * indicate that the p is allocated. Find the memory range
			 * for p and mark the range "free"
			 */
			set_deallocated(persist_, common::make_byte_span(p, _dt.bytes()));
			_dt.clear(persist_, __func__);
		}
	}
	return *this;
}

auto ccpm::area_ctl::level() const -> level_ix_t
{
	return _level;
}

auto ccpm::area_ctl::root(void *const ptr_) -> area_ctl *
{
	/* The base of the region contains an array of area_ctls.
	 * The first area_ctl is probably not the root of the area_ctl tree,
	 * but it contains the height of the tree, and the last area_ctl
	 * in the array (at position height-1) *is* the root area_ctl.
	 */
	auto ctl0 = static_cast<area_ctl *>(ptr_);
	unsigned full_height = ctl0->full_height();
	if ( 0 == full_height || 42 < full_height )
	{
		std::ostringstream s{};
		s << "area at " << ctl0 << " does not look like an area: implausible full height " << full_height;
		throw std::domain_error(s.str());
	}

	for ( const auto *c = ctl0; c != &ctl0[ctl0->full_height()]; ++c )
	{
		if ( c->_magic != magic )
		{
			std::ostringstream s{};
			s << "in area " << ctl0 << ", area_ctl at " << c << " has bad magic number";
			throw std::domain_error(s.str());
		}
		if ( c->full_height() != full_height )
		{
			std::ostringstream s{};
			s << "in area " << ctl0 << ", area_ctl at " << c << " full_height field is " << unsigned(c->full_height()) << " expected " << full_height;
			throw std::domain_error(s.str());
		}
		if ( c->_level != c - ctl0 )
		{
			std::ostringstream s{};
			s << "in area " << ctl0 << ", area_ctl at " << c << " level field is " << unsigned(c->_level) << " expected " << (c - ctl0);
			throw std::domain_error(s.str());
		}
	}
	return &ctl0[ctl0->full_height()-1];
}

void ccpm::area_ctl::set_root(const byte_span & iov, persist_type persist_) {
  _root = iov;
  persist_->persist(common::make_byte_span(&_root, sizeof _root));
}

auto ccpm::area_ctl::get_root() const -> byte_span
{
  return _root;
}
