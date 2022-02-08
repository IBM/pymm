/*
   Copyright [2018-2021] [IBM Corporation]
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
#include "hstore_kv_types.h"
#include "monitor_emplace.h"
#include <common/perf/tm.h>
#include <gsl/pointers>
#include <algorithm> /* copy, move */
#include <array>
#include <stdexcept> /* out_of_range */
#include <string>
#include <vector>

struct perishable_expiry;

/* NOTE: assumes a valid map, so must be constructed *after* the map
 */
template <typename Table>
	impl::persist_atomic_controller<Table>::persist_atomic_controller(
			persist_type &persist_
			, allocator_type al_
			, construction_mode mode_
		)
			: allocator_type(al_)
			, _persist(&persist_)
#if 0
			, _tick_expired(false)
#endif
		{
			if ( mode_ == construction_mode::reconstitute && al_.pool()->can_reconstitute() )
			{
				/* reconstitute allocated memory */
				_persist->mod_key.reconstitute(allocator_type(*this));
				_persist->mod_mapped.reconstitute(allocator_type(*this));
				if ( 0 < _persist->mod_size )
				{
					allocator_type(*this).reconstitute(std::size_t(_persist->mod_size), _persist->mod_ctl);
				}
				else
				{
				}
			}
			try
			{
				TM_ROOT()
				do_op(TM_REF0);
			}
			catch ( const std::range_error & )
			{
			}
		}

template <typename Table>
	auto impl::persist_atomic_controller<Table>::do_op(TM_ACTUAL0) -> void
	{
		if ( _persist->mod_size != 0 )
		{
			if ( 0 < _persist->mod_size )
			{
				do_update(TM_REF0);
			}
			else if ( -2 == _persist->mod_size )
			{
				do_swap();
			}
			else /* Issue 41-style replacement */
			{
				do_replace();
			}
		}
	}

template <typename Table>
	auto impl::persist_atomic_controller<Table>::do_finish() -> void
	{
		_persist->mod_size = 0;
		persist_range(&_persist->mod_size, &_persist->mod_size + 1, "atomic size");

		monitor_emplace<allocator_type> m(*this);

		/* Identify the element owner for the allocations to be freed */
		if ( _persist->ase() )
		{
			auto pe = gsl::not_null<allocator_type *>(this);
			_persist->ase()->em_record_owner_addr_and_bitmask(&_persist->mod_owner, 1, *pe);
		}

		/* Atomic set of initiative to clear elements owned by persist->mod_owner */
		_persist->mod_owner = 0;
		this->persist(&_persist->mod_owner, sizeof _persist->mod_owner);
		/* Clear the elements */
		_persist->mod_key.clear();
		_persist->mod_mapped.clear();
	}

template <typename Table>
	auto impl::persist_atomic_controller<Table>::do_replace() -> void
	{
		TM_ROOT()
		/*
		 * Note: relies on the Table::mapped_type::operator=(Table::mapped_type &)
		 * being restartable after a crash.
		 */
		auto &v = _persist->map->at(TM_REF _persist->mod_key);
		std::get<0>(v) = _persist->mod_mapped;
		/* Unclear whether timestamps should be updated. The only guidance we have is the
		 * mapstore implementation, which does update timestamps on a replace.
		 */
#if ENABLE_TIMESTAMPS
		std::get<1>(v) = tsc_now();
#endif
		this->persist(&v, sizeof v);
		do_finish();
	}

template <typename Table>
	impl::persist_atomic_controller<Table>::update_finisher::update_finisher(impl::persist_atomic_controller<Table> &ctlr_)
		: _ctlr(ctlr_)
	{}

template <typename Table>
	impl::persist_atomic_controller<Table>::update_finisher::~update_finisher() noexcept(! TEST_HSTORE_PERISHABLE)
	{
		if ( ! perishable_expiry::is_current() )
		{
			_ctlr.update_finish();
		}
	}

template <typename Table>
	auto impl::persist_atomic_controller<Table>::do_update(TM_ACTUAL0) -> void
	{
		TM_SCOPE()
		{
			update_finisher uf(*this);
			char *src = _persist->mod_mapped.data();
			/* NOTE: depends on mapped type */
			auto &v = _persist->map->at(TM_REF _persist->mod_key);
			char *dst = std::get<0>(v).data();
			auto mod_ctl = &*(_persist->mod_ctl);
			for ( auto i = mod_ctl; i != &mod_ctl[_persist->mod_size]; ++i )
			{
				std::size_t o_d = i->offset_dst;
				auto dst_first = &dst[o_d];
				switch(i->op)
				{
				case component::IKVStore::Op_type::WRITE:
					{
						std::size_t o_s = i->offset_src;
						auto src_first = &src[o_s];
						std::size_t sz = i->size;
						auto src_last = src_first + sz;
						/* NOTE: could be replaced with a pmem persistent memcpy */
						persist_range(
							dst_first
							, std::copy(src_first, src_last, dst_first)
							, "atomic_update write"
						);
					}
					break;
				case component::IKVStore::Op_type::ZERO:
					/* NOTE: could be replaced with a pmem persistent memcpy */
					persist_range(
						dst_first
						, std::fill_n(dst_first, 0, i->size)
						, "atomic_update zero"
					);
					break;
				default:
					throw std::logic_error("Unsupported update code " + std::to_string(int(persistent_load(i->op))));
				}
			}
			/* Unclear whether timestamps should be updated. The only guidance we have is the
			 * mapstore implementation, which does update timestamps on a replace.
			 */
#if ENABLE_TIMESTAMPS
			std::get<1>(v) = tsc_now();
#endif
		}
#if 0
		if ( is_tick_expired() )
		{
			throw perishable_expiry(__LINE__);
		}
#endif
	}

template <typename Table>
	auto impl::persist_atomic_controller<Table>::update_finish() -> void
	{
		std::size_t ct = std::size_t(_persist->mod_size);
		do_finish();
		allocator_type(*this).deallocate(_persist->mod_ctl, ct);
	}

template <typename Table>
	auto impl::persist_atomic_controller<Table>::do_swap() -> void
	{
#pragma GCC diagnostic push
#if 8 <= __GNUC__
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
		std::memcpy(_persist->_swap.pd0, _persist->_swap.pd1, sizeof *_persist->_swap.pd0);
#pragma GCC diagnostic pop
		this->persist(_persist->_swap.pd0, sizeof *_persist->_swap.pd0, "do swap part 1");
#pragma GCC diagnostic push
#if 8 <= __GNUC__
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
		std::memcpy(_persist->_swap.pd1, &_persist->_swap.temp[0], sizeof *_persist->_swap.pd1);
#pragma GCC diagnostic pop
		this->persist(_persist->_swap.pd1, sizeof *_persist->_swap.pd1, "do swap part 2");
		/* Unclear whether timestamps should be updated. The only guidance we have is the
		 * mapstore implementation, which does update timestamps on a swap.
		 */
#if ENABLE_TIMESTAMPS
		auto now = tsc_now();
		std::get<1>(*_persist->_swap.pd0) = now;
		std::get<1>(*_persist->_swap.pd1) = now;
#endif
#if 0
		if ( is_tick_expired() )
		{
			throw perishable_expiry(__LINE__);
		}
#endif
		do_finish();
	}

template <typename Table>
	void impl::persist_atomic_controller<Table>::persist_range(
		const void *first_
		, const void *last_
		, const char *what_
	)
	{
		this->persist(first_, std::size_t(static_cast<const char *>(last_) - static_cast<const char *>(first_)), what_);
	}

template <typename Table>
	void impl::persist_atomic_controller<Table>::enter_replace(
		AK_ACTUAL
		TM_ACTUAL
		typename table_type::allocator_type al_
		, table_type *map_
		, lock_state lock_
		, const string_view key
		, const char *data_
		, std::size_t data_len_
		, std::size_t zeros_extend_
		, std::size_t alignment_
	)
	{
		TM_SCOPE()
		/* leaky */

		_persist->mod_owner = 0;
		{
			monitor_emplace<allocator_type> m(*this);

			if ( al_.pool()->is_crash_consistent() )
			{
				auto pe = gsl::not_null<allocator_type *>(this);
				_persist->ase()->em_record_owner_addr_and_bitmask(&_persist->mod_owner, 1, *pe);
			}

			_persist->mod_key.assign(AK_REF key.begin(), key.end(), lock_state::free, al_);
			_persist->mod_mapped.assign(AK_REF data_, data_ + data_len_, zeros_extend_, alignment_, lock_, al_);
			_persist->map = map_;
			this->persist(&_persist->map, sizeof _persist->map);
			_persist->mod_owner = 1;
			this->persist(&_persist->mod_owner, sizeof _persist->mod_owner);
		}

		/* 8-byte atomic write */
		_persist->mod_size = -1;
		this->persist(&_persist->mod_size, sizeof _persist->mod_size);
		do_op(TM_REF0);
	}

template <typename Table>
	template <typename IT>
		void impl::persist_atomic_controller<Table>::enter_update(
			TM_ACTUAL
			AK_ACTUAL
			typename table_type::allocator_type al_
			, table_type *map_
			, lock_state lock_
			, const string_view key
			, IT first
			, IT last
		)
		{
			TM_SCOPE()
			std::vector<char> src;
			std::vector<mod_control> mods;
			for ( ; first != last ; ++first )
			{
				auto op = (*first)->type();
				switch ( op )
				{
				case component::IKVStore::Op_type::WRITE:
					{
						const auto &wr =
							*static_cast<component::IKVStore::Operation_write *>(
								*first
							);
						auto src_offset = src.size();
						auto dst_offset = wr.offset();
						auto size = wr.size();
						auto op_src = static_cast<const char *>(wr.data());
						std::copy(op_src, op_src + size, std::back_inserter(src));
						mods.emplace_back(op, src_offset, dst_offset, size);
					}
					break;
				case component::IKVStore::Op_type::ZERO:
					{
						const auto &zr =
							*static_cast<component::IKVStore::Operation_zero *>(
								*first
							);
						mods.emplace_back(op, 0, zr.offset(), zr.size());
					}
					break;
				default:
					throw std::invalid_argument("Unsupported update code " + std::to_string(int(op)));
				};
			}

		/* leaky */
		_persist->mod_key.assign(AK_REF key.begin(), key.end(), lock_state::free, al_);
		_persist->mod_mapped.assign(AK_REF src.begin(), src.end(), lock_, al_);

		{
			/* leaky ERROR: local pointer can leak */
			persistent_t<typename std::allocator_traits<allocator_type>::pointer> ptr = nullptr;
			allocator_type(*this).allocate(
				AK_REF
				ptr
				, mods.size()
				, alignof(mod_control)
			);
			new (&*ptr) mod_control[mods.size()];
			_persist->mod_ctl = ptr;
		}

		std::copy(mods.begin(), mods.end(), &*_persist->mod_ctl);
		persist_range(
			&*_persist->mod_ctl
			, &*_persist->mod_ctl + mods.size()
			, "mod control"
		);
		_persist->map = map_;
		this->persist(&_persist->map, sizeof _persist->map);
		/* 8-byte atomic write */
		_persist->mod_size = std::ptrdiff_t(mods.size());
		this->persist(&_persist->mod_size, sizeof _persist->mod_size);
		do_op(TM_REF0);
	}

template <typename Table>
	void impl::persist_atomic_controller<Table>::enter_swap(
		mt &d0
		, mt &d1
	)
	{
		TM_ROOT()
		_persist->_swap.pd0 = &d0;
		_persist->_swap.pd1 = &d1;
		std::memcpy(&_persist->_swap.temp[0], &d0, _persist->_swap.temp.size());
		this->persist(&_persist->_swap, sizeof _persist->_swap);

		/* 8-byte atomic write */
		_persist->mod_size = -2;
		this->persist(&_persist->mod_size, sizeof _persist->mod_size);
		do_op(TM_REF0);
	}
