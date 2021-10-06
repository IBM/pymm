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

#include "heap_mm.h"

#include "as_emplace.h"
#include "as_extend.h"
#include "as_pin.h"
#include "clean_align.h"
#include "dax_manager.h"
#include "heap_mc_ephemeral.h"
#include "heap_mr_ephemeral.h"
#include "hstore_config.h"
#include "mm_plugin_itf.h"
#include "tracked_header.h"
#include "valgrind_memcheck.h"
#include <ccpm/cca.h> /* ctor_args */
#include <common/env.h>
#include <common/utils.h>
#include <algorithm> /* max */
#include <cinttypes>
#include <memory> /* make_unique */
#include <numeric> /* acccumulate */
#include <stdexcept> /* range_error */
#include <string> /* to_string */

namespace
{
	bool leak_check = common::env_value<bool>("LEAK_CHECK", false);

	using byte_span = common::byte_span;
	auto open_region(const std::unique_ptr<dax_manager> &dax_manager_, std::uint64_t uuid_) -> byte_span
	{
		auto & iovs = dax_manager_->open_region(std::to_string(uuid_), 0).address_map();
		if ( iovs.size() != 1 )
		{
			throw std::range_error("failed to re-open region " + std::to_string(uuid_));
		}
		return iovs.front();
	}

	const ccpm::region_vector_t add_regions_full(
		ccpm::region_vector_t &&rv_
		, const byte_span pool0_heap_
		, const std::unique_ptr<dax_manager> &dax_manager_
		, const byte_span *iov_addl_first_
		, const byte_span *iov_addl_last_
		, std::uint64_t *first_, std::uint64_t *last_
	)
	{
		auto v = std::move(rv_);
		for ( auto it = first_; it != last_; ++it )
		{
			auto r = open_region(dax_manager_, *it);
			if ( it == first_ )
			{
				(void) pool0_heap_;
				VALGRIND_MAKE_MEM_DEFINED(::base(pool0_heap_), ::size(pool0_heap_));
				VALGRIND_CREATE_MEMPOOL(::base(pool0_heap_), 0, true);
				for ( auto a = iov_addl_first_; a != iov_addl_last_; ++a )
				{
					VALGRIND_MAKE_MEM_DEFINED(::base(*a), ::size(*a));
					VALGRIND_CREATE_MEMPOOL(::base(*a), 0, true);
				}
			}
			else
			{
				VALGRIND_MAKE_MEM_DEFINED(::base(r), ::size(r));
				VALGRIND_CREATE_MEMPOOL(::base(r), 0, true);
			}
			v.push_back(r);
		}

		return v;
	}

	struct persister final
		: public ccpm::persister
	{
		void persist(common::byte_span s) override
		{
			::pmem_persist(::base(s), ::size(s));
		}
	};
	persister p_cc{};

	MM_plugin_wrapper heap_mm_make_wrapper(
		common::string_view path_
		, ccpm::persister *pe_
	)
	{
		ccpm::cca::ctor_args args{pe_, gsl::span<common::byte_span>{}, true /* force_init */, [] (const void *) -> bool { return true; }};
		MM_plugin_wrapper pw(std::string(path_), "", &args);
		pw.init();
		return pw;
	}

	MM_plugin_wrapper heap_mm_make_wrapper(
		common::string_view path_
		, ccpm::persister *pe_
		, gsl::span<common::byte_span> range_
		, std::function<bool(const void *)> callee_owns_
	)
	{
		ccpm::cca::ctor_args args{pe_, range_, false /* force_init */, callee_owns_};
		MM_plugin_wrapper pw(std::string(path_), "", &args);
		pw.init();
		return pw;
	}

	using string_view = common::string_view;
	std::unique_ptr<heap_mm_ephemeral> make_ephemeral_clear(
		const unsigned debug_level_
		, const string_view plugin_path_
		, impl::allocation_state_emplace *const ase_
		, impl::allocation_state_pin *const aspd_
		, impl::allocation_state_pin *const aspk_
		, impl::allocation_state_extend *const asx_
		, const string_view id_
		, const string_view backing_file_
		, const byte_span pool0_full_
		, const byte_span pool0_heap_
	)
	{
		/* MM_plugin_wrapper must be constructed first to determine whether
		 * the plugin supports crash consistency.
		 */
		MM_plugin_wrapper pw(heap_mm_make_wrapper(plugin_path_, &p_cc));
		return
			pw.is_crash_consistent()
			? std::unique_ptr<heap_mm_ephemeral>(
				new heap_mc_ephemeral(
					debug_level_
					, false /* clear, not restore */
					, std::move(pw)
					, ase_
					, aspd_
					, aspk_
					, asx_
					, id_
					, backing_file_
					, std::vector<byte_span>(1, pool0_full_)
					, pool0_heap_
				)
			)
			: std::unique_ptr<heap_mm_ephemeral>(
				new heap_mr_ephemeral(debug_level_, std::move(pw), id_, backing_file_)
			)
			;
	}
}

/* When used with ADO, this space apparently needs a 2MiB alignment.
 * 4 KiB sometimes produces a disagreement between server and ADO mappings
 * which manifests as incorrect key and data values as seen on the ADO side.
 */
heap_mm::heap_mm(
	const unsigned debug_level_
	, const string_view plugin_path_
	, impl::allocation_state_emplace *const ase_
	, impl::allocation_state_pin *const aspd_
	, impl::allocation_state_pin *const aspk_
	, impl::allocation_state_extend *const asx_
	, const byte_span pool0_full_
	, const byte_span pool0_heap_
	, const unsigned numa_node_
	, const string_view id_
	, const string_view backing_file_
)
	: heap(pool0_full_, pool0_heap_, numa_node_)
	, _tracked_anchor(debug_level_, &_tracked_anchor, &_tracked_anchor, sizeof(_tracked_anchor), sizeof(_tracked_anchor))
	, _eph(
		make_ephemeral_clear(
			debug_level_
			, plugin_path_
			, ase_
			, aspd_
			, aspk_
			, asx_
			, id_
			, backing_file_
			, pool0_full_
			, pool0_heap_
		)
	)
	, _pin_data(&heap_mm::pin_data_arm, &heap_mm::pin_data_disarm, &heap_mm::pin_data_get_cptr)
	, _pin_key(&heap_mm::pin_key_arm, &heap_mm::pin_key_disarm, &heap_mm::pin_key_get_cptr)
{
#if 0
	if ( _eph->is_crash_consistent() )
	{
		/* cursor now locates the best-aligned region */
		_eph->add_managed_region(_pool0_full, _pool0_heap);
		hop_hash_log<trace_heap_summary>::write(
			LOG_LOCATION
			, " pool ", ::base(_pool0_heap), " .. ", ::end(_pool0_heap)
			, " size ", ::size(_pool0_heap)
			, " new"
		);
		VALGRIND_CREATE_MEMPOOL(::base(_pool0_heap), 0, false);
	}
	else
#endif
	{
		void *last = ::end(pool0_heap_);
		if ( 0 < debug_level_ )
		{
			PLOG("%s: split %p .. %p) into segments", __func__, ::base(pool0_heap_), last);
	
			PLOG("%s: pool0 full %p: 0x%zx", __func__, ::base(_pool0_full), ::size(_pool0_full));
			PLOG("%s: pool0 heap %p: 0x%zx", __func__, ::base(_pool0_heap), ::size(_pool0_heap));
		}
		/* cursor now locates the best-aligned region */
		if ( _eph->is_crash_consistent() )
		{
#if 0
			_eph->reconstitute_managed_region(
				_pool0_full
				, _pool0_heap
				, [ase_, aspd_, aspk_, asx_] (const void *p) -> bool {
					/* To answer whether the map or the allocator owns pointer p?
					 * "true" means that the map (us, the callee) owns p
					 */
					auto cp = const_cast<void *>(p);
					return ase_->is_in_use(cp) || aspd_->is_in_use(p) || aspk_->is_in_use(p) || asx_->is_in_use(p, true);
				}
			);
#endif
		}
		else
		{
			_eph->add_managed_region(_pool0_full, _pool0_heap);
		}
		hop_hash_log<trace_heap_summary>::write(
			LOG_LOCATION
			, " pool ", ::base(_pool0_full), " .. ", ::end(_pool0_full)
			, " size ", ::size(_pool0_full)
			, " new"
		);
		VALGRIND_CREATE_MEMPOOL(::base(_pool0_heap), 0, false);
		persister_nupm::persist(this, sizeof(*this));
	}
}

namespace
{
	using string_view = common::string_view;
	std::unique_ptr<heap_mm_ephemeral> make_mm_ephemeral_reconstitute(
		unsigned debug_level_
		, const string_view plugin_path_
		, const std::unique_ptr<dax_manager> &dax_manager_
		, const string_view id_
		, const string_view backing_file_
		, const byte_span *iov_addl_first_
		, const byte_span *iov_addl_last_
		, impl::allocation_state_emplace *const ase_
		, impl::allocation_state_pin *const aspd_
		, impl::allocation_state_pin *const aspk_
		, impl::allocation_state_extend *const asx_
		, const byte_span pool0_full_
		, const byte_span pool0_heap_
		, std::uint64_t * const more_region_uuids_first_
		, std::uint64_t * const more_region_uuids_last_
	)
	{
		/* MM_plugin_wrapper must be constructed first to determine whether
		 * the plugin supports crash consistency.
		 */
		MM_plugin_wrapper
			pw(
				heap_mm_make_wrapper(
					plugin_path_
					, &p_cc
					, ccpm::region_span(&*ccpm::region_vector_t(pool0_heap_).begin(), 1)
					, [ase_, aspd_, aspk_, asx_] (const void *p) -> bool {
						/* To answer whether the map or the allocator owns pointer p?
						 * "true" that true means that the map (us, the calllee) owns p
						 */
						auto cp = const_cast<void *>(p);
						return ase_->is_in_use(cp) || aspd_->is_in_use(p) || aspk_->is_in_use(p) || asx_->is_in_use(p, true);
					}
				)
			);
		return
			pw.is_crash_consistent()
			? std::unique_ptr<heap_mm_ephemeral>(
				new heap_mc_ephemeral(
					debug_level_
					, true /* reconstitute, not clear */
					, std::move(pw)
					, ase_
					, aspd_
					, aspk_
					, asx_
					, id_
					, backing_file_
					, add_regions_full(
						ccpm::region_vector_t(
							::data(pool0_full_), ::size(pool0_full_)
						)
						, pool0_heap_
						, dax_manager_
						, iov_addl_first_
						, iov_addl_last_
						, more_region_uuids_first_
						, more_region_uuids_last_
					)
					, pool0_heap_
				)
			)
			: std::unique_ptr<heap_mm_ephemeral>(
				new heap_mr_ephemeral(
					debug_level_
					, std::move(pw)
					, id_
					, backing_file_
				)
			)
			;
	}
}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winit-self"
#pragma GCC diagnostic ignored "-Wuninitialized"
heap_mm::heap_mm(
	unsigned debug_level_
	, const string_view plugin_path_
	, const std::unique_ptr<dax_manager> &dax_manager_
	, const string_view id_
    , const string_view backing_file_
	, const byte_span *iov_addl_first_
	, const byte_span *iov_addl_last_
	, impl::allocation_state_emplace *const ase_
	, impl::allocation_state_pin *const aspd_
	, impl::allocation_state_pin *const aspk_
	, impl::allocation_state_extend *const asx_
)
	: heap(*this)
	, _tracked_anchor(this->_tracked_anchor)
	, _eph(
		make_mm_ephemeral_reconstitute(
			debug_level_
			, plugin_path_
			, dax_manager_
			, id_
		    , backing_file_
			, iov_addl_first_
			, iov_addl_last_
			, ase_
			, aspd_
			, aspk_
			, asx_
			, _pool0_full
			, _pool0_heap
			, &_more_region_uuids[0]
			, &_more_region_uuids[_more_region_uuids_size]
		)
	)
		
	, _pin_data(&heap_mm::pin_data_arm, &heap_mm::pin_data_disarm, &heap_mm::pin_data_get_cptr)
	, _pin_key(&heap_mm::pin_key_arm, &heap_mm::pin_key_disarm, &heap_mm::pin_key_get_cptr)
{
	if ( is_crash_consistent() )
	{
/* Might be a good idea to construct a heap_mc_ephemeral with *no* regions and
 * add the reconstiuted region here. But for the moment we let let
 * make_mm_ephemeral_reconstitute do that work.
 */
#if 0
		_eph->reconstitute_managed_region(
			_pool0_full
			, _pool0_heap
			, [ase_, aspd_, aspk_, asx_] (const void *p) -> bool {
				/* To answer whether the map or the allocator owns pointer p?
				 * "true" means that the map (us, the callee) owns p
				 */
				auto cp = const_cast<void *>(p);
				return ase_->is_in_use(cp) || aspd_->is_in_use(p) || aspk_->is_in_use(p) || asx_->is_in_use(p, true);
			}
		);
#endif
		hop_hash_log<trace_heap_summary>::write(
			LOG_LOCATION
			, " pool ", ::base(_pool0_heap), " .. ", ::end(_pool0_heap)
			, " size ", ::size(_pool0_heap)
			, " reconstituting"
		);
		VALGRIND_MAKE_MEM_DEFINED(::base(_pool0_heap), ::size(_pool0_heap));
		VALGRIND_CREATE_MEMPOOL(::base(_pool0_heap), 0, true);
	}
	else
	{
		_eph->add_managed_region(_pool0_full, _pool0_heap);
		hop_hash_log<trace_heap_summary>::write(
			LOG_LOCATION
			, " pool ", ::base(_pool0_full), " .. ", ::end(_pool0_full)
			, " size ", ::size(_pool0_full)
			, " reconstituting"
		);

		VALGRIND_MAKE_MEM_DEFINED(::base(_pool0_heap), ::size(_pool0_heap));
		VALGRIND_CREATE_MEMPOOL(::base(_pool0_heap), 0, true);

		for ( auto r = iov_addl_first_; r != iov_addl_last_; ++r )
		{
			_eph->add_managed_region(*r, *r);
		}

		for ( std::size_t i = 0; i != _more_region_uuids_size; ++i )
		{
			auto r = open_region(dax_manager_, _more_region_uuids[i]);
			_eph->add_managed_region(r, r);
			VALGRIND_MAKE_MEM_DEFINED(::base(r), ::size(r));
			VALGRIND_CREATE_MEMPOOL(::base(r), 0, true);
		}
		if ( auto eph = dynamic_cast<heap_mr_ephemeral *>(_eph.get()) )
		{
			_tracked_anchor.recover(debug_level_, eph);
		}
	}
}
#pragma GCC diagnostic pop

heap_mm::~heap_mm()
{
	quiesce();
}

auto heap_mm::regions() const -> nupm::region_descriptor
{
	return _eph->get_managed_regions();
}

namespace
{
	using byte_span = common::byte_span;
	std::size_t region_size(const std::vector<byte_span> &v)
	{
		return
			std::accumulate(
				v.begin()
				, v.end()
				, std::size_t(0)
				, [] (std::size_t s, byte_span iov) -> std::size_t
					{
						return s + ::size(iov);
					}
			);
	}
}

auto heap_mm::grow(
	const std::unique_ptr<dax_manager> & dax_manager_
	, std::uint64_t uuid_
	, std::size_t increment_
) -> std::size_t
{
	if ( is_crash_consistent() )
	{
		if ( 0 < increment_ )
		{
			if ( _more_region_uuids_size == _more_region_uuids.size() )
			{
				throw std::bad_alloc(); /* max # of regions used */
			}
			const auto hstore_grain_size = std::size_t(1) << (HSTORE_LOG_GRAIN_SIZE);
			auto size = ( (increment_ - 1) / hstore_grain_size + 1 ) * hstore_grain_size;

			auto grown = false;
			{
				const auto old_regions = regions();
				const auto &old_region_list = old_regions.address_map();
				const auto old_list_size = old_region_list.size();
				const auto old_size = region_size(old_region_list);
				_eph->set_managed_regions(dax_manager_->resize_region(old_regions.id(),  _numa_node, old_size + increment_));
				const auto new_region_list = regions().address_map();
				const auto new_size = region_size(new_region_list);
				const auto new_list_size = new_region_list.size();

				if ( old_size <  new_size )
				{
					for ( auto i = old_list_size; i != new_list_size; ++i )
					{
						const auto &r = new_region_list[i];
						_eph->add_managed_region(r, r);
						hop_hash_log<trace_heap_summary>::write(
							LOG_LOCATION
							, " pool ", ::base(r), " .. ", ::end(r)
							, " size ", ::size(r)
							, " grow"
						);
					}
				}
				grown = true;
			}

			if ( ! grown )
			{
				auto uuid = _more_region_uuids_size == 0 ? uuid_ : _more_region_uuids[_more_region_uuids_size-1];
				auto uuid_next = uuid + 1;
				for ( ; uuid_next != uuid; ++uuid_next )
				{
					if ( uuid_next != 0 )
					{
						try
						{
							/* Note: crash between here and "Slot persist done" may cause dax_manager_
							 * to leak the region.
							 */
							std::vector<byte_span> rv = dax_manager_->create_region(std::to_string(uuid_next), _numa_node, size).address_map();
							{
								auto &slot = _more_region_uuids[_more_region_uuids_size];
								slot = uuid_next;
								persister_nupm::persist(&slot, sizeof slot);
								/* Slot persist done */
							}
							{
								++_more_region_uuids_size;
								persister_nupm::persist(&_more_region_uuids_size, _more_region_uuids_size);
							}
							for ( const auto & r : rv )
							{
								_eph->add_managed_region(r, r);
								hop_hash_log<trace_heap_summary>::write(
									LOG_LOCATION
									, " pool ", ::base(r), " .. ", ::end(r)
									, " size ", ::size(r)
									, " grow"
								);
							}
							break;
						}
						catch ( const std::bad_alloc & )
						{
							/* probably means that the uuid is in use */
						}
						catch ( const General_exception & )
						{
							/* probably means that the space cannot be allocated */
							throw std::bad_alloc();
						}
					}
				}
				if ( uuid_next == uuid )
				{
					throw std::bad_alloc(); /* no more UUIDs */
				}
			}
		}
	}
	else
	{
		if ( 0 < increment_ )
		{
			if ( _more_region_uuids_size == _more_region_uuids.size() )
			{
				throw std::bad_alloc(); /* max # of regions used */
			}
			const auto hstore_grain_size = std::size_t(1) << (HSTORE_LOG_GRAIN_SIZE);
			auto size = ( (increment_ - 1) / hstore_grain_size + 1 ) * hstore_grain_size;

			auto grown = false;
			{
				const auto old_regions = regions();
				const auto &old_region_list = old_regions.address_map();
				const auto old_list_size = old_region_list.size();
				const auto old_size = region_size(old_region_list);
				_eph->set_managed_regions(dax_manager_->resize_region(old_regions.id(),  _numa_node, old_size + increment_));
				const auto new_region_list = regions().address_map();
				const auto new_size = region_size(new_region_list);
				const auto new_list_size = new_region_list.size();

				if ( old_size <  new_size )
				{
					for ( auto i = old_list_size; i != new_list_size; ++i )
					{
						const auto &r = new_region_list[i];
						_eph->add_managed_region(r, r);
						hop_hash_log<trace_heap_summary>::write(
							LOG_LOCATION
							, " pool ", ::base(r), " .. ", ::end(r)
							, " size ", ::size(r)
							, " grow"
						);
					}
				}
				grown = true;
			}

			if ( ! grown )
			{
				auto uuid = _more_region_uuids_size == 0 ? uuid_ : _more_region_uuids[_more_region_uuids_size-1];
				auto uuid_next = uuid + 1;
				for ( ; uuid_next != uuid; ++uuid_next )
				{
					if ( uuid_next != 0 )
					{
						try
						{
							/* Note: crash between here and "Slot persist done" may cause dax_manager_
							 * to leak the region.
							 */
							auto rv = dax_manager_->create_region(std::to_string(uuid_next), _numa_node, size).address_map();
							{
								auto &slot = _more_region_uuids[_more_region_uuids_size];
								slot = uuid_next;
								persister_nupm::persist(&slot, sizeof slot);
								/* Slot persist done */
							}
							{
								++_more_region_uuids_size;
								persister_nupm::persist(&_more_region_uuids_size, _more_region_uuids_size);
							}
							for ( const auto &r : rv )
							{
								_eph->add_managed_region(r, r);
								hop_hash_log<trace_heap_summary>::write(
									LOG_LOCATION
									, " pool ", ::base(r), " .. ", ::end(r)
									, " size ", ::size(r)
									, " grow"
								);
							}
							break;
						}
						catch ( const std::bad_alloc & )
						{
							/* probably means that the uuid is in use */
						}
						catch ( const General_exception & )
						{
							/* probably means that the space cannot be allocated */
							throw std::bad_alloc();
						}
					}
				}
				if ( uuid_next == uuid )
				{
					throw std::bad_alloc(); /* no more UUIDs */
				}
			}
		}
	}
	return _eph->capacity();
}

void heap_mm::quiesce()
{
	if ( is_crash_consistent() )
	{
	}
	auto eph = dynamic_cast<heap_mr_ephemeral *>(_eph.get());
	if ( eph )
	{
		hop_hash_log<trace_heap_summary>::write(LOG_LOCATION, " size ", ::size(_pool0_heap), " allocated ", eph->allocated());
	}
	_eph->write_hist<trace_heap_summary>(_pool0_heap);
	VALGRIND_DESTROY_MEMPOOL(::base(_pool0_heap));
	VALGRIND_MAKE_MEM_UNDEFINED(::base(_pool0_heap), ::size(_pool0_heap));
	_eph.reset(nullptr);
}

namespace
{
	/* Round up to (ceiling) power of 2, from Hacker's Delight 3-2 */
	std::size_t clp2(std::size_t sz_)
	{
		if ( sz_ != 0 )
		{
			--sz_;
			sz_ |= sz_ >> 1;
			sz_ |= sz_ >> 2;
			sz_ |= sz_ >> 4;
			sz_ |= sz_ >> 8;
			sz_ |= sz_ >> 16;
			sz_ |= sz_ >> 32;
		}
		return sz_ + 1;
	}
}

void heap_mm::alloc(persistent_t<void *> &p_, std::size_t sz_, std::size_t align_)
{
	if ( is_crash_consistent() )
	{
		auto align = clean_align(align_, sizeof(void *));

		/* allocation must be multiple of alignment */
		auto sz = (sz_ + align - 1U)/align * align;

		try
		{
			auto eph = dynamic_cast<heap_mc_ephemeral *>(_eph.get());
			if ( eph && eph->is_crash_consistent() )
			{
				if ( eph->_aspd->is_armed() )
				{
				}
				else if ( eph->_aspk->is_armed() )
				{
				}
				/* Note: order of testing is important. An extend arm+allocate) can occur while
				 * emplace is armed, but not vice-versa
				 */
				else if ( eph->_asx->is_armed() )
				{
					eph->_asx->record_allocation(&persistent_ref(p_), persister_nupm());
				}
				else if ( eph->_ase->is_armed() )
				{
					eph->_ase->record_allocation(&persistent_ref(p_), persister_nupm());
				}
				else
				{
					if ( leak_check )
					{
						PLOG(PREFIX "leaky allocation, size %zu", LOCATION, sz_);
					}
				}
			}

			/* IHeap interface does not support abstract pointers. Cast to regular pointer */
			eph->allocate(p_, sz, align);
			/* We would like to carry the persistent_t through to the crash-conssitent allocator,
			 * but for now just assume that the allocator has modifed p_, and call tick to indicate that.
			 */
			perishable::tick();

			VALGRIND_MEMPOOL_ALLOC(::base(_pool0_heap), p_, sz);
			/* size grows twice: once for aligment, and possibly once more in allocation */
			hop_hash_log<trace_heap>::write(LOG_LOCATION, "pool ", ::base(_pool0_heap), " addr ", p_, " size ", sz_, "->", sz);
			eph->_hist_alloc.enter(sz);
		}
		catch ( const std::bad_alloc & )
		{
			_eph->write_hist<true>(_pool0_heap);
			/* Sometimes lack of space will cause heap to throw a bad_alloc. */
			throw;
		}
	}
	else
	{
		auto align = clean_align(align_, sizeof(void *));

		auto sz = sz_;

		if ( sz < align )
		{
			/* round up only to a power of 2, so Rca_LB will find the element
			 * on free.
			 */
			sz = clp2(sz);
			assert( (sz & (sz - 1)) == 0 );
			/* Allocation must be a multiple of alignment. In the case,
			 * adjust alignment. */
			align = std::max(sizeof(void *), sz);
		}

		/* In any case, sz must be a multiple of alignment. */
		sz = (sz + align - 1U)/align * align;

		try {
			/* Good use of eph */
			_eph->allocate(p_, sz, align);
			/* Note: allocation exception from Rca_LB is General_exception, which does not derive
			 * from std::bad_alloc.
			 */

			VALGRIND_MEMPOOL_ALLOC(::base(_pool0_heap), p_, sz);
			hop_hash_log<trace_heap>::write(LOG_LOCATION, "pool ", ::base(_pool0_full), " addr ", p_, " align ", align_, " -> ", align, " size ", sz_, " -> ", sz);
		}
		catch ( const std::bad_alloc & )
		{
			_eph->write_hist<true>(_pool0_heap);
			/* Sometimes lack of space will cause heap to throw a bad_alloc. */
			throw;
		}
		catch ( const General_exception &e )
		{
			_eph->write_hist<true>(_pool0_heap);
			/* Sometimes lack of space will cause heap to throw a General_exception with this explanation. */
			/* Convert to bad_alloc. */
			if ( e.cause() == string_view("region allocation out-of-space") )
			{
				throw std::bad_alloc();
			}
			throw;
		}
	}
}

void *heap_mm::alloc_tracked(const std::size_t sz_, const std::size_t align_)
{
	if ( is_crash_consistent() )
	{
		auto align = clean_align(align_);
		auto sz = round_up(sz_, align);
		void *p = nullptr;
		_eph->allocate(p, sz, align);
		return p;
	}
	else
	{
		/* alignment: enough for tracked_header prefix, and a power of 2 */
		auto align = clp2(std::max(clean_align(align_), sizeof(tracked_header)));

		/* size: a multiple of alignment */
		auto sz = round_up(sz_ + align, align);

		try {
			void *p = nullptr;
			_eph->allocate(p, sz, align);
			/* Note: allocation exception from Rca_LB is General_exception, which does not derive
			 * from std::bad_alloc.
			 */
	
			VALGRIND_MEMPOOL_ALLOC(::base(_pool0_heap), p, sz);
			hop_hash_log<trace_heap>::write(LOG_LOCATION, "pool ", ::base(_pool0_full), " addr ", p, " align ", align_, " -> ", align, " size ", sz_, " -> ", sz);
			tracked_header *h = new (static_cast<char *>(p) + align - sizeof(tracked_header))
				tracked_header(_eph->debug_level(), &_tracked_anchor, _tracked_anchor._next, sz, align);
			persister_nupm::persist(h, sizeof *h);

			_tracked_anchor._next->_prev = h; /* _prev, need not flush */
			_tracked_anchor._next = h; /* _next, must flush */
			persister_nupm::persist(&_tracked_anchor._next, sizeof _tracked_anchor._next);

#if 0
			PLOG(
				"%s: TH %p prev %p next %p size %zu align %zu"
				, __func__
				, common::p_fmt(h)
				, common::p_fmt(h->_prev)
				, common::p_fmt(h->_next)
				, h->_size
				, h->_align
			);
#endif
			return h + 1;
		}
		catch ( const std::bad_alloc & )
		{
			_eph->write_hist<true>(_pool0_heap);
			/* Sometimes lack of space will cause heap to throw a bad_alloc. */
			throw;
		}
		catch ( const General_exception &e )
		{
			_eph->write_hist<true>(_pool0_heap);
			/* Sometimes lack of space will cause heap to throw a General_exception with this explanation. */
			/* Convert to bad_alloc. */
			if ( e.cause() == string_view("region allocation out-of-space") )
			{
				throw std::bad_alloc();
			}
			throw;
		}
	}
}

void heap_mm::inject_allocation(const void * p, std::size_t sz_)
{
	auto alignment = sizeof(void *);
	sz_ = std::max(sz_, alignment);
	auto sz = (sz_ + alignment - 1U)/alignment * alignment;
	/* NOTE: inject_allocation should take a const void* */
	{
		auto &eph = dynamic_cast<heap_mr_ephemeral &>(*_eph);
		eph.inject_allocation(const_cast<void *>(p), sz);
	}
	VALGRIND_MEMPOOL_ALLOC(::base(_pool0_heap), p, sz);
	hop_hash_log<trace_heap>::write(LOG_LOCATION, "pool ", ::base(_pool0_heap), " addr ", p, " size ", sz);
}

void heap_mm::free(void *&p_, std::size_t sz_)
{
	if ( is_crash_consistent() )
	{
		VALGRIND_MEMPOOL_FREE(::base(_pool0_heap), p_);
		auto sz = _eph->free(p_, sz_);
		hop_hash_log<trace_heap>::write(LOG_LOCATION, "pool ", ::base(_pool0_heap), " addr ", p_, " size ", sz_, "->", sz);
	}
	else
	{
		auto sz = std::max(sz_, sizeof(void *));
		VALGRIND_MEMPOOL_FREE(::base(_pool0_heap), p_);
		hop_hash_log<trace_heap>::write(LOG_LOCATION, "pool ", ::base(_pool0_heap), " addr ", p_, " size ", sz);
		_eph->free(p_, sz);
	}
}

void heap_mm::free_tracked(
	const void *p_
	, std::size_t sz_
)
{
	if ( is_crash_consistent() )
	{
		VALGRIND_MEMPOOL_FREE(::base(_pool0_heap), p_);
		hop_hash_log<trace_heap>::write(LOG_LOCATION, "pool ", ::base(_pool0_heap), " addr ", p_, " size ", sz_);
		return _eph->free_tracked(p_, sz_);
	}
	else
	{
		tracked_header *h = static_cast<tracked_header *>(const_cast<void *>(p_))-1;
		auto align = h->_align;
		/* size: a multiple of alignment */
		auto sz = round_up(sz_ + align, align);
		if ( 3 < _eph->debug_level() )
		{
			PLOG(
				"%s: TH %p prev %p next %p size %zu align %zu"
				, __func__
				, common::p_fmt(h)
				, common::p_fmt(h->_prev)
				, common::p_fmt(h->_next)
				, h->_size
				, h->_align
			);
		}
		h->_next->_prev = h->_prev; /* _prev, need not flush */
		h->_prev->_next = h->_next; /* _next, must flush */
		persister_nupm::persist(&h->_prev->_next, sizeof h->_prev->_next);

		auto p = static_cast<const char *>(p_) - h->_align;
		assert(sz == h->_size);
		VALGRIND_MEMPOOL_FREE(::base(_pool0_heap), p);
		hop_hash_log<trace_heap>::write(LOG_LOCATION, "pool ", ::base(_pool0_heap), " addr ", p, " size ", sz);
		return _eph->free_tracked(p, sz);
	}
}

unsigned heap_mm::percent_used() const
{
	return
		unsigned(
			_eph->capacity()
			? _eph->allocated() * 100U / _eph->capacity()
			: 100U
		);
}

bool heap_mm::is_reconstituted(const void * p_) const
{
	auto eph = dynamic_cast<heap_mr_ephemeral *>(_eph.get());
	return eph && eph->is_reconstituted(p_);
}

impl::allocation_state_pin *heap_mm::aspd() const
{
	auto &eph = dynamic_cast<heap_mc_ephemeral &>(*_eph);
	return eph._aspd;
}

impl::allocation_state_pin *heap_mm::aspk() const
{
	auto &eph = dynamic_cast<heap_mc_ephemeral &>(*_eph);
	return eph._aspk;
}

char *heap_mm::pin_data_get_cptr() const
{
	/*
	 * Should only be called by monitor_pin, and then only when heap is a
	 * consistent type. If not crash consistent, call is an error and
	 * dynamic_cast will throw an exception.
	 */
	auto &eph = dynamic_cast<heap_mc_ephemeral &>(*_eph);
	assert( eph._aspd->is_armed() );
	return eph._aspd->get_cptr();
}

char *heap_mm::pin_key_get_cptr() const
{
	/*
	 * Should only be called by monitor_pin, and then only when heap is a
	 * consistent type. If not crash consistent, call is an error and
	 * dynamic_cast will throw an exception.
	 */
	auto &eph = dynamic_cast<heap_mc_ephemeral &>(*_eph);
	assert( eph._aspk->is_armed() );
	return eph._aspk->get_cptr();
}

void heap_mm::pin_data_arm(
	cptr & cptr_
) const
{
	if ( auto eph = dynamic_cast<heap_mc_ephemeral *>(_eph.get()) )
	{
		eph->_aspd->arm(cptr_, persister_nupm());
	}
}

void heap_mm::pin_key_arm(
	cptr & cptr_
) const
{
	if ( auto eph = dynamic_cast<heap_mc_ephemeral *>(_eph.get()) )
	{
		eph->_aspk->arm(cptr_, persister_nupm());
	}
}

void heap_mm::pin_data_disarm() const
{
	if ( auto eph = dynamic_cast<heap_mc_ephemeral *>(_eph.get()) )
	{
		eph->_aspd->disarm(persister_nupm());
	}
}

void heap_mm::pin_key_disarm() const
{
	if ( auto eph = dynamic_cast<heap_mc_ephemeral *>(_eph.get()) )
	{
		eph->_aspk->disarm(persister_nupm());
	}
}

bool heap_mm::is_crash_consistent() const { return _eph->is_crash_consistent(); }
bool heap_mm::can_reconstitute() const { return _eph->can_reconstitute(); }

void heap_mm::extend_arm() const
{
	if ( auto eph = dynamic_cast<heap_mc_ephemeral *>(_eph.get()) )
	{
		eph->_asx->arm(persister_nupm());
	}
}

void heap_mm::extend_disarm() const
{
	if ( auto eph = dynamic_cast<heap_mc_ephemeral *>(_eph.get()) )
	{
		eph->_asx->disarm(persister_nupm());
	}
}

void heap_mm::emplace_arm() const
{
	if ( auto eph = dynamic_cast<heap_mc_ephemeral *>(_eph.get()) )
	{
		eph->_ase->arm(persister_nupm());
	}
}

void heap_mm::emplace_disarm() const
{
	if ( auto eph = dynamic_cast<heap_mc_ephemeral *>(_eph.get()) )
	{
		eph->_ase->disarm(persister_nupm());
	}
}
