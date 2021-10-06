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


#ifndef MCAS_HSTORE_PERSIST_CONTROLLER_H
#define MCAS_HSTORE_PERSIST_CONTROLLER_H

#include "alloc_key.h" /* AK_FORMAL */
#include "construction_mode.h"
#include "hop_hash_log.h"
#include "persist_data.h"
#include "test_flags.h" /* TEST_HSTORE_PERISHABLE */

#include <boost/iterator/transform_iterator.hpp>

#include <cassert>
#include <cstddef> /* size_t */
#include <functional> /* mem_fn */
#include <type_traits>

/* Controller for persistent data.
 * Modification of persistent data (except for writes to persist_data::_sc)
 * goes through this class. Ideally this should also get writes to persist_data::_sc.
 */

struct perishable_expiry;

namespace impl
{
	template <typename Allocator, typename SizeChange>
		struct persist_size_change
			: public SizeChange
		{
		private:
			persist_map_controller<Allocator> *_pc;
		public:
			persist_size_change(persist_map_controller<Allocator> &pc_)
				: SizeChange(pc_.get_size_control())
				, _pc(&pc_)
			{
				_pc->size_destabilize();
			}
			persist_size_change(const persist_size_change &) = delete;
			persist_size_change& operator=(const persist_size_change &) = delete;
			~persist_size_change() noexcept(! TEST_HSTORE_PERISHABLE)
			{
				if ( ! perishable_expiry::is_current() )
				{
					/* Note: change and size_stabilize are separate calls which could be combined. */
					this->SizeChange::change();
					_pc->size_stabilize();
				}
			}
		};

	template <typename Allocator>
		struct persist_map_controller
			: public Allocator
		{
		private:
			using allocator_type = Allocator;
			using persist_data_t = persist_map<allocator_type>;
			using value_type = typename allocator_type::value_type;
		public:
			using size_type = std::size_t;
			using bix_t = std::size_t; /* sufficient for all bucket indexes */
			using bucket_aligned_t = bucket_aligned<hash_bucket<value_type>>;
			using content_t = content<value_type>;
			static constexpr auto _segment_capacity = persist_data_t::_segment_capacity;
			static constexpr unsigned log2_base_segment_size =
				persist_data_t::log2_base_segment_size;
			static constexpr bix_t base_segment_size = persist_data_t::base_segment_size;
		private:
			using bucket_allocator_t = typename persist_data_t::bucket_allocator_t;
			persist_data_t *_persist;
			std::size_t _bucket_count_cached;

			void persist_segment_table(); /* Flush the bucket pointers (*_b) */
			void persist_internal(
				const void *first
				, const void *last
				, const char *what
			);
			auto bucket_count_uncached() -> size_type
			{
				return base_segment_size << (segment_count_actual().value_not_stable() - 1U);
			}

		public:
			explicit persist_map_controller(
				AK_FORMAL
				const allocator_type &av, persist_data_t *persist, construction_mode mode);

			persist_map_controller(const persist_map_controller &) = delete;
			persist_map_controller(persist_map_controller &&) noexcept = default;
			auto operator=(
				const persist_map_controller &
			) -> persist_map_controller & = delete;

			auto resize_prolog(AK_FORMAL0) -> bucket_aligned_t *;
			auto resize_restart_prolog() -> bucket_aligned_t *;
			void resize_interlog();
			void resize_epilog();

			void size_stabilize();
			void size_destabilize();

			void persist_owner(
				const owner &b
				, const char *what = "bucket_owner"
			);
			void persist_content(
				const content_t &b
				, const char *what = "bucket_content"
			);
			void persist_segment_count(); /* Flush the bucket pointer count (_count) */
			void persist_size();
			void persist_existing_segments(const char *what = "old segments");
			void persist_new_segment(const char *what = "new segments");
			void em_record_owner_addr_and_bitmask(
				persistent_atomic_t<owner::value_type> *pmask_
				, owner::value_type mask_
			)
			{
				auto pe = static_cast<allocator_type *>(this);
				_persist->ase()
					->em_record_owner_addr_and_bitmask(
						pmask_
						, mask_
						, *pe
					);
			}
			void er_record_owner_addr_and_bitmask(
				persistent_atomic_t<owner::value_type> *pmask_
				, owner::value_type mask_
			)
			{
				auto pe = static_cast<allocator_type *>(this);
				_persist->ase()
					->er_record_owner_addr_and_bitmask(
						pmask_
						, mask_
						, *pe
					);
			}
			void record_segment_count_addr_and_target_value(
				segment_count *psegment_count_
				, segment_layout::six_t segment_count_expected_
			)
			{
				auto pe = static_cast<allocator_type *>(this);
				_persist->asx()
					->record_segment_count_addr_and_target_value(
						psegment_count_
						, segment_count_expected_
						, *pe
					);
			}
			void clear_allocation_doubt()
			{
				_persist->ase().clear_allocation_doubt(static_cast<allocator_type *>(this));
				_persist->asx()->clear_allocation_doubt(static_cast<allocator_type *>(this));
			}

			auto segment_count_actual() const
			{
				return _persist->_segment_count.actual();
			}
			std::size_t segment_count_specified() const
			{
				return _persist->_segment_count.specified();
			}

			auto size_unstable() const /* debugging only */
			{
				return _persist->_size_control.value_not_stable();
			}

			std::size_t size() const
			{
				return _persist->_size_control.value();
			}

			size_control &get_size_control()
			{
				return _persist->_size_control;
			}

			/* NOTE: this function returns an non-const iterator over _persist data,
			 * allowing successive accesses to *_persist without intervening "ticks."
			 * The user needs to provide intervening ticks.
			 */
			auto bp_src()
			{
				return boost::make_transform_iterator(
					/* original iterator */
					_persist->_sc
					/* transform function applied to that each item returned from the original iterator */
					, [] ( const typename persist_data_t::segment_control &sc ) -> auto
					{
						return sc.bp;
					}
				);
			}
			bool is_size_stable() const;
			void size_set(std::size_t n);

			auto bucket_count() const -> size_type
			{
				return _bucket_count_cached;
			}

			auto max_bucket_count() const -> size_type
			{
				return base_segment_size << (_segment_capacity - 1U);
			}

			auto mask() const { return bucket_count() - 1U; }

			auto distance_wrapped(
				bix_t first, bix_t last
			) -> unsigned
			{
				return
					unsigned
					(
						(
							last < first
							? last + bucket_count()
							: last
						) - first
					);
			}
		};
}

#include "persist_map_controller.tcc"

#endif
