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


#ifndef MCAS_HSTORE_ATOMIC_CTL_H_
#define MCAS_HSTORE_ATOMIC_CTL_H_

#include "alloc_key.h" /* AK_FORMAL */
#include "construction_mode.h"
#include "lock_state.h"
#include "mod_control.h"
#include <api/kvstore_itf.h> /* component */
#include <common/perf/tm_fwd.h>

#include <common/string_view.h>
#include <tuple> /* tuple_element */
#include <type_traits> /* is_base_of */
#include <vector>

namespace impl
{
	template <typename Value>
		struct persist_atomic;

	template <typename Table>
		struct persist_atomic_controller
			: private std::allocator_traits<typename Table::allocator_type>::template rebind_alloc<mod_control>
		{
		private:
			using string_view = common::string_view;
			using table_type = Table;
			using allocator_type =
				typename std::allocator_traits<typename table_type::allocator_type>::template rebind_alloc<mod_control>;

			using persist_type = persist_atomic<table_type>;
			persist_type *_persist; /* persist_atomic is a bad name. Should be a noun. */
#if 0
			bool _tick_expired;
#endif
			struct update_finisher
			{
			private:
				impl::persist_atomic_controller<table_type> &_ctlr;
			public:
				update_finisher(impl::persist_atomic_controller<table_type> &ctlr_);
				~update_finisher() noexcept(! TEST_HSTORE_PERISHABLE);
			};

			void do_update(TM_FORMAL0);
			void update_finish();
			void do_replace();
			void do_swap();
			void do_finish();
#if 0
			/* Helpers for the perishable test, to avoid an exception in the finish_update destructor */
			void tick_expired() { _tick_expired = true; }
			bool is_tick_expired() { auto r = _tick_expired; _tick_expired = false; return r; }
#endif
			void persist_range(const void *first_, const void *last_, const char *what_);
		public:
			persist_atomic_controller(
				persist_type &persist_
				, allocator_type al_
				, construction_mode mode_
			);
			persist_atomic_controller(const persist_atomic_controller &) = delete;
			persist_atomic_controller& operator=(const persist_atomic_controller &) = delete;

		private:
			void do_op(TM_FORMAL0);
		public:
			template <typename IT> /* *IT shall be a component::IKVStore::Operation * */
				void enter_update(
					AK_FORMAL
					TM_FORMAL
					typename table_type::allocator_type al_
					, table_type *map_
					, lock_state lock
					, string_view key
					, IT first
					, IT last
				);

			void enter_replace(
				AK_FORMAL
				TM_FORMAL
				typename table_type::allocator_type al
				, table_type *map_
				, lock_state lock
				, string_view key
				, const char *data
				, std::size_t data_len
				, std::size_t zeros_extend
				, std::size_t alignment
			);

			using mt = typename table_type::mapped_type;

			void enter_swap(
				mt &d0
				, mt &d1
			);

			friend struct persist_atomic_controller<table_type>::update_finisher;
	};
}

#include "persist_atomic_controller.tcc"

#endif
