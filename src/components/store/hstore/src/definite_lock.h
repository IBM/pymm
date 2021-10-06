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

#ifndef MCAS_DEFINITE_LOCK_H
#define MCAS_DEFINITE_LOCK_H

#include "hstore_config.h"

#include "alloc_key.h" /* AK_ACTUAL */
#include "heap_access.h"
#include "hstore_alloc_type.h" /* hstore_alloc_type */
#include "hstore_nupm_types.h" /* Persister */
#include "key_not_found.h"
#include "is_locked.h"
#include "monitor_pin.h"

#include <common/perf/tm.h>

#include <stdexcept>
#include <tuple>

class Exception;

template <typename Table, typename Allocator>
	struct definite_lock
	{
		using table_t = Table;
	private:
		typename table_t::iterator _it;
	public:
		template <typename K>
			definite_lock(
				AK_ACTUAL
				TM_ACTUAL
				table_t & map_
				, const K &key_
				, Allocator al_
			)
				: _it(map_.find(TM_REF key_))
			{
				if ( _it == map_.end() )
				{
					throw impl::key_not_found{};
				}

				auto &d = data();

				if ( ! d.lockable() )
				{
					/* Allocating space for a lockable value is tricky.
					 *
					 * allocator_cc (crash-consistent allocator):
					 *   see notes in as_pin.h
					 *
					 * allocatpr_rc (reconstituting allocator):
					 *   TBD
					 */

					/* convert value to lockable */
					monitor_pin<hstore_alloc_type<Persister>::heap_alloc_access_type> mp(d, al_.pool(), al_.pool()->pin_control_data());
					/* convert d to immovable data */
					d.pin(AK_REF mp.get_cptr(), al_);
				}

				if ( ! d.try_lock_exclusive() )
				{
					throw impl::is_locked{};
				}
			}

		~definite_lock()
		{
			try
			{
				if ( ! perishable_expiry::is_current() )
				{
					/* release lock */
					const auto &d = data();
					d.unlock_exclusive();
				}
			}
			catch ( const Exception & )
			{
			}
			catch ( const std::exception & )
			{
			}
		}

		auto &mapped() const
		{
			return _it->second;
		}

		auto &data() const
		{
			auto &m = mapped();
			return std::get<0>(m);
		}
	};

#endif
