/*
   Copyright [2019-2021] [IBM Corporation]
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


#ifndef MCAS_MONITOR_EMPLACE_H
#define MCAS_MONITOR_EMPLACE_H

#include "hstore_config.h"
#include "perishable_expiry.h"
#include "test_flags.h"
#include "logging.h"

template <typename Allocator>
	struct monitor_emplace
	{
	private:
		Allocator _a;
	public:
		monitor_emplace(const Allocator &a_)
			: _a(a_)
		{
			if ( _a.pool()->is_crash_consistent() )
			{
				_a.emplace_arm();
			}
		}
		~monitor_emplace() noexcept(! TEST_HSTORE_PERISHABLE)
		{
			if ( ! perishable_expiry::is_current() )
			{
				if ( _a.pool()->is_crash_consistent() )
				{
					_a.emplace_disarm();
				}
			}
		}
	};

#endif
