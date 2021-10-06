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

#ifndef MCAS_HSTORE_MONITOR_EXTEND_H_
#define MCAS_HSTORE_MONITOR_EXTEND_H_

#include "hstore_config.h"
#include <common/logging.h>

template <typename Allocator>
	struct monitor_extend
	{
	private:
		Allocator _a;
	public:
		monitor_extend(const Allocator &a_)
			: _a(a_)
		{
#if HSTORE_TRACE_EXTEND
			PLOG(PREFIX "ctor", LOCATION);
#endif
			if ( _a.pool()->is_crash_consistent() )
			{
				_a.extend_arm();
			}
		}
		~monitor_extend()
		{
#if HSTORE_TRACE_EXTEND
			PLOG(PREFIX "dtor", LOCATION);
#endif
			if ( _a.pool()->is_crash_consistent() )
			{
				_a.extend_disarm();
			}
		}
	};

#endif
