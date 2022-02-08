/*
   Copyright [2017-2020] [IBM Corporation]
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


#ifndef MCAS_HSTORE_KV_TYPES_H_
#define MCAS_HSTORE_KV_TYPES_H_

#include "hstore_config.h"

#include "persist_fixed_string.h"

#include <api/kvstore_itf.h>
#include <common/utils.h> /* epoch_now */
#include <common/time.h>
#include <tuple>

namespace impl
{

inline common::tsc_time_t epoch_to_tsc(common::epoch_time_t e)
{
  return e;
}

inline common::epoch_time_t tsc_to_epoch(common::tsc_time_t t)
{
  return t.to_epoch();
}

inline common::tsc_time_t tsc_now()
{
  return epoch_to_tsc(common::epoch_now());
}

}

template <typename Deallocator>
	struct hstore_kv_types
	{
		using dealloc_type = Deallocator;
		using key_type = persist_fixed_string<char, 24, dealloc_type>;
		using mapped_type =
			std::tuple<
				persist_fixed_string<char, 24, dealloc_type>
#if ENABLE_TIMESTAMPS
				, common::tsc_time_t
#endif
			>;
	};

#endif
