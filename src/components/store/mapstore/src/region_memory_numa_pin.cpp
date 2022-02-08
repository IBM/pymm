/*
  Copyright [2021] [IBM Corporation]
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

#include "region_memory_numa_pin.h"

#include <common/logging.h>
#include <sys/mman.h> /* mlock, munlock */

region_memory_numa_pin::region_memory_numa_pin(
	unsigned debug_level_
	, std::size_t size_
	, const bitmask *numa_node_mask_
	, bool do_pin_
)
	: region_memory_numa(debug_level_, size_, numa_node_mask_)
	, lock_rc(do_pin_ ? ::mlock(iov_base, iov_len) : -1)
{}

region_memory_numa_pin::~region_memory_numa_pin()
{
	if ( 0 == lock_rc )
	{
		CFLOGM(1, "unpinning region memory ({},{})", iov_base, iov_len);
		::munlock(iov_base, iov_len);
	}
}
