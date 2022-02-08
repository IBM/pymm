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

#ifndef _MCAS_REGION_MEMORY_NUMA_PIN_H_
#define _MCAS_REGION_MEMORY_NUMA_PIN_H_

#include "region_memory_numa.h"

#include <numa.h> /* bitmask */
#include <cstddef> /* size_t */

/* Note: a tempated type, e.g. memory_pin<typename Allocation_owner> would be more general */
struct region_memory_numa_pin
	: public region_memory_numa
{
	int lock_rc;
	region_memory_numa_pin(unsigned debug_level, std::size_t size, const bitmask *nodes, bool do_pin);
	region_memory_numa_pin(const region_memory_numa &) = delete;
	region_memory_numa_pin &operator=(const region_memory_numa &) = delete;
	~region_memory_numa_pin() override;
};

#endif
