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

#include "region_memory_numa.h"

#include <common/logging.h>
#include <numa.h> /* numa_alloc_onnode, numa_free */
#include <cstring> /* memset */

namespace
{
	/* scoped numa "binding" */
	struct membind
	{
		membind(const bitmask *node_mask_)
			: old(numa_get_membind())
		{
			numa_set_membind(const_cast<bitmask *>(node_mask_));
		}
		membind(const membind &) = delete;
		membind &operator=(const membind &) = delete;
		~membind()
		{
			numa_set_membind(old);
			numa_free_nodemask(old);
		}
	private:
		bitmask *old;
	};

	void *alloc_on_numa(std::size_t size_, const bitmask *node_mask_)
	{
		membind m(node_mask_);
		auto p = numa_alloc(size_);
		return p;
	}
}

region_memory_numa::region_memory_numa(
	unsigned debug_level_
	, std::size_t size_
	, const bitmask *node_mask_
)
	: region_memory(debug_level_, alloc_on_numa(size_, node_mask_), size_)
{
}

region_memory_numa::~region_memory_numa()
{
	if ( iov_base )
	{
        /* github 185: Clear memory on pool deletion */
		CFLOGM(0, "CLEAR {},{:x}", iov_base, iov_len);
        std::memset(iov_base, 0, iov_len);

		CFLOGM(1, "freeing region memory ({},{})", iov_base, iov_len);
		numa_free(iov_base, iov_len);
	}
}
