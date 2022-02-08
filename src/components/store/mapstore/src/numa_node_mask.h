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

#ifndef _MCAS_NUMA_NODE_MASK__H_
#define _MCAS_NUMA_NODE_MASK__H_

#include <numa.h> /* bitmask */
#include <common/string_view.h>
#include <gsl/pointers> /* not_null */
#include <memory> /* unique_ptr */

struct numa_node_mask_dtor
{
	void operator()(bitmask *b) { numa_free_nodemask(b); }
};

struct numa_node_mask
{
	numa_node_mask(common::string_view mask_)
		: _mask(std::unique_ptr<bitmask, numa_node_mask_dtor>(mask_not_null(std::string(mask_))))
	{
	}
	numa_node_mask(const bitmask *b)
		: _mask(std::unique_ptr<bitmask, numa_node_mask_dtor>(numa_allocate_nodemask()))
	{
		copy_bitmask_to_bitmask(const_cast<bitmask *>(b), get());
	}
	const bitmask *get() const { return _mask.get().get(); }
	bitmask *get() { return _mask.get().get(); }
	/* get first 64 bits in mask */
	uint64_t get64()
	{
		uint64_t r = 0;
		for ( unsigned i = 0; i != 64; ++i )
		{
			if ( numa_bitmask_isbitset(get(), i) )
			{
				r |= ( 1U << i );
			}
		}
		return r;
	}
private:
	static bitmask *mask_not_null(common::string_view mask_)
	{
		auto m = numa_parse_nodestring(std::string(mask_).c_str());
		return m ? m : numa_allocate_nodemask();
	}
	gsl::not_null<std::unique_ptr<bitmask, numa_node_mask_dtor>> _mask;
};

#endif
