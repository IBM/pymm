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


#ifndef MCAS_HSTORE_PERISHABLE_H
#define MCAS_HSTORE_PERISHABLE_H

#include <cstdint>
#include <limits>
#include <map>
#include <vector>

struct perishable
{
private:
	using syndrome = std::vector<void *>;
	struct less
	{
		bool operator() (const syndrome &a, const syndrome &b) const;
	};
	using syndrome_map = std::map<syndrome, std::uint64_t, less>;
	static syndrome_map seen;
	static syndrome make_syndrome();
	static bool _enabled;
	static std::uint64_t _initial;
	static std::uint64_t _time_to_live;
	static void check();
public:
	static auto constexpr use_syndrome = std::numeric_limits<std::uint64_t>::max();
	static void reset(std::uint64_t n);
	static void enable(bool e);
	static bool tick();
	static void test(); /* like tick, but without the decrement */
	static void report();
};

#endif
