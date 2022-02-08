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

#include "hstore_config.h"
#include "hop_hash_log.h"
#include "logging.h"
#include "perishable.h"
#include "perishable_expiry.h"

#include <execinfo.h>

#include <cstddef> /* uint64_t */
#include <exception> /* uncaught_excpetion(s) */


bool perishable::_enabled = false;
std::uint64_t perishable::_initial = 0;
std::uint64_t perishable::_time_to_live = 0;

void perishable::check()
{
	if (
#if __cplusplus < 201703L
		std::uncaught_exception()
#else
		0 != std::uncaught_exceptions()
#endif
	)
	{
		PWRN(PREFIX_STATIC "TTL 0 during exception", LOCATION_STATIC);
	}
	else
	{
		throw perishable_expiry{__LINE__};
	}
}

auto perishable::make_syndrome() -> syndrome
{
	syndrome sy(100);
	auto sz = unsigned(::backtrace(&sy[0], int(sy.size())));
	sy.resize(sz);
	return sy;
}

bool perishable::less::operator()(const syndrome &a, const syndrome &b) const
{
	if ( a.size() < b.size() ) return true;
	if ( b.size() < a.size() ) return false;
	auto m = std::mismatch(a.begin(), a.end(), b.begin());
	if ( m.first == a.end() ) return false;
	return *m.first < *m.second;
}

perishable::syndrome_map perishable::seen{};

bool perishable::tick()
{
	if ( _enabled )
	{
		if ( _time_to_live == use_syndrome )
		{
			auto sy = make_syndrome();
#if 0
			return ! seen.insert(sy).second;
#else
			auto seen_count = ++seen[sy];
			if ( seen_count < 3 )
			{
				PWRN(PREFIX_STATIC " new perishable syndrome", LOCATION_STATIC);
				check();
				return false;
			}
			return true;
#endif
		}

		if ( _time_to_live == 0 )
		{
			return false;
		}
		if ( --_time_to_live == 0 )
		{
			check();
			return false;
		}
	}
	return true;
}

void perishable::reset(std::uint64_t n)
{
	_initial = n;
	_time_to_live = n;
}

void perishable::enable(bool e)
{
	_enabled = e;
}

void perishable::test()
{
	if ( _enabled && _time_to_live == 0 )
	{
		check();
	}

}

void perishable::report()
{
	if ( _initial != 0 )
	{
		hop_hash_log<true>::write(LOG_LOCATION_STATIC, "perishable: ", _time_to_live
			, " of ", _initial, " ticks left");
	}
}
