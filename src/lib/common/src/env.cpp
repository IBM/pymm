/*
   Copyright [2013-2021] [IBM Corporation]
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

#include <common/env.h>

#include <cstdlib> /* getenv, strtod, strtoul */
#include <cstring> /* strlen */
#include <iostream> /* cout */
#include <limits>

namespace common
{
	template <>
		double env_value<double>(const char *const env_key, double dflt)
		{
			const char *env_str = std::getenv(env_key);
			if ( env_str )
			{
				char *endptr = nullptr;
				double env_value = std::strtod(env_str, &endptr);
				if ( endptr != env_str + std::strlen(env_str) )
				{
					std::cerr << "For key '" << env_key << "', value '" << env_str << "' is malformed, and ignored\n";
					goto fail;
				}
				dflt = env_value;
			}
		fail:
			return dflt;
		}

	template <>
		unsigned long env_value<unsigned long>(const char *const env_key, unsigned long dflt)
		{
			const char *env_str = std::getenv(env_key);
			if ( env_str )
			{
				char *endptr = nullptr;
				unsigned long env_value = std::strtoul(env_str, &endptr, 0);
				if ( endptr != env_str + std::strlen(env_str) )
				{
					std::cerr << "For key '" << env_key << "', value '" << env_str << "' is malformed, and ignored\n";
					goto fail;
				}
				if ( std::numeric_limits<unsigned long>::max() < env_value )
				{
					std::cerr << "For key '" << env_key << "', value " << " exceeds " << std::numeric_limits<unsigned>::max() << ", and is ignored\n";
					goto fail;
				}
				dflt = env_value;
			}
		fail:
			return dflt;
		}

	template <>
		unsigned env_value<unsigned>(const char *const env_key, unsigned dflt)
		{
			return unsigned(env_value<unsigned long>(env_key, dflt));
		}

	template <>
		bool env_value<bool>(const char *const env_key, bool dflt)
		{
			return env_value(env_key, unsigned(dflt));
		}

	template <>
		const char *env_value<const char *>(const char *const env_key, const char *dflt)
		{
			const char *env_str = std::getenv(env_key);
			return env_str ? env_str : dflt;
		}
}
