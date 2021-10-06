/*
   Copyright [2017-2019] [IBM Corporation]
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
#ifndef _mcas_HSTORE_TEST_PROFILER_H_
#define _mcas_HSTORE_TEST_PROFILER_H_

#if defined HAS_PROFILER
#include <gperftools/profiler.h> /* Alas, no __has_include until C++17 */
#else
namespace
{
	int ProfilerStart(const char *) { return -1; }
	void ProfilerStop() {}
}
#endif

#include <cstdlib> /* getenv */
#include <string>

struct profiler
{
private:
	bool _run;
public:
	profiler(const std::string &s)
		: _run(bool(std::getenv("PROFILE")))
	{
		if ( _run )
		{
			ProfilerStart(s.c_str());
		}
	}
	~profiler()
	{
		if ( _run )
		{
			ProfilerStop();
		}
	}
};

#endif
