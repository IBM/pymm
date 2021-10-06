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

#ifndef MCAS_COMMON_PERF_TIMER_SPLIT_H
#define MCAS_COMMON_PERF_TIMER_SPLIT_H

#include <chrono> /* time_point */

namespace common
{
	namespace perf
	{
		/*
		 * This struct is intended to time its entire lifetime, splitting the
		 * various phases but never losing any of the time. But the last piece
		 * will be lost because the struct has no idea *where* to report the time;
		 * it is passive in that respect. (See timer_to_exit for a struct which
		 * knows where to report time.)
		 */
		struct timer_split
		{
			using clock_t = std::chrono::steady_clock;
		private:
			clock_t::time_point _split_time;
		public:
			timer_split()
				: _split_time{clock_t::now()}
			{}
			virtual ~timer_split() = default;
			timer_split(const timer_split &) = delete;
			timer_split(timer_split &&) = default;
			timer_split& operator=(const timer_split &) = delete;
			timer_split& operator=(timer_split &&) = default;
			clock_t::duration split_duration(clock_t::time_point t)
			{
				auto d = t - _split_time;
				_split_time = t;
				return d;
			}
			clock_t::duration split_duration()
			{
				return this->split_duration(clock_t::now());
			}
			clock_t::time_point base() const { return _split_time; }
		};
	}
}

#endif
