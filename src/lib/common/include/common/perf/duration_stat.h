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

#ifndef MCAS_COMMON_PERF_DURATION_STAT_H
#define MCAS_COMMON_PERF_DURATION_STAT_H

#include <atomic>
#include <cstdint> /* uint64_t */
#include <chrono>
#include <ostream>
#include <stdexcept> /* domain_error */

#include <cassert>
#include <iostream>

namespace common
{
	namespace perf
	{
		struct duration_stat
		{
		private:
			using clock_type = std::chrono::steady_clock;
		public:
			using duration_type = clock_type::duration;
			using time_point_type = clock_type::time_point;
			using count_type = std::uint64_t;
		private:
			using rep_type = std::uint64_t;
			std::atomic<duration_type::rep> _duration;
			std::atomic<duration_type::rep> _delta; /* partial time incurred */
			std::atomic<rep_type>           _dur_sq;
			std::atomic<count_type>         _count;
			std::atomic<duration_type::rep> _min;
			std::atomic<duration_type::rep> _max;
		private:
			double mean() const;
			double variance() const;
			double stddev() const;
			double cv() const;

			/* Absolute (or "active") versions */
			static time_point_type a_now()
			{
				return clock_type::now();
			}

			void a_charge(const duration_type &d)
			{
				_delta += d.count();
			}

			void a_record(const duration_type &d)
			{
				auto c = d.count() + _delta;
				_delta = 0;
				_duration += c;
				_dur_sq += static_cast<count_type>(c * c);
				if ( c < _min ) { _min = c; }
				if ( _max < c ) { _max = c; }
				++_count;
			}

			time_point_type a_record(const time_point_type &s)
			{
				auto n = this->now();
				auto d = n-s;
				if ( n < s )
				{
std::cerr << "duration_stat: negative duration: then " << s.time_since_epoch().count() << " now " << n.time_since_epoch().count() << "\n";
assert(0);
					throw std::domain_error{"duration_stat: negative duration"};
				}
				record(d);
				return n;
			}

			/* Inactive functions, for use when disabled */
			static time_point_type i_now()
			{
				return time_point_type{};
			}

			void i_charge(const duration_type &)
			{
			}

			void i_record(const duration_type &)
			{
			}

			time_point_type i_record(const time_point_type &)
			{
				return time_point_type{};
			}
		public:
			duration_stat();

			static bool _clock_enabled;

			static time_point_type now()
			{
				return _clock_enabled ? a_now() : i_now();
			}

			void charge(const duration_type &d)
			{
				return _clock_enabled ?  a_charge(d) : i_charge(d);
			}

			void record(const duration_type &d)
			{
				return _clock_enabled ?  a_record(d) : i_record(d);
			}

			time_point_type record(const time_point_type &s)
			{
				return _clock_enabled ? a_record(s) : i_record(s);
			}

			count_type count() const;

			double mean_or_zero() const;

			double stddev_or_zero() const;
			double cv_or_zero() const;

			std::chrono::nanoseconds::rep sum_durations_ns() const;
			double sum_durations_sec() const;
			double mean_durations_sec() const;
			double durations_sec_min() const;
			double durations_sec_max() const;

			unsigned long long sum_durations_ns_squared() const;
			operator bool() const { return _count != 0; }
		};

		std::ostream &operator<<(std::ostream &o, const duration_stat &d);
	}
}

#endif
