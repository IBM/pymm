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

#include <common/perf/duration_stat.h>

#include <common/env.h>

#include <cmath> /* sqrt */
#include <limits> /* numeric_limits */
#include <ostream>
#include <stdexcept> /* domain_error */

using namespace common;
using namespace perf;

bool duration_stat::_clock_enabled{env_value("MCAS_DURATION_CLOCK_ENABLED", false)};

duration_stat::duration_stat()
	: _duration{}
	, _delta{}
	, _dur_sq{}
	, _count{}
	, _min(std::numeric_limits<duration_type::rep>::max())
	, _max(0)
{}

double duration_stat::mean() const
{
	return static_cast<double>(_duration)/static_cast<double>(_count);
}

double duration_stat::variance() const
{
	auto m = mean();
	return static_cast<double>(_dur_sq)/static_cast<double>(_count) - m*m;
}

double duration_stat::stddev() const
{
	return std::sqrt(variance());
}

auto duration_stat::count() const
	-> count_type
{
	return _count;
}

double duration_stat::cv() const
{
	return stddev()/mean();
}

double duration_stat::stddev_or_zero() const
{
	return _count == 0 ? 0.0 : stddev();
}

double duration_stat::cv_or_zero() const
{
	return _count == 0 ? 0.0 : cv();
}

double duration_stat::mean_or_zero() const
{
	return _count == 0 ? 0.0 : mean();
}

std::chrono::nanoseconds::rep duration_stat::sum_durations_ns() const
{
	return std::chrono::duration_cast<std::chrono::nanoseconds>(clock_type::duration(_duration)).count();
}

double duration_stat::sum_durations_sec() const
{
	using period = clock_type::duration::period;
	return double(this->_duration)*period::num/period::den;
}

double duration_stat::mean_durations_sec() const
{
	using period = clock_type::duration::period;
	return double(this->mean_or_zero())*period::num/period::den;
}

double duration_stat::durations_sec_min() const
{
	using period = clock_type::duration::period;
	return double(this->_min)*period::num/period::den;
}

double duration_stat::durations_sec_max() const
{
	using period = clock_type::duration::period;
	return double(this->_max)*period::num/period::den;
}

unsigned long long duration_stat::sum_durations_ns_squared() const
{
	/*
	 * convert the sum of durations squared to nanoseconds squared.
	 * We use sqrt followed by square, to convert from high_requlution_clock::rep to
	 * nanoseconds::rep, but we could use the square of the ratios of the periods to
	 * avoid the trip through sqrt and square.
	 */
	using dpd = duration_type::period;
	using npd = std::chrono::nanoseconds::period;
	auto ns2 = static_cast<double>(_dur_sq) * static_cast<double>(dpd::num * dpd::num * npd::den * npd::den) / static_cast<double>(dpd::den * dpd::den * npd::num * npd::num);
	return static_cast<unsigned long long>(ns2);
}

std::ostream &common::perf::operator<<(std::ostream &o, const duration_stat &d)
{
	return o << "total " << d.sum_durations_sec() << " min " << d.durations_sec_min() << " mean " << d.mean_durations_sec() << " max " << d.durations_sec_max() << " cv " << d.cv_or_zero() << " ct " << d.count();
}
