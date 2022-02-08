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

#include <common/perf/timer_to_exit.h>

#include <common/perf/duration_stat.h>
#include <common/perf/timer_split.h>

using namespace common;
using namespace perf;

/* For the "root" tte, pass a reference to a null timer_to_exit as pe_ */
timer_to_exit::timer_to_exit(const timer_to_exit *&pa_, timer_split &tm_, duration_stat &st_)
	: _tm(&tm_)
	, _st(&st_)
	, _pp(&pa_) /* where we got our parent pointer (points to a function argument parent pointer, or root timer_to_exit* if this is root) */
	, _pa(pa_) /* value of that parent pointer */
{
	pa_ = this;
}

timer_to_exit::timer_to_exit(const timer_to_exit *&pa_, duration_stat &st_)
	: _tm(pa_->_tm)
	, _st(&st_)
	, _pp(&pa_) /* where we got our parent pointer (points to a function argument parent pointer, or root timer_to_exit* if this is root) */
	, _pa(pa_)
{
	pa_ = this;
	/* If there is a parent, charge time-to-now to the parent */
	if ( _pa )
	{
		_pa->charge();
	}
}

timer_to_exit::timer_to_exit(timer_to_exit &&t_)
	: _tm(t_._tm)
	, _st(t_._st)
	, _pp(t_._pp)
	, _pa(t_._pa)
{
	/* If t_ has a durations_stat (and it should), give time until now to that stat */
	if ( _st )
	{
		t_.finish();
	}

	t_._st = nullptr;
	t_._pp = nullptr;
}

timer_to_exit &timer_to_exit::operator=(timer_to_exit &&t_)
{
	this->charge();
	_tm = t_._tm;
	_st = t_._st;
	_pp = t_._pp;
	_pa = t_._pa;

	/* If t_ has a durations_stat (and it should), give time until now to that stat */
	if ( _st )
	{
		t_.finish();
	}
	t_._st = nullptr;
	t_._pp = nullptr;
	return *this;
}

#if 0
/*
 * As above, but also attribute the timer split_time at construction to sp_
 */
timer_to_exit::timer_to_exit(const timer_to_exit *&pa_, timer_split &tm_, duration_stat &sp_, duration_stat &st_)
	: timer_to_exit(pa_, tm_, st_)
{
	this->chargs(sp_);
}
#endif

/*
 * iattribute the timer split_time up to now to sp_
 */
void timer_to_exit::finish() const
{
	if ( _st )
	{
		_st->record(_tm->split_duration());
	}
}

void timer_to_exit::charge() const
{
	if ( _st )
	{
		_st->charge(_tm->split_duration());
	}
}

timer_to_exit::~timer_to_exit()
{
	this->finish();
	/* If there is a current pointer to the "tte address" restore it to its previous value */
	if ( _pp )
	{
		*_pp = _pa;
	}
}
