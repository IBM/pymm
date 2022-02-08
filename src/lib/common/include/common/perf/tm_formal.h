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

#ifndef MCAS_COMMON_PERF_TM_FORMAL_H
#define MCAS_COMMON_PERF_TM_FORMAL_H

#if ! defined MCAS_TM_ENABLED
#define MCAS_TM_ENABLED 0
#endif

/* Definitions for opionally passing a timer_split argument (header file use)
 */

#if MCAS_TM_ENABLED
namespace common
{
	namespace perf
	{
		struct timer_split;
	}
}

/* timer_split formal argument (without and with other args) */
#define TM_FORMAL0 common::perf::timer_split &
#define TM_FORMAL TM_FORMAL0,
#else
#define TM_FORMAL0
#define TM_FORMAL
#endif

#endif
