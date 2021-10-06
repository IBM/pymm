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


#ifndef _MCAS_HSTORE_CONFIG_H_
#define _MCAS_HSTORE_CONFIG_H_

/*
 *   USE_CC_HEAP 2: simple allocation using offsets from a large region obtained from dax_map (NOT TESTED)
 *   USE_CC_HEAP 3: AVL-based allocation using actual addresses from a large region obtained from dax_map
 *   USE_CC_HEAP 4: bitmap-based allocation from a large region; crash-consistent
 *   USE_CC_HEAP 5: pluggable allocator (ccpm)
 *   USE_CC_HEAP 6: pluggable allocator (rcalb)
 *
 */

#if defined MCAS_HSTORE_USE_CC_HEAP
#define USE_CC_HEAP MCAS_HSTORE_USE_CC_HEAP
#else
#define USE_CC_HEAP 3
#endif

#if USE_CC_HEAP == 3
#define HEAP_MM 0
#define HEAP_RECONSTITUTE 1
#define HEAP_CONSISTENT 0
#elif USE_CC_HEAP == 4
#define HEAP_MM 0
#define HEAP_RECONSTITUTE 0
#define HEAP_CONSISTENT 1
#elif USE_CC_HEAP == 5
#error Obsolete
#define HEAP_MM 1
#define HEAP_RECONSTITUTE 0
#define HEAP_CONSISTENT 1
#elif USE_CC_HEAP == 6
#error Obsolete
#define HEAP_MM 1
#define HEAP_RECONSTITUTE 1
#define HEAP_CONSISTENT 0
#elif USE_CC_HEAP == 7
#define HEAP_MM 1
#undef HEAP_RECONSTITUTE
#undef HEAP_CONSISTENT
#else
#define HEAP_MM 0
#define HEAP_RECONSTITUTE 0
#define HEAP_CONSISTENT 0
#endif

#ifndef THREAD_SAFE_HASH
#define THREAD_SAFE_HASH 0
#endif
#define PREFIX_STATIC "HSTORE %s %s:%d "
#define LOCATION_STATIC __func__, __FILE__, __LINE__
#define PREFIX PREFIX_STATIC "%p "
#define LOCATION LOCATION_STATIC, common::p_fmt(this)

/* JIRA DAWN-292 requested a compile-configured grain size, default 32MiB 1<<25 */
#if ! defined HSTORE_LOG_GRAIN_SIZE
#define HSTORE_LOG_GRAIN_SIZE DM_REGION_LOG_GRAIN_SIZE
#endif

/* timestamps are enabled to match mapstore. To disable, compile with -DENABLE_TIMESTAMPS=0 */

#if ! defined ENABLE_TIMESTAMPS
#error Top level CMaleLists.txt should have defined ENABLE_TIMESTAMPS
#define ENABLE_TIMESTAMPS 1
#endif

#if THREAD_SAFE_HASH == 1
/* thread-safe hash */
#include <shared_mutex>
namespace hstore_impl
{
	using shared_mutex = std::shared_timed_mutex;
}
#else
/* not a thread-safe hash */
#include "dummy_shared_mutex.h"
namespace hstore_impl
{
	using shared_mutex = dummy::shared_mutex;
}
#endif

#endif
