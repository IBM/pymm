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


#ifndef _MCAS_HSTORE_TRACE_FLAGS_H
#define _MCAS_HSTORE_TRACE_FLAGS_H

#ifndef HSTORE_TRACE_ALL
#define HSTORE_TRACE_ALL 0
#endif

/* things to report */
#ifndef HSTORE_TRACE_MANY
#define HSTORE_TRACE_MANY HSTORE_TRACE_ALL
#endif

#ifndef HSTORE_TRACE_PALLOC
#define HSTORE_TRACE_PALLOC HSTORE_TRACE_ALL
#endif

#ifndef HSTORE_TRACE_PERSIST
#define HSTORE_TRACE_PERSIST HSTORE_TRACE_ALL
#endif

#ifndef HSTORE_TRACE_LOCK
#define HSTORE_TRACE_LOCK HSTORE_TRACE_ALL
#endif

#ifndef HSTORE_TRACE_RESIZE
#define HSTORE_TRACE_RESIZE HSTORE_TRACE_ALL
#endif

#ifndef HSTORE_TRACE_OWNER
#define HSTORE_TRACE_OWNER HSTORE_TRACE_ALL
#endif

#ifndef HSTORE_TRACE_BUCKET_IX
#define HSTORE_TRACE_BUCKET_IX HSTORE_TRACE_ALL
#endif

#ifndef HSTORE_TRACE_EXTEND
#define HSTORE_TRACE_EXTEND HSTORE_TRACE_ALL
#endif

#ifndef HSTORE_TRACE_PERISHABLE_EXPIRY
#define HSTORE_TRACE_PERISHABLE_EXPIRY HSTORE_TRACE_ALL
#endif

#ifndef HSTORE_TRACE_HEAP
#define HSTORE_TRACE_HEAP HSTORE_TRACE_ALL
#endif

static constexpr bool trace_heap = bool(HSTORE_TRACE_HEAP);
static constexpr bool trace_heap_summary = bool(HSTORE_TRACE_HEAP);
static constexpr bool trace_perishable_expiry = bool(HSTORE_TRACE_PERISHABLE_EXPIRY);

/* derived traces */
#define TRACED_TABLE (HSTORE_TRACE_MANY || HSTORE_TRACE_PERISHABLE_EXPIRY || HSTORE_TRACE_RESIZE)
#define TRACED_BUCKET TRACED_TABLE
#define TRACED_CONTENT TRACED_TABLE
#define TRACED_OWNER (HSTORE_TRACE_OWNER || TRACED_TABLE)

/* Data to track which is not normally needed but is required by some TRACE */
#define TRACK_OWNER (TRACED_OWNER || TRACED_CONTENT)
#define TRACK_POS TRACED_OWNER

#endif
