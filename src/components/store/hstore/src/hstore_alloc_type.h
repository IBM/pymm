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


#ifndef MCAS_HSTORE_ALLOC_TYPE_H_
#define MCAS_HSTORE_ALLOC_TYPE_H_

#include "hstore_config.h"

struct heap_mr;
struct heap_rc;
struct heap_mc;
struct heap_cc;
struct heap_mm;

template <typename Data, typename Persister>
	struct allocator_co;
template <typename Data, typename AllocType, typename Persister>
	struct allocator_cc;
template <typename AllocType>
    struct heap_access;

template <typename Persister>
	struct hstore_alloc_type
	{
#if HEAP_OID
		using alloc_type = allocator_co<char, Persister>;
		using heap_alloc_type = heap_co;
#elif HEAP_MM  && ! defined HEAP_RECONSTITUTE && ! defined HEAP_CONSISTENT
		using heap_alloc_shared_type = heap_mm;
		using alloc_type = allocator_cc<char, heap_alloc_shared_type, Persister>;
#elif HEAP_RECONSTITUTE
#if HEAP_MM
		using heap_alloc_shared_type = heap_mr;
#else
		using heap_alloc_shared_type = heap_rc;
#endif
		using alloc_type = allocator_cc<char, heap_alloc_shared_type, Persister>;
#elif HEAP_CONSISTENT
#if HEAP_MM
		using heap_alloc_shared_type = heap_mc;
#else
		using heap_alloc_shared_type = heap_cc;
#endif
		using alloc_type = allocator_cc<char, heap_alloc_shared_type, Persister>;
#endif
		using heap_alloc_access_type = heap_access<heap_alloc_shared_type>;
	};

#endif
