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

#ifndef MCAS_HSTORE_VALGRIND_MEMCHECK_H
#define MCAS_HSTORE_VALGRIND_MEMCHECK_H

#if 0
#include <valgrind/memcheck.h>
#else
#define VALGRIND_CREATE_MEMPOOL(pool, x, y) do { (void) (pool); (void) (x); (void) (y); } while(0)
#define VALGRIND_DESTROY_MEMPOOL(pool) do { (void) (pool); } while(0)
#define VALGRIND_MAKE_MEM_DEFINED(pool, size) do { (void) (pool); (void) (size); } while(0)
#define VALGRIND_MAKE_MEM_UNDEFINED(pool, size) do { (void) (pool); (void) (size); } while(0)
#define VALGRIND_MEMPOOL_ALLOC(pool, addr, size) do { (void) (pool); (void) (addr); (void) (size); } while(0)
#define VALGRIND_MEMPOOL_FREE(pool, size) do { (void) (pool); (void) (size); } while(0)
#endif

#endif
