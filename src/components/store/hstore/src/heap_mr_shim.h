/*
   Copyright [2021] [IBM Corporation]
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

#ifndef MCAS_HSTORE_HEAP_MR_SHIM_H
#define MCAS_HSTORE_HEAP_MR_SHIM_H

#include <common/byte_span.h>
#include <common/types.h> /* status_t */
#include <common/string_view.h>
#include <mm_plugin_itf.h>

#include <cstddef> /* size_t */

struct heap_mr_shim
{
	using string_view = common::string_view;
	using byte_span = common::byte_span;
private:
	MM_plugin_wrapper _mm;
public:
	heap_mr_shim(string_view path);
	heap_mr_shim(MM_plugin_wrapper &&pw);

	status_t allocate(void * & ptr
		, std::size_t bytes
		, std::size_t alignment
	);

	status_t free(void * & ptr
		, std::size_t bytes
	);

	status_t inject_allocation(void *ptr, std::size_t size);

	void add_managed_region(byte_span region);
};

#endif
