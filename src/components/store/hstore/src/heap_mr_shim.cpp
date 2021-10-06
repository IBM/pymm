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

#include "heap_mr_shim.h"

#include "hstore_config.h"
#include "mm_plugin_itf.h"
#include <common/errors.h> /* S_OK */
#include <stdexcept> /* runtime_error */


heap_mr_shim::heap_mr_shim(string_view path)
	: _mm(std::string(path), "", nullptr)
{
}

heap_mr_shim::heap_mr_shim(MM_plugin_wrapper &&pw_)
	: _mm(std::move(pw_))
{
}

status_t heap_mr_shim::allocate(
	void * & ptr
	, std::size_t bytes
	, std::size_t alignment
)
{
	return
		alignment
		? _mm.aligned_allocate( bytes, alignment, &ptr)
		: _mm.allocate(bytes, &ptr)
		;
}

status_t heap_mr_shim::free(
	void * & ptr
	, std::size_t bytes
)
{
	return
		bytes
		? _mm.deallocate(&ptr, bytes)
		: _mm.deallocate_without_size(&ptr)
		;
}

status_t heap_mr_shim::inject_allocation(void *ptr, std::size_t size)
{
	return _mm.inject_allocation(ptr, size);
}

void heap_mr_shim::add_managed_region(byte_span region_)
{
	_mm.add_managed_region(::base(region_), ::size(region_));
}
