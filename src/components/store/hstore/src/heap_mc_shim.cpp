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

#include "heap_mc_shim.h"

#include "mm_plugin_itf.h"
#include <ccpm/cca.h> /* ctor_args */
#include <stdexcept>

namespace
{
	MM_plugin_wrapper temp_make(
		common::string_view path_
		, ccpm::persister *pe_
		, gsl::span<common::byte_span> range_
		, bool force_init_
		, std::function<bool(const void *)> callee_owns_
	)
	{
		ccpm::cca::ctor_args args{pe_, range_, force_init_, callee_owns_};
		MM_plugin_wrapper pw(std::string(path_), "", &args);
		pw.init();
		return pw;
	}
}

heap_mc_shim::heap_mc_shim(
	common::string_view path_
	, ccpm::persister *pe_
	, gsl::span<common::byte_span> range_
	, std::function<bool(const void *)> callee_owns_
)
	: _mm(temp_make(path_, pe_, range_, false /* force_init */, callee_owns_))
{
}

heap_mc_shim::heap_mc_shim(
	common::string_view path_
	, ccpm::persister *pe_
)
	: _mm(temp_make(path_, pe_, gsl::span<common::byte_span>{}, false /* force_init */, [] (const void *) -> bool { return true; }))
{
}

heap_mc_shim::heap_mc_shim(
	MM_plugin_wrapper &&pw_
)
	: _mm(std::move(pw_))
{
}

bool heap_mc_shim::reconstitute(
	ccpm::region_span // regions
	, ccpm::ownership_callback_t // resolver
	, bool // force_init
)
{
	return true; /* Reconstitute support is in the constructor, not here. */
}

status_t heap_mc_shim::allocate(
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

status_t heap_mc_shim::free(
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

status_t heap_mc_shim::remaining(std::size_t &out_size) const
{
	return _mm.bytes_remaining(&out_size);
}

ccpm::region_vector_t heap_mc_shim::get_regions() const
{
	ccpm::region_vector_t r{};
	status_t rc;
	unsigned region_id = 0;
	do
	{
		void *region_base;
		std::size_t region_size;
		rc = const_cast<MM_plugin_wrapper &>(_mm).query_managed_region(region_id, &region_base, &region_size);
		if ( rc != E_INVAL )
		{
			r.push_back(common::make_byte_span(static_cast<common::byte *>(region_base), region_size));
		}
		++region_id;
	} while ( rc == S_MORE );
	return r;
}

void heap_mc_shim::add_regions(ccpm::region_span regions)
{
	for ( auto r : regions )
	{
		_mm.add_managed_region(::base(r), ::size(r));
	}
	return;
}

bool heap_mc_shim::includes(
	const void * // ptr
) const
{
	throw std::runtime_error("includes not supported");
}

bool heap_mc_shim::is_crash_consistent() const
{
	return true;
}
