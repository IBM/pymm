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

#ifndef MCAS_HSTORE_HEAP_MC_SHIM_H
#define MCAS_HSTORE_HEAP_MC_SHIM_H

#include <mm_plugin_itf.h>

#include <ccpm/interfaces.h> /* ownership_callback, (IHeap_expandable, region_vector_t) */
#include <common/byte_span.h>
#include <common/string_view.h>
#include <gsl/span>

#include <cstddef> /* size_t */
#include <functional> /* function */

struct heap_mc_shim
	: public ccpm::IHeap_expandable
{
private:
	MM_plugin_wrapper _mm;
public:
	heap_mc_shim(common::string_view path, ccpm::persister *pe, gsl::span<common::byte_span> range, std::function<bool(const void *)> callee_owns);
	heap_mc_shim(common::string_view path, ccpm::persister *pe);
	heap_mc_shim(MM_plugin_wrapper &&pw);

	bool reconstitute(
		ccpm::region_span regions
		, ccpm::ownership_callback_t resolver
		, bool force_init
	) override;

	status_t allocate(void * & ptr
		, std::size_t bytes
		, std::size_t alignment
	) override;

	status_t free(void * & ptr
		, std::size_t bytes
	) override;

	status_t remaining(std::size_t& out_size) const override;

	ccpm::region_vector_t get_regions() const override;

	void add_regions(ccpm::region_span regions) override;

	bool includes(const void *ptr) const override;

	bool is_crash_consistent() const;
};

#endif
