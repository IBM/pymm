/*
   Copyright [2019, 2021] [IBM Corporation]
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

#include <ccpm/cca.h>

#include "area_top.h"
#include "logging.h"
#include <common/errors.h> // S_OK, E_FAIL
#include <cassert>
#include <ostream>

#include <iostream>
namespace
{
	static unsigned accession = 0;
	static const char *c_trace_level = std::getenv("CCA_FINE_TRACE");
	unsigned trace_level = c_trace_level ? unsigned(std::stoull(c_trace_level)) : 0U;
	bool trace_coarse() { return 0 < trace_level; }
	bool trace_fine() { return 1 < trace_level; }
}

ccpm::cca::cca(persist_type persist_)
	: _top()
	, _last_top_allocate(0)
	, _last_top_free(0)
	, _persist(persist_)
{}

ccpm::cca::cca(persist_type persist_, const region_span regions_, ownership_callback_type resolver_)
	: cca(persist_)
{
	init(regions_, resolver_, false);
}

ccpm::cca::cca(persist_type persist_, const region_span regions_)
	: cca(persist_)
{
	init(regions_, [] (const void *) -> bool { return true; }, true);
}

ccpm::cca::~cca()
{
}

bool ccpm::cca::reconstitute(
	const region_span regions_
	, ownership_callback_type resolver_
	, const bool force_init_
)
{
	if ( ! _top.empty() ) { return false; }
	init(regions_, resolver_, force_init_);
	return ! _top.empty() && force_init_;
}

void ccpm::cca::init(
	const region_span regions_
	, ownership_callback_type resolver_
	, const bool force_init_
)
{
	PLOG(PREFIX "(%s)", LOCATION, force_init_ ? "clear" : "recover");
	for ( const auto & r : regions_ )
	{
		_top.push_back(
			force_init_
			? std::make_unique<area_top>(_persist, r, trace_level, std::cerr)
			: std::make_unique<area_top>(_persist, r, resolver_, trace_level, std::cerr)
		);
	}
	if ( trace_fine() )
	{
		this->print(std::cerr);
	}
}

void ccpm::cca::add_regions(const region_span regions_)
{
	for ( const auto & r : regions_ )
	{
		_top.push_back(std::make_unique<area_top>(_persist, r, trace_level, std::cerr));
	}
	if ( trace_fine() )
	{
		this->print(std::cerr);
	}
}

bool ccpm::cca::includes(const void *addr) const
{
	for ( const auto &it : _top )
	{
		if ( it->includes(addr) )
		{
			return true;
		}
	}
	return false;
}

auto ccpm::cca::allocate(
	void * & ptr_
	, std::size_t bytes_
	, std::size_t alignment_
) -> status_t
{
	assert(ptr_ == nullptr);
	/* Try all regions, round robin.
	 * In C++20 this can be done by a concatenation
	 * of the ranges [i .. end) and [begin .. i)
	 * Until then, use two loops.
	 */

	auto split = _top.begin() + _last_top_allocate;
	if (  trace_coarse() )
	{
		PLOG(PREFIX "AL %u %zx", LOCATION, accession++, bytes_);
	}
	if ( trace_fine() )
	{
		this->print(std::cerr);
	}
	for ( auto it = split; it != _top.end(); ++it )
	{
		(*it)->allocate(_persist, ptr_, bytes_, alignment_);
		if ( ptr_ != nullptr )
		{
			_last_top_allocate = it - _top.begin();
			if ( trace_fine() )
			{
				PLOG(PREFIX "allocate %p.%zx", LOCATION, ptr_, bytes_);
				this->print(std::cerr);
			}
			return S_OK;
		}
	}

	for ( auto it = _top.begin(); it != split; ++it )
	{
		(*it)->allocate(_persist, ptr_, bytes_, alignment_);
		if ( ptr_ != nullptr )
		{
			_last_top_allocate = it - _top.begin();
			if ( trace_fine() )
			{
				PLOG(PREFIX "allocate %p.%zx", LOCATION, ptr_, bytes_);
				this->print(std::cerr);
			}
			return S_OK;
		}
	}

	if ( trace_coarse() )
	{
		this->print(std::cerr, "Failed allocate " + std::to_string(bytes_) + " aligned " + std::to_string(alignment_));
	}

	return E_FAIL;
}

auto ccpm::cca::free(
	void * & ptr_
	, std::size_t bytes_
) -> status_t
{
	if ( trace_coarse() )
	{
		PLOG(PREFIX "cca DE %u %p", LOCATION, accession++, ptr_);
		this->print(std::cerr);
	}

	/* Change for C++20 ranges (see note for allocate) */
	auto split = _top.begin() + _last_top_free;

	for ( auto it = split; it != _top.end(); ++it )
	{
		if ( (*it)->contains(ptr_) )
		{
			(*it)->deallocate(_persist, ptr_, bytes_);
			_last_top_free = it - _top.begin();
			return ptr_ == nullptr ? S_OK : E_FAIL;
		}
	}

	for ( auto it = _top.begin(); it != split; ++it )
	{
		if ( (*it)->contains(ptr_) )
		{
			(*it)->deallocate(_persist, ptr_, bytes_);
			_last_top_free = it - _top.begin();
			return ptr_ == nullptr ? S_OK : E_FAIL;
		}
	}

	return E_FAIL;
}

auto ccpm::cca::remaining(
	std::size_t & out_size_
) const -> status_t
{
	std::size_t size = 0;
	for ( const auto & t : _top )
	{
		size += t->bytes_free();
	}
	out_size_ = size;
	return _top.empty() ? E_FAIL : S_OK;
}

auto ccpm::cca::get_regions() const -> region_vector_t
{
	region_vector_t rv;
	for ( const auto & t : _top )
	{
		rv.emplace_back(t->get_region());
	}
	return rv;
}

void ccpm::cca::set_root(
  byte_span root
)
{
  if(_top.size() == 0)
    throw std::runtime_error("unexpected empty top vector");
  auto& first_top = _top[0];
  first_top->set_root(root, _persist);
}

auto ccpm::cca::get_root() const -> byte_span
{
  if(_top.size() == 0)
    throw std::runtime_error("unexpected empty top vector");
  return _top[0]->get_root();
}


void ccpm::cca::print(std::ostream &o_, const std::string &title_) const
{
	o_ << title_ << ": " << "\n";
	for ( const auto & t : _top )
	{
		t->print(o_, std::ios_base::hex);
	}
	o_ << title_ << " end" << "\n";
}
