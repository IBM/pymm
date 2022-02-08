/*
   Copyright [2020] [IBM Corporation]
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

#ifndef _MCAS_NUPM_ARENA_NONE_
#define _MCAS_NUPM_ARENA_NONE_

#include "arena.h"

#include "filesystem.h"

/* An unsupported arena */
struct arena_none
	: arena
{
private:
#if _NUPM_FILESYSTEM_STD_
	using path = std::filesystem::path;
#else
	using path = std::experimental::filesystem::path;
#endif
	path _dir;
public:
	arena_none(const common::log_source &ls, path dir_) : arena(ls), _dir{dir_} {}
	region_descriptor region_get(
		const string_view // id
	) override
	{ return region_descriptor{}; }
	region_descriptor region_create(
		const string_view // id
		, gsl::not_null<registry_memory_mapped *> // mh
		, std::size_t // size
	) override
	{ return region_descriptor{}; }
	void region_resize(
      gsl::not_null<space_registered *> // sr
      , std::size_t // size
    ) override { }
	void region_erase(
		const string_view // id
		, gsl::not_null<registry_memory_mapped *> // mh
	) override {};
	std::size_t get_max_available() override { return 0; }
	bool is_file_backed() const override { return false; }
	void debug_dump() const override {}
	std::string describe() const override { return "<none>"; }
	std::list<std::string> names_list() const override { return {}; }
};

#endif
