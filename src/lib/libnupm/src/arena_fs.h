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

#ifndef _MCAS_NUPM_ARENA_FS_
#define _MCAS_NUPM_ARENA_FS_

#include "arena.h"

#include "filesystem.h"

#include <common/byte_span.h>
#include <common/fd_locked.h>
#include <common/logging.h>

#include <string>

namespace common
{
	struct memory_mapped;
}

/* An arena implemented by an fsdax directory */
struct arena_fs
	: arena
{
private:
	using byte_span = common::byte_span;
#if _NUPM_FILESYSTEM_STD_
	using path = std::filesystem::path;
#else
	using path = std::experimental::filesystem::path;
#endif
	path _dir;

	void *region_create_inner(
		common::fd_locked &&fd
		, string_view id_
		, gsl::not_null<registry_memory_mapped *> mh
		, const std::vector<byte_span> &mapping
	);
	path path_data(string_view id) const;
	path path_map(string_view id) const;
	static std::vector<byte_span> get_mapping(const path &path_map, const std::size_t expected_size);
public:
	arena_fs(const common::log_source &ls, path dir);
	region_descriptor region_get(const string_view id) override;
	region_descriptor region_create(const string_view id, gsl::not_null<registry_memory_mapped *> mh, std::size_t size) override;
	void region_resize(gsl::not_null<space_registered *> mh, std::size_t size) override;
	void region_erase(const string_view id, gsl::not_null<registry_memory_mapped *> mh) override;
	std::size_t get_max_available() override;
    bool is_file_backed() const override { return true; }
	void debug_dump() const override;
	static std::pair<std::vector<byte_span>, std::size_t> get_mapping(const path &path_map);
	static std::vector<common::memory_mapped> fd_mmap(int fd, const std::vector<byte_span> &map, int flags, ::off_t size);
	std::string describe() const override { return _dir.string(); }
	std::list<std::string> names_list() const override;
};

#endif
