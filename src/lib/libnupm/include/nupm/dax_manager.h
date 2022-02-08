/*
   Copyright [2017-2020] [IBM Corporation]
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


/*
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef __NUPM_DAX_MANAGER_H__
#define __NUPM_DAX_MANAGER_H__

#include "nd_utils.h" /* ND_control */
#include "region_descriptor.h"
#include "space_opened.h"
#include "space_registered.h"
#include <common/byte_span.h>
#include <common/fd_open.h>
#include <common/logging.h>
#include <common/memory_mapped.h>
#include <common/moveable_ptr.h>
#include <common/string_view.h>
#include <boost/icl/interval_set.hpp>
#if ! defined _NUPM_FILESYSTEM_STD_ && defined __has_include
  #if __has_include (<filesystem>) && __cplusplus >= 201703L
    #include <filesystem>
    #define _NUPM_FILESYSTEM_STD_ 1
  #endif
#endif
#if ! defined _NUPM_FILESYSTEM_STD_
  #include <experimental/filesystem>
  #define _NUPM_FILESYSTEM_STD_ 0
#endif
#include <list>
#include <map>
#include <mutex>
#include <string>
#include <vector>

/* Control of the space behind a single config_t entry.
 * No longer called a region because
 * {create,open,erase}region functions use region to mean something else.
 */
struct arena;

namespace nupm
{
struct dax_manager;
class DM_region_header;

struct registry_memory_mapped
{
  using byte_span = common::byte_span;
  using string_view = common::string_view;
  virtual ~registry_memory_mapped() {}
  virtual bool enter(common::fd_locked &&fd, const string_view & id, const std::vector<byte_span> &m) = 0;
  virtual void remove(const string_view &id) = 0;
  virtual void * locate_free_address_range(std::size_t size) = 0;
};

/**
 * Lowest level persisent manager for devdax devices. See dax_map.cc for static
 * configuration.
 *
 */
struct dax_manager : protected common::log_source, private registry_memory_mapped {
 private:
  using byte = common::byte;

 public:
  using arena_id_t = unsigned;
  using string_view = common::string_view;
  static const bool have_odp;
  static const int effective_map_locked;

  struct config_t {
    std::string path;
    addr_t addr;
    /* Through no fault of its own, config_t may begin life with no proper values */
    config_t() : path(), addr(0) {}
  };

  struct config_mapped
  {
    std::string path;
    addr_t addr;
  };

  /**
   * Constructor e.g.
     nupm::dax_manager ddm({{"/dev/dax0.3", 0x9000000000, 0},
                               {"/dev/dax1.3", 0xa000000000, 1}},
                                true);
   *
   * @param dax_config Vector of dax-path, address, arena_id tuples.
   * @param force_reset
   */
  dax_manager(const common::log_source &ls,
              const std::vector<config_t>& dax_config,
              bool force_reset = false
		, common::byte_span address_span_ = common::make_byte_span(nullptr, 0)
	);

  /**
   * Destructor will not unmap memory/nor invalidate pointers?
   *
   */
  ~dax_manager();

  /**
   * Open a region of memory
   *
   * @param id Unique identifier
   * @param arena_id Arena identifier
   * @param out_length Out length of region in bytes
   *
   * @return backing file name (empty string if none);
   *   (pointer, length) pairs to the mapped memory, or empty vector
   *   if not found.
   *   Until fsdax supports extending a region, the vector will not be more
   *   than one element long.
   */
  region_descriptor open_region(const string_view & id, arena_id_t arena_id);

  /**
   * Create a new region of memory
   *
   * @param id Unique identifier
   * @param arena_id Arena identifier
   * @param size Size of the region requested in bytes
   *
   * @return backing file name (empty string if none);
   *   Pointer to and size of mapped memory
   */
  region_descriptor create_region(const string_view &id, arena_id_t arena_id, const size_t size);

  /**
   * Resize a region of memory
   *
   * @param id Unique identifier
   * @param arena_id Arena identifier
   * @param size Requested new size of the region requested in bytes
   *
   * The new size is just a request. The size may not change if, for example,
   * the underlying mechanism does not support resize. If the size does change,
   * it may change to something other than the requested size, due to rounding.
   * Returned values will not move existing mappings.
   *
   * Since open_region does not change state (perhaps it should be renamed
   * locate_region), it can be used to retrieve the new mapping of a resized region.
   *
   */
  region_descriptor resize_region(const string_view & id, arena_id_t arena_id, size_t size);

  /**
   * Erase a previously allocated region
   *
   * @param id Unique region identifier
   * @param arena_id Arena identifier
   */
  void erase_region(const string_view & id, arena_id_t arena_id);

  /**
   * Return a list of region names
   *
   */
  std::list<std::string> names_list(arena_id_t arena_id);

  /**
   * Get the maximum "hole" size.
   *
   *
   * @return Size in bytes of max hole
   */
  size_t get_max_available(arena_id_t arena_id);

  /**
   * Debugging information
   *
   * @param arena_id Arena identifier
   */
  void debug_dump(arena_id_t arena_id);

  void register_range(const void *begin, std::size_t size);
  void deregister_range(const void *begin, std::size_t size);
  void * locate_free_address_range(std::size_t size) override;
 private:
  using byte_span = common::byte_span;
  space_opened map_space(const std::string &path, addr_t base_addr);
  DM_region_header *recover_metadata(
                         byte_span iov,
                         bool    force_rebuild = false);
  arena *lookup_arena(arena_id_t arena_id);
  /* callback for arena_dax to register mapped memory */
  bool enter(common::fd_locked &&fd, const string_view & id, const std::vector<byte_span> &m) override;
  void remove(const string_view & id) override;
#if _NUPM_FILESYSTEM_STD_
  using path = std::filesystem::path;
  using directory_entry = std::filesystem::directory_entry;
#else
  using path = std::experimental::filesystem::path;
  using directory_entry = std::experimental::filesystem::directory_entry;
#endif
  void data_map_remove(const directory_entry &e, const std::string &origin);
  void map_register(const directory_entry &e, const std::string &origin);
  void files_scan(const path &p, const std::string &origin, void (dax_manager::*action)(const directory_entry &, const std::string &));
  std::unique_ptr<arena> make_arena_fs(const path &p, addr_t base, bool force_reset);
  std::unique_ptr<arena> make_arena_dev(const path &p, addr_t base, bool force_reset);
  std::unique_ptr<arena> make_arena_none(const path &p, addr_t base, bool force_reset);

 private:
  using guard_t = std::lock_guard<std::mutex>;
  using mapped_spaces = std::map<std::string, space_registered>;

  ND_control                                _nd;
  using AC = boost::icl::interval_set<byte *>;
  AC                                        _address_coverage;
  AC                                        _address_fs_available;
  mapped_spaces                             _mapped_spaces;
  std::vector<std::unique_ptr<arena>>       _arenas;
  std::mutex                                _reentrant_lock;
 public:
  friend struct nupm::range_use; /* access to _address_coverage */
};
}  // namespace nupm

#endif
