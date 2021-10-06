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


/*
 * Authors:
 *
 */

#ifndef _NUPM_OPENED_SPACE_H_
#define _NUPM_OPENED_SPACE_H_

#include <common/byte_span.h>
#include <common/fd_locked.h>
#include <common/logging.h> /* log_source */
#include <common/memory_mapped.h>
#include <common/moveable_ptr.h>
#include <common/string_view.h>
#include <common/types.h> /* addr_t */
#include <gsl/pointers>
#include <cassert>
#include <cstddef>
#include <string>
#include <vector>

namespace nupm
{
struct dax_manager;

struct range_use
{
private:
  common::moveable_ptr<dax_manager> _dm;
  /* Note: arena_fs used multiple ranges */
  std::vector<common::memory_mapped> _iovm;

  std::vector<common::memory_mapped> address_coverage_check(std::vector<common::memory_mapped> &&iovm);
  using byte = common::byte;
  using byte_span = common::byte_span;
public:
#if 0
  const auto & operator[](std::size_t i) const { return _iovm.at(i).iov(); }
#else
  byte_span operator[](std::size_t i) const { const auto &iov = _iovm.at(i).iov(); return common::make_byte_span(::base(iov), ::size(iov)); }
#endif
  range_use(dax_manager *dm_, std::vector<common::memory_mapped> &&);
  range_use(const range_use &) = delete;
  range_use &operator=(const range_use &) = delete;
  range_use(range_use &&) noexcept = default;
  void grow(std::vector<common::memory_mapped> &&);
  void shrink(std::size_t size);
  ~range_use();
  gsl::not_null<dax_manager *> dm() const { return _dm; }
  ::off_t size() const;
};

struct space_opened : private common::log_source
{
private:
  using byte_span = common::byte_span;
  common::fd_locked _fd_locked;
  /* Note: arena_fs may someday use multiple ranges */
  range_use _range;

  /* owns the file mapping */
  std::vector<common::memory_mapped> map_dev(int fd, const addr_t base_addr);
  std::vector<common::memory_mapped> map_fs(int fd, const std::vector<byte_span> &mapping, ::off_t offset);
public:
  space_opened(const common::log_source &, dax_manager * dm_, common::fd_locked && fd, const addr_t base_addr);
  space_opened(const common::log_source &, dax_manager * dm_, common::fd_locked && fd, const std::vector<byte_span> &mapping);
  space_opened(space_opened &&) noexcept = default;
  void shrink(std::size_t size);
  void grow(std::vector<byte_span> && iovv);
  int fd() const { return _fd_locked.fd(); }
  range_use &range() { return _range; }
};
}
#endif
