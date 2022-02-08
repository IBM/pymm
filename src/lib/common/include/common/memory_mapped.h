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

#ifndef __COMMON_MEMORY_MAPPED_H__
#define __COMMON_MEMORY_MAPPED_H__

#include <common/byte_span.h>
#include <common/moveable_struct.h>
#include <cstddef>

namespace common
{
  struct iovec_moveable_traits
  {
    static constexpr ::iovec none = common::make_iovec(nullptr, 0);
  };

  /*
   * Memory which is mapped, and which is to be unmapped
   */
  struct memory_mapped : public moveable_struct<::iovec, iovec_moveable_traits>
  {
    using byte_span = common::byte_span;
    using byte = common::byte;
  private:
    byte *iov_end() const { return ::data_end(*this); }
  public:
#if 0 // who uses? Can they use ::data(*this)?
    byte *iov_begin() const { return ::data(*this); }
#endif
    /* minimalist: argument is pointer and size */
    memory_mapped(byte_span iov) noexcept;
    /* non-minimalist: arguments are input to ::mmap */
    memory_mapped(byte_span area, int prot, int flags, int fd) noexcept;
    memory_mapped(byte_span area, int prot, int flags, int fd, off_t offset) noexcept;
    memory_mapped(memory_mapped &&) = default;
    memory_mapped &operator=(memory_mapped &&) = default;
    ~memory_mapped();
    using ::iovec::iov_base;
    using ::iovec::iov_len;
    operator bool() const { return ::base(*this) != nullptr; }
    const ::iovec &iov() const { return *this; }
    /* shrinks iov_len. iov_base remains constant */
    int shrink_by(std::size_t size);
  };
}

namespace
{
  inline auto base(const common::memory_mapped &m) { return m.iov_base; }
  inline auto data(const common::memory_mapped &m) { return static_cast<common::byte *>(::base(m)); }
  inline auto size(const common::memory_mapped &m) { return m.iov_len; }
  inline auto data_end(const common::memory_mapped &m) { return ::data(m) + ::size(m); }
  inline void *end(const common::memory_mapped &m) { return data_end(m); }
}
#endif
