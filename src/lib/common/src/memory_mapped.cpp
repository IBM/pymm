/*
   Copyright [2017-2019] [IBM Corporation]
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
 */

#include <common/memory_mapped.h>

#include <common/byte_span.h>
#include <common/logging.h>
#include <sys/mman.h>
#include <cerrno>
#include <cstring>

namespace
{
  /* not-present value must be compile-time evaluable.
  * MMAP_FAILED is not; use nullptr instead
  */
  static ::iovec translate_mmap_result(common::byte_span iov_)
  {
    return ::base(iov_) == MAP_FAILED ? common::make_iovec(nullptr, std::size_t(errno)) : common::make_iovec(::base(iov_), ::size(iov_));
  }
}

constexpr ::iovec common::iovec_moveable_traits::none;

#if 0
common::memory_mapped::memory_mapped(void *vaddr, std::size_t size, int prot, int flags, int fd) noexcept
  : memory_mapped(vaddr, size, prot, flags, fd, 0)
{
}

common::memory_mapped::memory_mapped(void *vaddr, std::size_t size, int prot, int flags, int fd, off_t offset) noexcept
  : memory_mapped(
      make_byte_span(
        ::mmap(vaddr, size, prot, flags, fd, offset)
        , size
      )
  )
{
}
#endif

common::memory_mapped::memory_mapped(const byte_span area, int prot, int flags, int fd) noexcept
  : memory_mapped(area, prot, flags, fd, 0)
{
}

common::memory_mapped::memory_mapped(const byte_span area, int prot, int flags, int fd, off_t offset) noexcept
  : memory_mapped(
      make_byte_span(
        ::mmap(::base(area), ::size(area), prot, flags, fd, offset)
        , ::size(area)
      )
  )
{
}

common::memory_mapped::memory_mapped(byte_span iov_) noexcept
  : moveable_struct<::iovec, iovec_moveable_traits>(translate_mmap_result(iov_))
{
}

int common::memory_mapped::shrink_by(std::size_t size_)
{
  void *discard_base = iov_end() - size_;
  auto rc = ::munmap(discard_base, size_);
  int e = errno;
  PLOG("%s: from %p,:0x%zx, %i = munmap(%p, %zu)", __func__, ::base(*this), ::size(*this), rc, discard_base, size_);
  if ( rc != 0 )
  {
    PLOG("%s: munmap(%p, %zu) failed: %s", __func__, discard_base, size_, ::strerror(e));
    errno = e;
  }
  return rc;
}

common::memory_mapped::~memory_mapped()
{
  if ( ::base(*this) != nullptr )
  {
    if ( ::munmap(::base(*this), ::size(*this)) != 0 )
    {
      auto e = errno;
      PLOG("%s: munmap(%p, %zu) failed: %s", __func__, ::base(*this), ::size(*this), ::strerror(e));
    }
  }
}
