/*
  Copyright [2017-2021] [IBM Corporation]
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

#ifndef _MCAS_REGION_MEMORY_H_
#define _MCAS_REGION_MEMORY_H_

#include <common/env.h>
#include <sys/uio.h> /* iovec */
#include <algorithm> /* find_if */
#include <cassert>
#include <cstddef> /* size_t */

struct region_memory
  : private ::iovec
{
private:
  int _debug_level;
public:
  region_memory(unsigned debug_level_, void *p_, std::size_t size_)
    : ::iovec{p_, size_}
    , _debug_level(debug_level_)
  {
    if ( p_ )
    {
      const auto s = static_cast<const char *>(p_);
      const auto e = s+size_;
      (void)e;
      if ( common::env_value("MCAS_CHECK_POOL_CLEAR", false) )
      {
        assert(std::find_if(s, e, [] ( const auto &c ) { return c != '\0'; }) == e);
      }
    }
  }
  unsigned debug_level() const { return _debug_level; }
  virtual ~region_memory() {}
  using ::iovec::iov_base;
  using ::iovec::iov_len;
};

#endif
