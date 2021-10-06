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

#ifndef _MCAS_FD_LOCKED_H_
#define _MCAS_FD_LOCKED_H_

#include <common/fd_open.h>

namespace common
{
  struct fd_locked
    : public Fd_open
  {
    fd_locked();
    /*
     * @throw std::logic_error : initialized with a negative value
     */
    explicit fd_locked(int fd);

    fd_locked(fd_locked &&) noexcept = default;
    fd_locked &operator=(fd_locked &&) noexcept = default;
  };
}

#endif
