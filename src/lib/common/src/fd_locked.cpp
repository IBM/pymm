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
 *
 */

#include <common/fd_locked.h>

#include <unistd.h> /* lockf */

#include <cstring> /* strerror */
#include <sstream> /* ostringstream */
#include <stdexcept>
#include <thread> /* this_thread */

common::fd_locked::fd_locked()
  : Fd_open()
{}

common::fd_locked::fd_locked(int fd_)
  : Fd_open(fd_)
{
  /* Protection against using the same file in different processes */
  if ( ::lockf(fd_, F_TLOCK, 0) != 0 )
  {
    auto e = errno;
    throw std::runtime_error(std::string(__func__) + " exclusive lock failed: " + ::strerror(e));
  }
}
