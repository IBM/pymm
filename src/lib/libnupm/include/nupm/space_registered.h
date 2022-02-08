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
 */

#ifndef _NUPM_SPACE_REGISTERED_H_
#define _NUPM_SPACE_REGISTERED_H_

#include "space_opened.h"
#include "path_use.h"
#include <common/byte_span.h>
#include <common/logging.h>
#include <string>
#include <vector>

namespace nupm
{
struct dax_manager;

struct space_registered
{
  using byte_span = common::byte_span;
  using string_view = common::string_view;
private:
  path_use _pu;
public:
  space_opened _or;
public:
  space_registered(
    const common::log_source &ls
    , dax_manager * dm
    , common::fd_locked &&fd
    , const string_view &path
    , addr_t base_addr
  );
  space_registered(
    const common::log_source &ls
    , dax_manager * dm
    , common::fd_locked &&fd
    , const string_view &name
    , const std::vector<byte_span> &mapping
  );

  space_registered(const space_registered &) = delete;
  space_registered &operator=(const space_registered &) = delete;
  space_registered(space_registered &&) noexcept = default;
  const std::string & path_name() const { return _pu.path_name(); }
};
}  // namespace nupm

#endif
