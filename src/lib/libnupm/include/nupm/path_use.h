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

#ifndef _NUPM_PATH_USE_H_
#define _NUPM_PATH_USE_H_

#include <common/logging.h>
#include <common/string_view.h>
#include <string>

namespace nupm
{
struct path_use
  : public common::log_source
{
  using string_view = common::string_view;
private:
  /* Would include a common::moveable_ptr<dax_manager>, except that the
   * registry is static, potentially covering multiple dax_manager instances.
   */
  std::string _name;
public:
  path_use(const common::log_source &ls, const string_view &name);
  path_use(const path_use &) = delete;
  path_use &operator=(const path_use &) = delete;
  path_use(path_use &&) noexcept;
  ~path_use();
  const std::string & path_name() const { return _name; }
};
}  // namespace nupm

#endif
