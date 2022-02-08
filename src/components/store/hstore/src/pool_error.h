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


#ifndef _MCAS_HSTORE_POOL_ERROR_H
#define _MCAS_HSTORE_POOL_ERROR_H

#include <string>
#include <system_error>

enum class pool_ec
{
  pool_fail,
  pool_unsupported_mode,
  region_fail,
  region_fail_general_exception,
  region_fail_api_exception,
};

struct pool_category
  : public std::error_category
{
  const char* name() const noexcept override;
  std::string message( int condition ) const noexcept override;
};

namespace
{
	pool_category pool_error_category;
}

struct pool_error
  : public std::error_condition
{
private:
  std::string _msg;
public:
  pool_error(const std::string &msg, pool_ec val);

};

#endif
