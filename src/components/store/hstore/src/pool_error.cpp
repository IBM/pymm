/*
   Copyright [2021] [IBM Corporation]
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

#include "pool_error.h"

const char* pool_category::name() const noexcept { return "pool_category"; }

std::string pool_category::message( int condition ) const noexcept
{
	switch ( condition )
	{
    case int(pool_ec::pool_fail):
		return "default pool failure";
	case int(pool_ec::pool_unsupported_mode):
		return "pool unsupported flags";
	case int(pool_ec::region_fail):
		return "region-backed pool failure";
	case int(pool_ec::region_fail_general_exception):
		return "region-backed pool failure (General_exception)";
	case int(pool_ec::region_fail_api_exception):
		return "region-backed pool failure (API_exception)";
	default:
		return "unknown pool failure";
	}
}

pool_error::pool_error(const std::string &msg_, pool_ec val_)
	: std::error_condition(int(val_), pool_error_category)
	, _msg(msg_)
{}
