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

#ifndef MCAS_HSTORE_SESSION_BASE_H
#define MCAS_HSTORE_SESSION_BASE_H

#include "hstore_config.h"
#include <common/logging.h> /* log_source */

#include <common/string_view.h>
#include <array>
#include <cstddef> /* size_t */
#include <cstdint> /* uint64_t */
#include <string>

template <typename Handle>
	struct session_base
		: public Handle
		, protected common::log_source
	{
	protected:
		using handle_type = Handle;
		using pool_type = typename handle_type::pool_type;
		using string_view = common::string_view;
		std::uint64_t _writes;
		static const std::string ac_prefix;

		session_base(
			Handle &&pop
			, unsigned debug_level
		);
	};

#endif
