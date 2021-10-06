/*
   Copyright [2020] [IBM Corporation]
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

#ifndef _MCAS_COMMON_STRING_VIEW_
#define _MCAS_COMMON_STRING_VIEW_

#define MCAS_STRING_VIEW_USES_EXPERIMENTAL 1

#if MCAS_STRING_VIEW_USES_EXPERIMENTAL

#include <experimental/string_view>
namespace common
{
	template<typename T> using basic_string_view = std::experimental::basic_string_view<T>;
	using string_view = std::experimental::string_view;
}

#else

#include <string_view>
namespace common
{
	template<typename T> using basic_string_view = std::basic_string_view<T>;
	using string_view = std::string_view;
}

#endif

#endif
