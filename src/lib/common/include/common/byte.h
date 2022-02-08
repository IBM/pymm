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

#ifndef _MCAS_COMMON_BYTE_
#define _MCAS_COMMON_BYTE_

#include <cstddef>
#include <gsl/gsl_byte>

#ifndef MCAS_BYTE_USES_STD
/* For compilation with C++14 use gsl::byte, not C++17 std::byte */
#define MCAS_BYTE_USES_STD 0
#endif

namespace common
{
#if MCAS_BYTE_USES_STD
	using byte = std::byte;
#else
	using byte = gsl::byte; /* can be std::byte in C++17 */
#endif
}

#endif
