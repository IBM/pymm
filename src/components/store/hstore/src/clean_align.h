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

#ifndef _MCAS_HSTORE_CLEAN_ALIGN_H_
#define _MCAS_HSTORE_CLEAN_ALIGN_H_

#include <algorithm> /* max */
#include <cstddef> /* size_t */
#include <stdexcept> /* invalid_argument */

namespace
{
	inline std::size_t clean_align(std::size_t align_, std::size_t min_align_)
	{
		auto align = std::max(align_, min_align_);
		if ( (align & (align - 1U)) != 0 )
		{
			throw std::invalid_argument("alignment is not a power of 2");
		}
		return align;
	}

	inline std::size_t clean_align(std::size_t align_)
	{
		return clean_align(align_, 1U);
	}
}

#endif
