/*
   Copyright [2018-2019] [IBM Corporation]
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

#ifndef MCAS_HSTORE_BITS_TO_INTS_H
#define MCAS_HSTORE_BITS_TO_INTS_H

#include <cstdint>
#include <string>
#include <vector>

inline std::vector<std::size_t> bits_to_ints(std::uint64_t b, std::size_t offset)
{
	std::vector<std::size_t> v;
	for ( ; b != 0; b>>=1, ++offset)
	{
		if ( b & 1 )
		{
			v.push_back(offset);
		}
	}
	return v;
}

inline std::string ints_to_string(std::vector<std::size_t> v)
{
	std::string s = "(";
	for ( auto i : v )
	{
		s += std::to_string(i) + " ";
	}
	return s + ")";
}

#endif
