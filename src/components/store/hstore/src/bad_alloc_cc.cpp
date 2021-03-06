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

#include "bad_alloc_cc.h"

bad_alloc_cc::bad_alloc_cc(AK_ACTUAL std::size_t pad, std::size_t count, std::size_t size)
	: _what(std::string(__func__) + ": " + std::to_string(pad) + "+" + std::to_string(count) + "*" + std::to_string(size))
{
	AK_REF_VOID;
}

const char *bad_alloc_cc::what() const noexcept
{
	return _what.c_str();
}
