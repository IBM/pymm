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

#ifndef _MCAS_LESS_GETTER_H_
#define _MCAS_LESS_GETTER_H_

#include <memory>

/*
 * Compare for find/count/lower_bound/upper_bound of unique_ptr key in a set
 * or map
 */
namespace common
{
	template<typename T>
		struct less_getter
		{
			using is_transparent = void;

			bool operator()(const T &a, const T &b) const
			{
				return a.get() < b.get();
			}

			bool operator()(const typename T::pointer a, const T &b) const
			{
				return a < b.get();
			}

			bool operator()(const T &a, const typename T::pointer b) const
			{
				return a.get() < b;
			}
		};
}

#endif
