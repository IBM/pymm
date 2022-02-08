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

#ifndef _MCAS_COMMON_MOVEABLE_STRUCT_
#define _MCAS_COMMON_MOVEABLE_STRUCT_

#include <common/moveable_traits.h>
#include <algorithm> /* swap */

namespace common
{
  /* Like moveable_value, but T is a class or struct and
   * therefore can be a base class
   */
  template <typename T, typename Traits = moveable_traits<T>>
    struct moveable_struct
      : protected T
    {
      /* construct a moveable_struct from T constructor args */
      template <typename ... Args>
        explicit moveable_struct(Args&& ... args)
          : T(std::forward<Args>(args)...)
        {}

      moveable_struct(moveable_struct &&o_) noexcept
        : T(Traits::none)
      {
        using std::swap;
        swap(*static_cast<T *>(this), static_cast<T &>(o_));
      }

      moveable_struct &operator=(moveable_struct &&o_) noexcept
      {
        using std::swap;
        swap(*static_cast<T *>(this), static_cast<T &>(o_));
        return *this;
      }

      T release()
      {
        T t(Traits::none);
        using std::swap;
        swap(t, static_cast<T &>(*this));
        return t;
      }
    };
}

#endif
