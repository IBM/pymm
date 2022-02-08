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

#ifndef _MCAS_COMMON_MOVEABLE_VALUE_
#define _MCAS_COMMON_MOVEABLE_VALUE_

#include <common/moveable_traits.h>
#include <algorithm> /* swap */

namespace common
{
  /*
   * A value which zero-initializes the source when moved.
   * helpful for classes which use a special value, such as
   * nullptr or false, to denote "moved from" state.
   * moveable_traits<T>::none is that special value.
   */
  template <typename T, typename Traits = moveable_traits<T>>
    struct moveable_value
    {
      T v;
      moveable_value(const T &v_)
        : v(v_)
      {}
      moveable_value(moveable_value &&o_) noexcept
        : v(Traits::none)
      {
        using std::swap;
        swap(v, o_.v);
      }

      moveable_value &operator=(moveable_value &&o_) noexcept
      {
        using std::swap;
        swap(v, o_.v);
        return *this;
      }

      T release()
      {
        T t(Traits::none);
        using std::swap;
        swap(t, v);
        return t;
      }

      operator T() const { return v; }
      T get() const { return v; }
    };
}

#endif
