/*
   Copyright [2017-2020] [IBM Corporation]
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

#ifndef _MCAS_PSTR_HASH_H_
#define _MCAS_PSTR_HASH_H_

#include <city.h>
#include <cstdint>
#include <string>

template <typename Key>
  struct pstr_hash
  {
    using argument_type = Key;
    using result_type = std::uint64_t;
    static result_type hf(const argument_type &s)
    {
      return CityHash64(s.data(), s.size());
    }
    /* Note: pstr_hash knows how to hash a std::string
     * even when it is not specialized for a string.
     * Why not require that the Key be std::string
     * if it is to compute a hash on a string?
     * Where do we require this special case?
     */
    static result_type hf(const std::string &s)
    {
      return CityHash64(s.data(), s.size());
    }
  };

#endif
