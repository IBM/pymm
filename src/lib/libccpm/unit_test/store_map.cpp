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
#include "store_map.h"

#include <common/json.h>
#include <string>

const std::string store_map::impl_default = "hstore-cc";
const store_map::impl_map_t store_map::impl_map = {
  { "hstore-cc", { "hstore-cc", component::hstore_factory } }
  , { "hstore", { "hstore", component::hstore_factory } }
};

namespace
{
  static auto store_env = ::getenv("STORE");
  static auto store_loc = ::getenv("STORE_LOCATION");
}

const store_map::impl_map_t::const_iterator store_map::impl_env_it =
  store_env
  ? impl_map.find(store_env)
  : impl_map.end()
  ;
const store_map::impl_spec *const store_map::impl =
  &( impl_env_it == impl_map.end()
  ? impl_map.find(impl_default)
  : impl_env_it)->second
  ;

namespace c_json = common::json;
using json = c_json::serializer<c_json::dummy_writer>;
const std::string store_map::location =
  store_loc
  ? store_loc
  : json::array(
      json::object(
        json::member("path", "/dev/dax1.0")
        , json::member("addr", 0x9000000000)
      )
    ).str()
  ;
