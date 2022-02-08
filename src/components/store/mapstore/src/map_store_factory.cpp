/*
  Copyright [2017-2021] [IBM Corporation]
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

#include "map_store_factory.h"

#include "map_store.h"

#include <unistd.h> /* access */
#include <cassert>
#include <string>

/**
 * Factory entry point
 *
 */
extern "C" void *factory_createInstance(component::uuid_t component_id) {
  if (component_id == Map_store_factory::component_id()) {
    return static_cast<void *>(new Map_store_factory());
  }
  else {
    return NULL;
  }
}

Map_store_factory::~Map_store_factory() {
}

void *Map_store_factory::query_interface(component::uuid_t &itf_uuid)
{
  if (itf_uuid == component::IKVStore_factory::iid()) {
    return static_cast<component::IKVStore_factory *>(this);
  }
  else return NULL;  // we don't support this interface
}

void Map_store_factory::unload() { delete this; }

component::IKVStore *Map_store_factory::create(unsigned debug_level,
                                    const IKVStore_factory::map_create &mc)
{
  auto owner_it = mc.find(+component::IKVStore_factory::k_owner);
  auto name_it = mc.find(+component::IKVStore_factory::k_name);
  auto numa_nodes_it = mc.find(+component::IKVStore_factory::k_numa_nodes);
  auto mm_plugin_path_it = mc.find(+component::IKVStore_factory::k_mm_plugin_path);

  std::string checked_mm_plugin_path;
  if(mm_plugin_path_it == mc.end()) {
    checked_mm_plugin_path = DEFAULT_MM_PLUGIN_PATH;
  }
  else {
    std::string path = mm_plugin_path_it->second;

    if(access(path.c_str(), F_OK) != 0) {
      path = DEFAULT_MM_PLUGIN_LOCATION + path;
      if(access(path.c_str(), F_OK) != 0) {
        PERR("inaccessible plugin path (%s) and (%s)", mm_plugin_path_it->second.c_str(), path.c_str());
        throw General_exception("unable to access mm_plugin");
      }
      checked_mm_plugin_path = path;
    }
    else {
      checked_mm_plugin_path = path;
    }
  }

  component::IKVStore *obj =
    static_cast<component::IKVStore *>
    (new Map_store(debug_level,
                   checked_mm_plugin_path,
                   owner_it == mc.end() ? "owner" : owner_it->second,
                   name_it == mc.end() ? "name" : name_it->second,
                   numa_nodes_it == mc.end() ? "" : numa_nodes_it->second
));
  assert(obj);
  obj->add_ref();
  return obj;
}
