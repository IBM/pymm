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

/*
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef _MCAS_MAP_STORE_FACTORY_H_
#define _MCAS_MAP_STORE_FACTORY_H_

#include <api/kvstore_itf.h>

class Map_store_factory : public component::IKVStore_factory {
public:
  /**
   * Component/interface management
   *
   */
  DECLARE_VERSION(1.0f);
  DECLARE_COMPONENT_UUID(0xfac20985, 0x1253, 0x404d, 0x94d7, 0x77, 0x92, 0x75, 0x21, 0xa1, 0x21);

  virtual ~Map_store_factory();

  void *query_interface(component::uuid_t &itf_uuid) override;

  void unload() override;

  component::IKVStore *create(unsigned debug_level,
                                      const IKVStore_factory::map_create &mc) override;
};

#endif
