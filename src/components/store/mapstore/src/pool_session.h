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

#ifndef _MCAS_POOL_SESSION_H_
#define _MCAS_POOL_SESSION_H_

#include <gsl/pointers>
#include <memory>

class Pool_instance;

struct pool_session {
  pool_session(gsl::not_null<std::shared_ptr<Pool_instance>> ph) : pool(ph), canary(0x45450101) {
  }

  ~pool_session() {
  }

  bool check() const { return canary == 0x45450101; }
  gsl::not_null<std::shared_ptr<Pool_instance>> pool;
  const unsigned canary;
};

#endif
