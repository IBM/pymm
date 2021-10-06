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


#ifndef _MCAS_HSTORE_LOGGING_H_
#define _MCAS_HSTORE_LOGGING_H_

#include <common/logging.h>

#define PREFIX_STATIC "HSTORE %s %s:%d "
#define LOCATION_STATIC __func__, __FILE__, __LINE__
#define PREFIX PREFIX_STATIC "%p "
#define LOCATION LOCATION_STATIC, common::p_fmt(this)

#endif
