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

#ifndef MCAS_HSTORE_ALLOC_KEY_H
#define MCAS_HSTORE_ALLOC_KEY_H

#include "test_flags.h" /* AK_USED */

/* Attempt to ensure that all throws of bad_alloc_cc are caught.
 * Construction of bad_alloc_cc requires an alloc_key instance,
 * which (by rule, not enforced) may only be created by functions
 * which catch bad_alloc_cc.
 */

#if AK_USED
struct alloc_key;
#define AK_INSTANCE0 alloc_key{}
#define AK_INSTANCE AK_INSTANCE0,
#define AK_FORMAL0 const alloc_key &
#define AK_FORMAL AK_FORMAL0,
#define AK_ACTUAL0 const alloc_key &ak_
#define AK_ACTUAL AK_ACTUAL0,
#define AK_REF0 ak_
#define AK_REF AK_REF0,
#define AK_REF_VOID ((void) AK_REF0)
#else
#define AK_INSTANCE0
#define AK_INSTANCE
#define AK_FORMAL0
#define AK_FORMAL
#define AK_ACTUAL0
#define AK_ACTUAL
#define AK_REF0
#define AK_REF
#define AK_REF_VOID ((void) 0)
#endif

#endif
