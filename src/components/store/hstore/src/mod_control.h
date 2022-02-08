/*
   Copyright [2017-2019] [IBM Corporation]
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


#ifndef _MCAS_HSTORE_MOD_CONTROL_H
#define _MCAS_HSTORE_MOD_CONTROL_H

#include "persistent.h" /* persistent_t */
#include <api/kvstore_itf.h> /* Component::IKVStore::Op_type */
#include <cstddef> /* size_t */

namespace impl
{
	struct mod_control
	{
		persistent_t<component::IKVStore::Op_type> op;
		persistent_t<std::size_t> offset_src; /* For operations which have a source: WRITE, CAS */
		persistent_t<std::size_t> offset_dst;
		persistent_t<std::size_t> size; /* For operations which have a size: WRITE, ZERO */
		explicit mod_control(component::IKVStore::Op_type op_, std::size_t s_, std::size_t d_, std::size_t z_)
			: op(op_)
			, offset_src(s_)
			, offset_dst(d_)
			, size(z_)
		{}
		explicit mod_control() : mod_control(component::IKVStore::Op_type::ZERO, 0, 0, 0) {}
	};
}

#endif
