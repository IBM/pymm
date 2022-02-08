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


#ifndef MCAS_HSTORE_PIN_CONTROL_H
#define MCAS_HSTORE_PIN_CONTROL_H

struct cptr;

template <typename Heap>
	struct pin_control
	{
		using arm_fn_type = void (Heap::*)(cptr &) const;
		using get_cptr_fn_type = char *(Heap::*)() const;
		using disarm_fn_type = void (Heap::*)() const;
		arm_fn_type _arm;
		disarm_fn_type _disarm;
		get_cptr_fn_type _get_cptr;

		pin_control(arm_fn_type a_, disarm_fn_type d_, get_cptr_fn_type c_)
			: _arm(a_)
			, _disarm(d_)
			, _get_cptr(c_)
		{}
};

#endif
