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

#ifndef _MCAS__HSTORE_NON_OWNER_H
#define _MCAS__HSTORE_NON_OWNER_H

/* A struct which resembles unique_ptr or shared_ptr, but for non-owing uses */
template <typename T>
	struct non_owner
	{
		using element_type = T;
	private:
		T *_p;
	public:
		explicit non_owner() : non_owner(nullptr) {}
		explicit non_owner(element_type *p_) : _p(p_) {}
		non_owner(const non_owner &other) = default;
		non_owner &operator=(const non_owner &other) = default;
		element_type *get() const { return _p; }
		operator bool() const { return bool(_p); }
		element_type * operator->() const { return _p; }
		element_type &operator*() const { return *_p; }
	};

#endif
