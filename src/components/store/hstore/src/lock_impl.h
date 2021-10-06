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

#ifndef MCAS_HSTORE_LOCK_IMPL_H
#define MCAS_HSTORE_LOCK_IMPL_H

#include <api/kvstore_itf.h>

#include <common/string_view.h>
#include <string>

struct lock_impl
	: public component::KVStore::Opaque_key
{
	using string_view = common::string_view;
private:
	std::string _s;
public:
	lock_impl(const string_view s_)
		: component::IKVStore::Opaque_key{}
		, _s(s_.begin(), s_.end())
	{
	}
	const std::string &key() const { return _s; }
	~lock_impl()
	{
	}
};

#endif
