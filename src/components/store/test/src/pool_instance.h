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

#ifndef MCAS_TEST_POOL_INSTANCE_H
#define MCAS_TEST_POOL_INSTANCE_H

#include "pool_opened.h"

#include <common/string_view.h>
#include <common/utils.h> /* MB */
#include <api/kvstore_itf.h>
#include <cstddef> /* size_t */
#include <memory> /* shared_ptr */
#include <string>

struct pool_named
{
	using IKVStore = component::IKVStore;
	std::shared_ptr<IKVStore> _kvstore;
	std::string _pool_name;
	bool _owned;
	pool_named(std::shared_ptr<IKVStore> kvstore_, common::string_view pool_name_)
		: _kvstore(kvstore_)
		, _pool_name(pool_name_)
		, _owned(false)
	{}
	~pool_named()
	{
		if ( _owned )
		{
			_kvstore->delete_pool(_pool_name);
		}
	}
};

struct pool_instance
	: pool_named
	, pool_opened
{
	using IKVStore = component::IKVStore;
private:
	template <typename ... Args>
		static IKVStore::pool_t create_pool(
			std::shared_ptr<IKVStore> kvstore_
			, common::string_view pool_name_
			, Args && ... args
		)
		{
			std::string n(pool_name_);
			kvstore_->delete_pool(n);
			return kvstore_->create_pool(n, std::forward<Args>(args) ...);
		}
public:
	pool_instance(std::shared_ptr<IKVStore> kvstore_, common::string_view pool_name_)
		: pool_named(kvstore_, pool_name_)
		, pool_opened(kvstore_, create_pool(kvstore_, _pool_name, MB(32)))
	{
		_owned = handle() != IKVStore::POOL_ERROR;
	}

	template <typename ... Args>
		pool_instance(std::shared_ptr<IKVStore> kvstore_, common::string_view pool_name_, Args && ... args)
			: pool_named(kvstore_, pool_name_)
			, pool_opened(kvstore_, create_pool(kvstore_, _pool_name, std::forward<Args>(args) ...))
		{}
};

#endif
