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

#ifndef MCAS_POOL_OPEN_H
#define MCAS_POOL_OPEN_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include <api/kvstore_itf.h>
#include <common/moveable_ptr.h>
#include <common/string_view.h>

#include <string>

/*
 * For debug, XLOG should assemble its arguments, add a terminating \n,
 * and send them to cerr.
 */
#define XLOG(a, ...) do {} while(0)

struct pool_open
{
private:
	common::moveable_ptr<component::IKVStore> _kvstore;
	component::IKVStore::pool_t _id;
public:
	pool_open(component::IKVStore *kvstore_, common::string_view name_)
		: _kvstore(kvstore_)
		, _id(_kvstore->open_pool(std::string(name_)))
	{}
	pool_open(component::IKVStore *kvstore_, component::IKVStore::pool_t id_)
		: _kvstore(kvstore_)
		, _id(id_)
	{}
	~pool_open()
	{
		if ( _kvstore )
		{
			auto rc = _kvstore->close_pool(_id);
			EXPECT_EQ(S_OK, rc);
		}
	}
	component::IKVStore::pool_t id() const { return _id; }
};

struct pool_create
{
private:
	common::moveable_ptr<component::IKVStore> _kvstore;
	std::string _name;
	component::IKVStore::pool_t _id;
public:
	template <typename ... Args>
		pool_create(component::IKVStore *kvstore_, common::string_view name_, Args ... args)
			: _kvstore(kvstore_)
			, _name(name_)
			, _id(_kvstore->create_pool(_name, args...))
	{
	}
	~pool_create()
	{
		if ( _kvstore && _id != +component::IKVStore::POOL_ERROR )
		{
			auto rc = _kvstore->delete_pool(_name);
			EXPECT_EQ(S_OK, rc);
		}
	}
	component::IKVStore::pool_t create_id() const { return _id; }
};

struct pool_temp
	: protected pool_create
	, public pool_open
{
	template <typename ... Args>
		pool_temp(component::IKVStore *kvstore_, common::string_view name_, Args ... args)
			: pool_create(kvstore_, name_, args...)
			, pool_open(kvstore_, this->create_id())
		{}
};

#endif
