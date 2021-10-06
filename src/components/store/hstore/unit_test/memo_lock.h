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

#ifndef MCAS_HSTORE_MEMO_LOCK_H
#define MCAS_HSTORE_MEMO_LOCK_H

#include "pool_open.h"
#include <api/kvstore_itf.h>
#include <common/string_view.h>

#include <string>

/*
 * For debug, XLOG should assemble its arguments, add a terminating \n,
 * and send them to cerr.
 */
#ifndef XLOG
#define XLOG(a, ...) do {} while(0)
#endif

struct memo_lock
{
	using IKVStore = component::IKVStore;
private:
	IKVStore *_kvstore;
	IKVStore::pool_t _pool;
	std::string _th;
public:
	IKVStore::key_t k;
	memo_lock(
		component::IKVStore *kvstore_
		, IKVStore::pool_t pool_id_
		, common::string_view th_
	)
		: _kvstore(kvstore_)
		, _pool(pool_id_)
		, _th(th_)
		, k(IKVStore::KEY_NONE)
	{
	}
#if 0
	memo_lock(
		component::IKVStore *kvstore_
		, pool_open & pool_
		, common::string_view th_
	)
		: memo_lock(kvstore_, pool_.id(), th_)
	{
	}
#endif
	memo_lock(const memo_lock &) = delete;
	memo_lock &operator=(const memo_lock &) = delete;
	~memo_lock()
	{
		if ( IKVStore::KEY_NONE != k )
		{
			XLOG(_th, " unlock");
			auto r = _kvstore->unlock(_pool, k, component::IKVStore::UNLOCK_FLAGS_FLUSH);
			EXPECT_EQ(S_OK, r);
		}
	}
};

struct shared_lock
	: private memo_lock
{
	using memo_lock::k;
	shared_lock(
		component::IKVStore *kvstore_
		, IKVStore::pool_t pool_id_
		, common::string_view th_ = common::string_view()
	)
		: memo_lock(kvstore_, pool_id_, th_)
	{}
	shared_lock(
		component::IKVStore *kvstore_
		, pool_open & pool_
		, common::string_view th_ = common::string_view()
	)
		: shared_lock(kvstore_, pool_.id(), th_)
	{}
};

struct exclusive_lock
	: private memo_lock
{
	using memo_lock::k;
	exclusive_lock(
		component::IKVStore *kvstore_
		, IKVStore::pool_t pool_id_
		, common::string_view th_ = common::string_view()
	)
		: memo_lock(kvstore_, pool_id_, th_)
	{}
	exclusive_lock(
		component::IKVStore *kvstore_
		, pool_open & pool_
		, common::string_view th_ = common::string_view()
	)
		: exclusive_lock(kvstore_, pool_.id(), th_)
	{}
};

#endif
