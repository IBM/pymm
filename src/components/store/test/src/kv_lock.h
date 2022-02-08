/*
   Copyright [2017-2022] [IBM Corporation]
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

#ifndef MCAS_TEST_KV_LOCK_H
#define MCAS_TEST_KV_LOCK_H

#include "pool_opened.h"

#include <api/kvstore_itf.h>
#include <common/string_view.h>
#include <cassert>
#include <memory> /* shared_ptr */
#include <string>

struct pool_opened;

struct kv_lock
{
	using IKVStore = component::IKVStore;
protected:
	using handle_t = IKVStore::key_t;
private:
	std::shared_ptr<pool_opened> _pool;
	std::string _th;
	IKVStore::unlock_flags_t _unlock_flags;
	handle_t _handle;
	const char *_key;
	status_t _rc;
public:
	template <typename ... Args>
		kv_lock(std::shared_ptr<pool_opened> pool_, common::string_view th_, IKVStore::unlock_flags_t unlock_flags_, Args && ... args_);
	template <typename ... Args>
		kv_lock(std::shared_ptr<pool_opened> pool_, std::string_view th_, Args && ... args_)
			: kv_lock(
				pool_
				, th_
				, IKVStore::unlock_flags_t(component::IKVStore::UNLOCK_FLAGS_FLUSH)
				, std::forward<Args>(args_) ...
			)
		{}
	template <typename ... Args>
		kv_lock(std::shared_ptr<pool_opened> pool_, IKVStore::unlock_flags_t unlock_flags_, Args && ... args_)
			: kv_lock(
				pool_
				, common::string_view("unnamed")
				, IKVStore::unlock_flags_t(unlock_flags_)
				, std::forward<Args>(args_) ...
			)
		{}
	template <typename ... Args>
		kv_lock(std::shared_ptr<pool_opened> pool_, Args && ... args_)
			: kv_lock(
				pool_
				, IKVStore::unlock_flags_t(component::IKVStore::UNLOCK_FLAGS_FLUSH)
				, std::forward<Args>(args_) ...
			)
		{}
	kv_lock(const kv_lock &) = delete;
	kv_lock &operator=(const kv_lock &) = delete;

	auto rc() const { return _rc; }
	auto key() const { return _key; }

	~kv_lock()
	{
		if ( (rc() == S_OK || rc() == S_OK_CREATED) && _handle != handle_t() )
		{
			unlock(_unlock_flags);
		}
	}

	operator bool() const { return rc() == S_OK; }
	template <typename ... Args>
		auto unlock(Args && ... args) -> status_t;
};

struct shared_lock
	: kv_lock
{
	template <typename ... Args>
		shared_lock(Args && ... args_)
			: kv_lock(std::forward<Args>(args_) ...)
		{}
};

struct exclusive_lock
	: kv_lock
{
	template <typename ... Args>
		exclusive_lock(Args && ... args_)
			: kv_lock(std::forward<Args>(args_) ...)
		{}
};

template <typename ... Args>
	kv_lock::kv_lock(std::shared_ptr<pool_opened> pool_, common::string_view th_, IKVStore::unlock_flags_t unlock_flags_, Args && ... args_)
		: _pool(pool_)
		, _th(th_)
		, _unlock_flags(unlock_flags_)
		, _handle(IKVStore::KEY_NONE)
		, _key(nullptr)
		, _rc(_pool->lock(std::forward<Args>(args_) ..., _handle, &_key))
	{
		/* Unit test used to expect that handle is valid exactly when lock succeds.
		 * Not sure if that is in kvstore interface, but perhaps it should be.
		 */
		assert((_rc == S_OK || _rc == S_OK_CREATED) == (_handle != IKVStore::KEY_NONE));
	}
	

template <typename ... Args>
	auto kv_lock::unlock(Args && ... args) -> status_t
	{
		status_t r = _pool->unlock(_handle, std::forward<Args>(args) ...);
#if 0
		FLOG("{}", _th);
#endif
		_handle = IKVStore::key_t(IKVStore::KEY_NONE);
		return r;
	}

#endif
