/*
   Copyright [2022] [IBM Corporation]
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

#ifndef MCAS_TEST_POOL_OPENED_H
#define MCAS_TEST_POOL_OPENED_H

#include "pool_iterator.h"
#include <common/utils.h> /* tsc_time, epoch_time */
#include <api/kvstore_itf.h>
#include <cassert>
#include <memory> /* shared_ptr */
#include <string>

struct kv_lock;

struct pool_opened
{
	using IKVStore = component::IKVStore;
	std::shared_ptr<IKVStore> _kvstore;
	IKVStore::pool_t _pool;
public:
	friend struct kv_lock;
	pool_opened(std::shared_ptr<IKVStore> kvstore_, IKVStore::pool_t pool_)
		: _kvstore(kvstore_)
		, _pool(pool_)
	{}
	~pool_opened()
	{
		if ( _pool != IKVStore::POOL_ERROR )
		{
			_kvstore->close_pool(_pool);
		}
	}
	pool_opened(const pool_opened &) = delete;
	pool_opened(pool_opened &&po_)
		: _kvstore(po_._kvstore)
		, _pool(IKVStore::POOL_ERROR)
	{
		std::swap(_pool, po_._pool);
	}
	pool_opened &operator=(const pool_opened &) = delete;
	pool_opened &operator=(pool_opened && b)
	{
		if (  _pool != IKVStore::POOL_ERROR )
		{
			_kvstore->close_pool(_pool);
		}
		_kvstore = std::move(b._kvstore);
		_pool = b._pool;
		b._pool = IKVStore::POOL_ERROR;
		return *this;
	}
	IKVStore::pool_t handle() const { return _pool; }

	/**
	 * Close pool
	 *
	 * @return see IKVStore::close_pool
	 */
	auto close()
	{
		auto a = _kvstore->close_pool(_pool);
		_pool = IKVStore::POOL_ERROR;
		return a;
	}

	/**
	 * Get mapped memory regions for pool.
	 *
	 * @param args see IKVStore::get_pool_regions
	 *
	 * @return see IKVStore::get_pool_regions
	 */

	template <typename ... Args>
		auto get_regions(Args && ... args)
		{
			return _kvstore->get_pool_regions(_pool, std::forward<Args>(args) ...);
		}

	/**
	 * Dynamically expand a pool.
	 *
	 * @param args see IKVStore::grow_pool
	 *
	 * @return see IKVStore::grow_pool
	 */
	template <typename ... Args>
		auto grow(Args && ... args)
		{
			return _kvstore->grow_pool(_pool, std::forward<Args>(args) ...);
		}

	/**
	 * Write or overwrite an object value.
	 *
	 * @param args see IKVStore::put
	 *
	 * @return see IKVStore::put
	 */
	template <typename ... Args>
		auto put(Args && ... args)
		{
			return _kvstore->put(_pool, std::forward<Args>(args) ...);
		}

	/**
	 * Zero-copy put operation.
	 *
	 * @param args see IKVStore::put_direct
	 *
	 * @return see IKVStore::put_direct
	 */
	template <typename ... Args>
		auto put_direct(Args && ... args)
		{
			return _kvstore->put_direct(_pool, std::forward<Args>(args) ...);
		}

	/**
	 * Resize memory for a value
	 *
	 * @param args see IKVStore::resize_value
	 *
	 * @return see IKVStore::resize_value
	 */
	template <typename ... Args>
		auto resize_value(Args && ... args)
		{
			return _kvstore->resize_value(_pool, std::forward<Args>(args) ...);
		}

	/**
	 * Read an object value
	 *
	 * @param args see IKVStore::get
	 *
	 * @return see IKVStore::get
	 */
	template <typename ... Args>
		auto get(Args && ... args)
		{
			return _kvstore->get(_pool, std::forward<Args>(args) ...);
		}

	/**
	 * Read an object value directly into client-provided memory.
	 *
	 * @param args see IKVStore::get_direct
	 *
	 * @return see IKVStore::get_direct
	 */
	template <typename ... Args>
			auto get_direct(Args && ... args)
		{
			return _kvstore->get_direct(_pool, std::forward<Args>(args) ...);
		}

	/**
	 * Get attribute for key or pool
	 *
	 * @param args see IKVStore::get_attribute
	 *
	 * @return see IKVStore::get_attribute
	 */
	template <typename ... Args>
		auto get_attribute(Args && ... args)
		{
			return _kvstore->get_attribute(_pool, std::forward<Args>(args) ...);
		}

	/**
	 * Atomically (crash-consistent for pmem) swap keys
	 *
	 * @param args see IKVStore::swap_keys
	 *
	 * @return see IKVStore::swap_keys
	 */
	template <typename ... Args>
		auto swap_keys(Args && ... args)
	{
			return _kvstore->swap_keys(_pool, std::forward<Args>(args) ...);
	}

	/**
	 * Set attribute on a pool.
	 *
	 * @param args see IKVStore::set_attribute
	 *
	 * @return see IKVStore::set_attribute
	 */
	template <typename ... Args>
		auto set_attribute(Args && ... args)
		{
			return _kvstore->set_attribute(_pool, std::forward<Args>(args) ...);
	}

private:
/* use struct kv_lock to manage locks */
	/**
	 * Take a lock on a key-value pair.
	 *
	 * @param args see IKVStore::lock
	 *
	 * @return see IKVStore::lock
	 */
	template <typename ... Args>
		auto lock(Args && ... args)
		{
			return _kvstore->lock(_pool, std::forward<Args>(args) ...);
		}

	/**
	 * Unlock a key-value pair
	 *
	 * @param args see IKVStore::unlock
	 *
	 * @return see IKVStore::unlock
	 */
	template <typename ... Args>
		auto unlock(Args && ... args)
		{
			return _kvstore->unlock(_pool, std::forward<Args>(args) ...);
		}

public:
	/**
	 * Update an existing value by applying a series of operations.
	 *
	 * @param args see IKVStore::atomic_update
	 *
	 * @return see IKVStore::atomic_update
	 */
	template <typename ... Args>
		auto atomic_update(Args && ... args)
		{
			return _kvstore->atomic_update(_pool, std::forward<Args>(args) ...);
		}

	/**
	 * Erase an object
	 *
	 * @param args see IKVStore::erase
	 *
	 * @return see IKVStore::erase
	 */
	template <typename ... Args>
		auto erase(Args && ... args)
		{
			return _kvstore->erase(_pool, std::forward<Args>(args) ...);
		}

	/**
	 * Return number of objects in the pool
	 *
	 * @return see IKVStore::count
	 */
	std::size_t count()
	{
		return _kvstore->count(_pool);
	}

	/**
	 * Apply functor to objects in the pool, possibly according to time constraints
	 *
	 * @param args see IKVStore::map
	 *
	 * @return see IKVStore::map
	 */
	template <typename ... Args>
		auto map(Args && ... args)
		{
			return _kvstore->map(_pool, std::forward<Args>(args) ...);
		}

	/**
	 * Apply functor to all keys only.
	 *
	 * @param args see IKVStore::map_keys
	 *
	 * @return see IKVStore::map_keys
	 */
	template <typename ... Args>
		auto map_keys(Args && ... args)
		{
			return _kvstore->map_keys(_pool, std::forward<Args>(args) ...);
		}

	/**
	 * Open pool iterator to iterate over objects in pool.
	 *
	 * @return see IKVStore::open_pool_iterator
	 */
	pool_iterator open_iterator()
	{
		return pool_iterator(*this, _kvstore->open_pool_iterator(_pool));
	}

	/**
	 * Allocate memory from pool
	 *
	 * @param args see IKVStore::allocate_memory
	 *
	 * @return see IKVStore::allocate_memory
	 */
	template <typename ... Args>
		auto allocate_memory(Args && ... args)
		{
			return _kvstore->allocate_pool_memory(_pool, std::forward<Args>(args) ...);
		}

	/**
	 * Free memory from pool
	 *
	 * @param args see IKVStore::free_memory
	 *
	 * @return see IKVStore::free_memory
	 */
	template <typename ... Args>
		auto free_memory(Args && ... args)
		{
			return _kvstore->free_pool_memory(_pool, std::forward<Args>(args) ...);
	}

	/**
	 * Flush memory from pool
	 *
	 * @param args see IKVStore::flush_memory
	 *
	 * @return see IKVStore::flush_memory
	 */
	template <typename ... Args>
		auto flush_memory(Args && ... args)
		{
			return _kvstore->flush_pool_memory(_pool, std::forward<Args>(args) ...);
		}

	auto free_memory(void* p)
	{
		return _kvstore->free_memory(p);
	}

	/**
	 * Debug routine
	 *
	 * @param args see IKVStore::debug
	 */
	template <typename ... Args>
		void debug(Args && ... args)
		{
			return _kvstore->debug(_pool, std::forward<Args>(args) ...);
		}

};

#endif
