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

#ifndef MCAS_TEST_POOL_ITERATOR_H
#define MCAS_TEST_POOL_ITERATOR_H

#include <api/kvstore_itf.h>
#include <memory> /* shared_ptr */

struct pool_opened;

struct pool_iterator
{
	using IKVStore = component::IKVStore;
	std::shared_ptr<IKVStore> _kvstore;
	IKVStore::pool_t _pool;
	IKVStore::pool_iterator_t _it;
	pool_iterator(const pool_opened & p_, IKVStore::pool_iterator_t _it);
	pool_iterator(const pool_iterator &) = delete;
	pool_iterator &operator=(const pool_iterator &) = delete;

	~pool_iterator();
	/**
 *
	 * Deference pool iterator position and optionally increment
	 *
	 * @param args see IKVStore::deref_pool_iterator
	 *
	 * @return see IKVStore::deref_pool_iterator
	 */
	template <typename ... Args>
		auto deref(Args && ... args)
		{
			return _kvstore->deref_pool_iterator(_pool, _it, std::forward<Args>(args) ...);
		}
};

#endif
