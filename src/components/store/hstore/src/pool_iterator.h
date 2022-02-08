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

#ifndef MCAS_HSTORE_POOL_ITERATOR_H
#define MCAS_HSTORE_POOL_ITERATOR_H

#include <api/kvstore_itf.h> /* component::IKVStore::Opaque_pool_iterator */

#include <cstdint> /* uint64_t */

template <typename Iterator>
	struct pool_iterator
		: public component::IKVStore::Opaque_pool_iterator
	{
	private:
		using iterator_t = Iterator;
		std::uint64_t _mark;
	public:
		iterator_t _iter;
	private:
		iterator_t _end;
	public:
		explicit pool_iterator(
			std::uint64_t writes_
			, iterator_t first_
			, iterator_t last_
		);

		bool is_end() const;
		bool check_mark(std::uint64_t writes) const;
	};

#include "pool_iterator.tcc"

#endif
