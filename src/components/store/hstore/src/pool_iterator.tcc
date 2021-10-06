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

#include "session.h"

template <typename Iterator>
	pool_iterator<Iterator>::pool_iterator(
		std::uint64_t writes_
		, iterator_t first_
		, iterator_t last_
	)
		: _mark(writes_)
		, _iter(first_)
		, _end(last_)
	{}

template <typename Iterator>
	bool pool_iterator<Iterator>::is_end() const { return _iter == _end; }

template <typename Iterator>
	bool pool_iterator<Iterator>::check_mark(std::uint64_t writes) const { return _mark == writes; }
