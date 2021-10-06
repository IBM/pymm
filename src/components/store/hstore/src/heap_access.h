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


#ifndef MCAS_HSTORE_HEAP_ACCESS_H
#define MCAS_HSTORE_HEAP_ACCESS_H

template <typename T>
	struct heap_access
	{
		using shared_type = T;
	private:
		using shared_t = shared_type;
		shared_type *_heap;

	public:
		explicit heap_access(shared_type *area)
			: _heap(area)
		{
		}

		~heap_access()
		{
		}

		heap_access(const heap_access &) noexcept = default;

		heap_access & operator=(const heap_access &) = default;

		shared_type *operator->() const
		{
			return _heap;
		}

		shared_type &operator*() const
		{
			return *_heap;
		}
	};

#endif
