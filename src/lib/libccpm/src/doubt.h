/*
   Copyright [2019] [IBM Corporation]
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

#ifndef CCPM_DOUBT_H
#define CCPM_DOUBT_H

#include <ccpm/interfaces.h>
#include <common/logging.h>
#include <gsl/pointers>
#include <cstddef>

namespace ccpm
{
	/*
	 * During transfer of ownership of a range of memory between the allocator
	 * and a client, the ownership of the memory is in doubt (as seen by the
	 * allocator).
	 * (On restore from a crash, the allocator will ask the client to resolve
	 * the doubt through ownership_callback_t.)
	 * Class doubt contains the element pointer when doubt exists.
	 */
	struct doubt
	{
		using persist_type = gsl::not_null<ccpm::persister *>;
	private:
		std::size_t _bytes;
		void *_in_doubt;
	public:
		doubt()
			: _bytes()
			, _in_doubt()
		{}

		void set(persist_type persist, const char *fn, void *p, std::size_t bytes);

		void clear(persist_type persist_, const char *fn)
		{
			set(persist_, fn, nullptr, 0);
		}

		void *get() const;

		std::size_t bytes() { return _bytes; }
	};
}

#endif
