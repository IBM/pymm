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


#ifndef _MCAS_HSTORE_HOP_HASH_EXCPETIONS_H
#define _MCAS_HSTORE_HOP_HASH_EXCPETIONS_H

#include <cstddef> /* size_t */
#include <stdexcept> /* range_error */
#include <string>

/*
 * Exceptions thrown by hop_hash
 */

namespace impl
{
	struct no_near_empty_bucket
		: public std::range_error
	{
		using bix_t = std::size_t;
	private:
		bix_t _bi;
	public:
		no_near_empty_bucket(bix_t bi, std::size_t size, const std::string &why);
		bix_t bi() const { return _bi; }
	};

	struct move_stuck
		: public no_near_empty_bucket
	{
		move_stuck(bix_t bi, std::size_t size);
	};

	struct hop_hash_full
		: public no_near_empty_bucket
	{
		hop_hash_full(bix_t bi, std::size_t size);
	};
}

#endif
