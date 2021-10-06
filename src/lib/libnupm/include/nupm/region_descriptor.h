/*
   Copyright [2020] [IBM Corporation]
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


/*
 * Authors:
 *
 */

#ifndef _NUPM_REGION_DESCRIPTOR_
#define _NUPM_REGION_DESCRIPTOR_

#include <common/byte_span.h>
#include <common/string_view.h>
#include <string>
#include <vector>

namespace nupm
{
	struct region_descriptor
	{
		using byte_span = common::byte_span;
		using address_map_t = std::vector<byte_span>;
		using string_view = common::string_view;
	private:
		std::string _id;
		std::string _data_file;
		address_map_t _address_map;
	public:
		/* fsdax data */
		region_descriptor(
			const string_view &id_
			, const string_view &data_file_
			, const address_map_t address_map_
		)
			: _id(id_)
			, _data_file(data_file_)
			, _address_map(address_map_)
		{}

		/* devdax data */
		region_descriptor(
			const address_map_t address_map_
		)
			: region_descriptor(string_view(), string_view(), address_map_)
		{}

		/* no data */
		region_descriptor()
			: region_descriptor(address_map_t())
		{}

		string_view id() const { return _id; }
		string_view data_file() const { return _data_file; }
		const address_map_t & address_map() const { return _address_map; }
		void address_map_push_back(address_map_t::value_type v) { _address_map.push_back(v); }
	};
}

#endif
