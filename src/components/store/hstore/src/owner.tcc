/*
   Copyright [2018-2021] [IBM Corporation]
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
 * Hopscotch hash table - template Key, Value, and allocators
 */

/*
 * ===== owner =====
 */

template <typename Lock>
	auto impl::owner::owned(std::size_t hop_hash_size_, Lock &s) const -> std::string
	{
		std::string st = "";
#if HSTORE_TRACE_OWNER
		if ( _pos != pos_undefined )
		{
			auto pos = _pos;
			std::string delim = "";
			for ( auto v = ownership_bits(s); v; v>>=1, ++pos %= hop_hash_size_ )
			{
				if ( v & mask_from_pos(0U) )
				{
					st += delim + std::to_string(pos);
					delim = " ";
				}
			}
		}
#else
		(void)hop_hash_size_;
		(void)s;
		st = "not tracked";
#endif
		return "(" + st + ")";
	}
