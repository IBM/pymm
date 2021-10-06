/*
   Copyright [2018-2019] [IBM Corporation]
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
 * Hopscotch hash content debug
 */

#include "cond_print.h"

#include <cassert>
#include <cstddef> /* size_t */
#include <sstream> /* ostringstream */

/*
 * ===== content =====
 */

/* NOTE: Not reliable if recovering from a crash */
#if TRACED_CONTENT
template <typename Value>
	auto impl::content<Value>::is_clear() const noexcept -> bool
	{
		return _owner == owner_undefined;
	}

template <typename Value>
	auto impl::content<Value>::to_string() const -> std::string
	{
		return
			descr()
			;
	}

template <typename Value>
	auto impl::content<Value>::descr() const -> std::string
	{
		std::ostringstream s;
		s << "(owner " << _owner;
		if ( _owner != owner_undefined )
		{
			s << " " << cond_print(key(),"(unprintable key)")
			<< "->"
			<< cond_print(mapped(), "(unprintable mapped)")
			<< ")"
			;
		}
		return s.str();
	}
#endif

template <typename Value>
	void impl::content<Value>::owner_verify(content::owner_t owner_) const
	{
		if ( _owner != owner_ )
		{
			hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION, "source owned by ", _owner, " was about to be moved by ", owner_);
		}
		assert(_owner == owner_);
	}

template <typename Value>
	void impl::content<Value>::owner_update(owner_t owner_delta)
	{
		_owner |= owner_delta;
	}

template <typename Value>
	auto impl::operator<<(
		std::ostream &o_
		, const content<Value> &c_
	) -> std::ostream &
	{
		return o_
			<< c_.to_string()
			;
	}
