/*
   Copyright [2018-2020] [IBM Corporation]
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

#include <type_traits> /* remove_const */
#include <utility> /* move */

#include "perishable.h"

/*
 * ===== content =====
 */

template <typename Value>
	impl::content<Value>::content()
		: _value()
#if TRACK_OWNER
		, _owner(owner_undefined)
#endif
	{
#if 0 && ENABLE_TIMESTAMPS
		std::get<1>(_value.second).set_free();
#endif
	}

template <typename Value>
	void impl::content<Value>::set_owner(owner_t
#if TRACK_OWNER
		owner_
#endif
	)
	{
#if TRACK_OWNER
		_owner = owner_;
#endif
	}

template <typename Value>
	auto impl::content<Value>::content_erase() -> void
	{
		_value.~value_type();
		set_owner(owner_undefined);
	}

template <typename Value>
	auto impl::content<Value>::content_share(
		const content &sr_
		, std::size_t bi_
	) -> content &
	{
		using k_t = typename value_type::first_type;
		using m_t = typename value_type::second_type;
		new
			(&const_cast<std::remove_const_t<k_t> &>(_value.first))
			k_t(sr_._value.first)
			;
		new (&_value.second) m_t(sr_._value.second);
		set_owner(bi_);
		return *this;
	}

template <typename Value>
	auto impl::content<Value>::content_share(
		content &from_
	) -> content &
	{
		using k_t = typename value_type::first_type;
		using m_t = typename value_type::second_type;
		new
			(&const_cast<std::remove_const_t<k_t> &>(_value.first))
			k_t(from_._value.first)
			;
		new (&_value.second) m_t(from_._value.second);
		set_owner(from_.get_owner());
		return *this;
	}

template <typename Value>
	template <typename ... Args>
		auto impl::content<Value>::content_construct(
			std::size_t bi_
			, Args && ... args_
		) -> content &
		{
			new (&_value) Value(std::forward<Args>(args_)...);
			set_owner(bi_);
			return *this;
		}

template <typename Value>
	constexpr char impl::content<Value>::lock_id;
