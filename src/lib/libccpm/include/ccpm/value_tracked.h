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

#ifndef MCAS_CCPM_VALUE_TRACKED_H
#define MCAS_CCPM_VALUE_TRACKED_H

#include <type_traits> // is_class

namespace ccpm
{
	template<typename T, typename Tracker, bool IsClass = std::is_class<T>::value>
		struct value_tracked;

	template<typename T, typename Tracker>
		struct value_tracked<T, Tracker, false>
		{
			using value_type = T;
			// using const_reference = const value_type &;
			using tracker_type = Tracker;
		private:
			value_type _v;
			tracker_type *_t; // not owned
		public:
			value_tracked(T v_, tracker_type &t_)
				: _v(v_)
				, _t(&t_)
			{
				_t->track_post(this, sizeof *this);
			}

			value_tracked()
				: _v(T())
				, _t(nullptr)
			{}
			value_tracked(const value_tracked &) = default;
			value_tracked &operator=(const value_tracked &o_)
			{
				if ( ! _t )
				{
					_t = o_._t;
				}

				_t->track_pre(this, sizeof *this);
				_v = o_._v;
				_t->track_post(this, sizeof *this);
				return *this;
			}
			~value_tracked()
			{
				if ( _t )
				{
					_t->track_pre(this, sizeof *this);
				}
			}

			/* an implicit output conversion from wrapper to wrapped type is
			 * not generally a good idea, but in this case it avoids complexity
			 * when using algorithms such as std::equal
			 */
			operator value_type() const { return _v; }
		};

	template<typename T, typename Tracker>
		struct value_tracked<T, Tracker, true>
			: public T
		{
			using value_type = T;
			// using const_reference = const value_type &;
			using tracker_type = Tracker;
		private:
			tracker_type *_t; // not owned
		public:
			value_tracked(T v_, tracker_type &t_)
				: T(v_)
				, _t(&t_)
			{
				_t->track_post(this, sizeof *this);
			}

			value_tracked()
				: T(T())
				, _t(nullptr)
			{}
			value_tracked(const value_tracked &) = default;
			value_tracked &operator=(const value_tracked &o_)
			{
				if ( ! _t )
				{
					_t = o_._t;
				}

				_t->track_pre(this, sizeof *this);
				static_cast<T &>(*this) = o_;
				_t->track_post(this, sizeof *this);
				return *this;
			}
			~value_tracked()
			{
				if ( _t )
				{
					_t->track_pre(this, sizeof *this);
				}
			}
		};
}

#endif
