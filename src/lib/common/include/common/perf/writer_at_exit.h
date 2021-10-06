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

#ifndef MCAS_COMMON_PERF_WRITER_AT_EXIT_H
#define MCAS_COMMON_PERF_WRITER_AT_EXIT_H

#include <common/string_view.h>
#include <string>
#include <ostream>

namespace common
{
	namespace perf
	{
		template<class T>
			auto bool_or_true(T const &t, int)
				-> decltype(bool(t), bool())
			{
				return bool(t);
			}
		template<class T>
			auto bool_or_true(T const &, long)
				-> bool
			{
				return true;
			}
		/*
		 * Class which prints its derived class, with a "tag", upon destruction.
		 */
		template <typename E>
			class writer_at_exit
				: public E
			{
				/*
				 * TODO: add a means to print a hierarchical tag.
				 * But it may be simpler to place the hiearachical logic
				 * in the including class.
				 */
				std::ostream &_o;
				std::string   _tag;
			public:
				template <typename ... Args>
					writer_at_exit(std::ostream &o_, std::string tag_, Args ... args)
						: E{args ...}
						, _o{o_}
						, _tag{tag_}
					{}
				template <typename ... Args>
					writer_at_exit(std::ostream &o_, long line_, common::string_view file_, common::string_view func_, common::string_view tag_, Args ... args)
						: E{args ...}
						, _o{o_}
						, _tag{std::string{func_.begin(), func_.end()} + ":" + std::string(tag_.begin(), tag_.end())}
					{
						(void)line_;
						(void)file_;
					}
				~writer_at_exit()
				{
					if ( true || bool_or_true(*this, 0) )
					{
#if 0
						_o << _tag << " " << *this << "\n";
#else
						_o << *this << " " << _tag << "\n";
#endif
					}
				}
			};
	}
}

#endif
