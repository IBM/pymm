/*
   Copyright [2019-2020] [IBM Corporation]
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

#ifndef MCAS_CCPM_TRACKER_LOG_H
#define MCAS_CCPM_TRACKER_LOG_H

#include <ccpm/cca.h>
#include <ccpm/log.h>

#include <cstddef> // size_t

namespace ccpm
{
	struct tracker_log
	{
		log *_log; // not owned

		inline void
			track_pre(
				const void *v
				, std::size_t s
				, char = '0'
			) const noexcept
		{
			/* tracker does not expect *v to be modified, but rollback
			 * will do just that. Perhaps track_pre and track_post parameters
			 * should be void *, not const void *.
			 */
			if ( _log->includes(v) )
			{
				_log->add(const_cast<void *>(v), s);
			}
		}

		inline void
			track_post(
				const void *
				, std::size_t
				, char = '1'
			) const noexcept
		{
		}

		/* after a successful allocate, log the allocate */
		inline void track_allocate(void *p, std::size_t n) const noexcept
		{
			_log->allocated(p, n);
		}

		inline void track_free(void *p, std::size_t n) const noexcept
		{
			_log->freed(p, n);
		}
	protected:
		explicit tracker_log(log *log_)
			: _log(log_)
		{}
	public:
		tracker_log() : _log(nullptr) {} // requied by ListNodeBase
		tracker_log(const tracker_log &) = default;
		tracker_log &operator=(const tracker_log &) = default;
		~tracker_log() = default;
	};
}

#endif
