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

#ifndef MCAS_CCPM_ALLOCATOR_TL_H
#define MCAS_CCPM_ALLOCATOR_TL_H

#include <ccpm/tracker_log.h>

#include <ccpm/cca.h>
// #include <ccpm/log.h>

#include <cstddef> // size_t
#include <new> // bad_alloc

struct bad_alloc_tl
	: public std::bad_alloc
{
	bad_alloc_tl()
		: std::bad_alloc()
	{}
	const char * what() const noexcept override { return "bad alloc in allocator_tl"; }
};

namespace ccpm
{
	struct allocator_tl
		: public tracker_log
	{
	private:
		cca *_cca; // not owned
	public:
		using tracker_type = tracker_log;
		explicit allocator_tl(cca *cca_, log *log_)
			: tracker_log(log_)
			, _cca(cca_)
		{}
		allocator_tl(const allocator_tl& x) = default;

		allocator_tl& operator=(const allocator_tl& x) = default;

		void* allocate(size_t n, int flags = 0)
		{
			return allocate(n, sizeof(void *), 0, flags);
		}
		void* allocate(std::size_t n, std::size_t alignment, std::size_t offset, int flags = 0)
		{
			(void)flags;
			(void)offset;
			void *p = nullptr;
			_cca->allocate(p, n, alignment);
			if ( !p )
			{
				throw bad_alloc_tl{};
			}
			track_allocate(p, n);
			return p;
		}
		void deallocate(void *p, size_t n)
		{
			/* Do not free until log is cleared (i.e, committed) */
			void *pl = p;
			track_free(pl, n);
		}

		const char* get_name() const { return "bob"; }
		void        set_name(const char *) {}
	};

	bool operator==(const allocator_tl& a, const allocator_tl& b); // unused
	bool operator!=(const allocator_tl& a, const allocator_tl& b); // unused

}

#endif
