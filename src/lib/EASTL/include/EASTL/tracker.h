/////////////////////////////////////////////////////////////////////////////
// Copyright (c) Electronic Arts Inc. All rights reserved.
/////////////////////////////////////////////////////////////////////////////

#ifndef EASTL_TRACKER_H
#define EASTL_TRACKER_H

#include <EASTL/region_modifications.h>

#include <cstddef> /* size_t */

////////////////////////////////////////////////////////////////////////////////////////////
// Support for tracked writes.
////////////////////////////////////////////////////////////////////////////////////////////

namespace eastl
{
	struct NupmTracker
	{
		inline void
			track_pre(
				const void *
				, std::size_t
				, char = '0'
			) const noexcept
		{
		}

		inline void
			track_post(
				const void *p
				, std::size_t s
				, char c = '1'
			) const noexcept
		{
			nupm::region_tracker_add(p, s, c);
		}
	protected:
		~NupmTracker() = default;
	};

	class DummyTracker
	{
	protected:
		~DummyTracker() = default;
	public:
		inline void track_pre(const void *, std::size_t, char = 'x') const noexcept { }
		inline void track_post(const void *, std::size_t, char = 'x') const noexcept { }
	};

	using  DefaultTracker = NupmTracker;

} // namespace eastl

#endif // EASTL_INTERNAL_TRACED_H

