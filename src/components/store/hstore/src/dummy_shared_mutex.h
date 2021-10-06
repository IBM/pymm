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


#ifndef MCAS_HSTORE_DUMMY_SHARED_MUTEX_H_
#define MCAS_HSTORE_DUMMY_SHARED_MUTEX_H_

#include <cassert>
#include <system_error>

namespace dummy
{
	struct shared_mutex
	{
		int _state; /* 0 => free, -1 => unique, other => shared_count */
	public:
		shared_mutex()
			: _state(0)
		{}
		/* BasicLockable */
		void lock()
		{
			if ( _state != 0 )
			{
				/* dummy is for single-threaded testing. cannot really block */
				throw std::system_error(std::make_error_code(std::errc::resource_deadlock_would_occur /* , std::system_category() */));
			}
			_state = -1;
		}
		void unlock()
		{
			assert( _state == -1 );
			_state = 0;
		}
		/* Lockable */
		bool try_lock()
		{
			if ( _state == 0 )
			{
				lock();
				return true;
			}
			return false;
		}
		/* SharedMutex */
		void lock_shared()
		{
			assert( _state != -1 );
			++_state;
		}
		bool try_lock_shared()
		{
			if ( 0 <= _state )
			{
				lock_shared();
				return true;
			}
			return false;
		}
		void unlock_shared()
		{
			assert( 0 < _state );
			--_state;
		}
	};
}

#endif
