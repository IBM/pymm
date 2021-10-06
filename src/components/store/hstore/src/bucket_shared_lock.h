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

#ifndef _MCAS_HSTORE_BUCKET_SHARED_LOCK_H
#define _MCAS_HSTORE_BUCKET_SHARED_LOCK_H

#include "bucket_ref.h"
#include <shared_mutex> /* shared_lock */
#include <utility> /* move */

namespace impl
{
	template <typename Bucket, typename Referent, typename SharedMutex>
		struct bucket_shared_lock
			: public bucket_ref<Bucket, Referent>
			, public std::shared_lock<SharedMutex>
		{
			using base_ref = bucket_ref<Bucket, Referent>;
			using base_lock = std::shared_lock<SharedMutex>;
			using typename base_ref::segment_and_bucket_t;
			bucket_shared_lock(
				Bucket &b_
				, const segment_and_bucket_t &i_
				, SharedMutex &m_
			)
				: base_ref(&b_, i_)
				, base_lock(m_)
			{
				hop_hash_log<HSTORE_TRACE_LOCK>::write(LOG_LOCATION, this->index());
			}
			bucket_shared_lock(bucket_shared_lock &&) noexcept = default;
			auto operator=(bucket_shared_lock &&other_) noexcept -> bucket_shared_lock &
			{
				hop_hash_log<HSTORE_TRACE_LOCK>::write(LOG_LOCATION, this->index(), "->", other_.index());
				base_lock::operator=(std::move(other_));
				base_ref::operator=(std::move(other_));
				return *this;
			}
			~bucket_shared_lock()
			{
				if ( this->owns_lock() )
				{
					hop_hash_log<HSTORE_TRACE_LOCK>::write(LOG_LOCATION, this->index());
				}
			}
		};
}

#endif
