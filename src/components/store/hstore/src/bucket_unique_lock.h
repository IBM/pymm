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

#ifndef _MCAS_HSTORE_BUCKET_UNIQUE_LOCK_H
#define _MCAS_HSTORE_BUCKET_UNIQUE_LOCK_H

#include "bucket_ref.h"
#include <mutex> /* unique_lock */
#include <utility> /* move */

namespace impl
{
	template <typename Bucket, typename Referent, typename SharedMutex>
		struct bucket_unique_lock
			: public bucket_ref<Bucket, Referent>
			, public std::unique_lock<SharedMutex>
		{
			using base_ref = bucket_ref<Bucket, Referent>;
			using typename base_ref::segment_and_bucket_t;
			bucket_unique_lock(
				Referent &b_
				, const segment_and_bucket_t &i_
				, SharedMutex &m_
			)
				: bucket_ref<Bucket, Referent>(&b_, i_)
				, std::unique_lock<SharedMutex>((hop_hash_log<HSTORE_TRACE_LOCK>::write(LOG_LOCATION, this->index(), Referent::lock_id), m_))
			{
			}
			bucket_unique_lock(bucket_unique_lock &&) noexcept = default;
			bucket_unique_lock &operator=(bucket_unique_lock &&other_) noexcept
			{
				hop_hash_log<HSTORE_TRACE_LOCK>::write(LOG_LOCATION, this->index(), Referent::lock_id
					, "->", other_.index(), Referent::lock_id);
				std::unique_lock<SharedMutex>::operator=(std::move(other_));
				base_ref::operator=(std::move(other_));
				return *this;
			}
			~bucket_unique_lock()
			{
				if ( this->owns_lock() )
				{
					hop_hash_log<HSTORE_TRACE_LOCK>::write(LOG_LOCATION, this->index(), Referent::lock_id);
				}
			}
			template <typename HopHash>
				void assert_clear(bool b, HopHash &t)
				{
					this->ref().assert_clear(b, *this, t);
				}
		};
}

#endif
