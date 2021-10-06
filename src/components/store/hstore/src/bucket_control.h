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


#ifndef _MCAS_HSTORE_BUCKET_CONTROL_H
#define _MCAS_HSTORE_BUCKET_CONTROL_H

#include "bucket_control_unlocked.h"

#include "bucket_mutexes.h"
#include <memory> /* unique_ptr */

namespace impl
{
	template <typename Bucket, typename Mutex>
		struct bucket_control
			: public bucket_control_unlocked<Bucket>
		{
			using base = bucket_control_unlocked<Bucket>;
			using typename base::six_t;
			using typename base::bucket_aligned_t;

			std::unique_ptr<bucket_mutexes<Mutex>[]> _bucket_mutexes;

			explicit bucket_control(
				six_t index_
				, bucket_aligned_t *buckets_
				, bool can_reconstitute_
			)
				: bucket_control_unlocked<Bucket>(index_, buckets_, can_reconstitute_)
				, _bucket_mutexes(nullptr)
			{
			}

			explicit bucket_control()
				: bucket_control(0U, nullptr, false)
			{
			}

			void extend(
				bucket_aligned_t *more_
				, bucket_control_unlocked<Bucket> *prev_
				, bucket_control_unlocked<Bucket> *next_
				, six_t segment_count_
			)
			{
				base::extend(more_, prev_, next_, segment_count_);
				_bucket_mutexes.reset(new bucket_mutexes<Mutex>[base::segment_size()]);
			}

			bucket_control(bucket_control &&) noexcept = default;
		};
}

#endif
