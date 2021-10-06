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


#ifndef _MCAS_HSTORE_BUCKET_REF_H
#define _MCAS_HSTORE_BUCKET_REF_H

#include "hash_bucket.h"
#include "segment_and_bucket.h"

#include <boost/iterator/transform_iterator.hpp>

#include <cstddef> /* size_t */

namespace impl
{
	template <typename Bucket, typename Referent>
		struct bucket_ref
		{
			using segment_and_bucket_t = segment_and_bucket<Bucket>;
		private:
			Referent *_ref;
			segment_and_bucket_t _sb;
		public:
			bucket_ref(Referent *ref_, const segment_and_bucket_t &sb_)
				: _ref(ref_)
				, _sb(sb_)
			{
			}
			std::size_t index() const { return sb().index(); }
			const segment_and_bucket_t &sb() const { return _sb; }
			Referent &ref() const { return *_ref; }
			/* A "content" bucket_ref also owns the "in-use bit, which, for space reasons,
			 * is currently stored in the "owner" object.
			 * The holder of a content lock can get the owner ref for the sole purpose of
			 * accessing the in-use bit.
			 */
			owner &owner_ref() const { return *static_cast<hash_bucket<typename Referent::value_type> *>(_ref); }
		};
}

#endif
