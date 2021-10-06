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


#ifndef MCAS_HSTORE_BUCKET_CONTROL_UNLOCKED_
#define MCAS_HSTORE_BUCKET_CONTROL_UNLOCKED_

#include "hstore_config.h"
#include "bucket_aligned.h"
#include "hop_hash_log.h"
#include "segment_layout.h"
#include "trace_flags.h"
#include <array>
#include <cassert>
#include <cstddef> /* size_t */


namespace impl
{
	template <typename Bucket>
		struct bucket_control_unlocked
		{
		private:
			unsigned bit_count_v(typename Bucket::owner_type::value_type v)
			{
				unsigned count = 0;
				for ( ; v; v &= (v - 1) )
				{
					++count;
				}
				return count;
			}
			void report()
			{
				/* Report the distribution of owned element counts for each owner.
				 * We do now wrap, so the first owner::size element counts will wrongly counted.
				 */
				std::array<unsigned, bucket_type::owner_type::size> h{};
				typename bucket_type::owner_type::value_type all_owners_mask = 0U;
				for ( auto it = _buckets; it != _buckets_end; ++it )
				{
					typename bucket_type::owner_type &o = *it;
					/* cheat: owner::value will take *any* reference as a Lock */
					int lock = 0;
					auto bit_count = bit_count_v(o.ownership_bits(lock));
					++h[bit_count];
				}
				{
					std::ostringstream hs{};
					for ( const auto &hn : h )
					{
						hs << " " << hn;
					}
					hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION, index(), ":", hs.str());
				}
				/* Report the distribution of owned element counts in each ownership range.
				 * We do now wrap, so the first owner::size element counts will wrongly counted. */
				unsigned j[bucket_type::owner_type::size] = {};
				for ( auto it = _buckets; it != _buckets_end; ++it )
				{
					typename bucket_type::owner_type &o = *it;
					int lock = 0;
					/* cheat: owner::ownership_bit_mask will take *any* reference as a Lock */
					all_owners_mask >>= 1U;
					assert((all_owners_mask & o.ownership_bits(lock)) == 0);
					all_owners_mask |= o.ownership_bits(lock);
					auto bit_count = bit_count_v(all_owners_mask);
					++j[bit_count];
				}
				{
					std::ostringstream js{};
					for ( const auto &jn : j )
					{
						js << " " << jn;
					}
					hop_hash_log<HSTORE_TRACE_MANY>::write(LOG_LOCATION, index(), ":", js.str());
				}
			}

			void deconstitute()
			{
				if ( _can_reconstitute )
				{
					for ( auto it = _buckets; it != _buckets_end; ++it )
					{
						typename bucket_type::owner_type &w = *it;
						/* deconsititute key and value */
						if ( w.is_adjacent_content_in_use() )
						{
							typename bucket_type::content_type &c = *it;
							c.value().first.deconstitute();
							/* ERROR: depends on the types of first and second,
							 * so should be handled by the session level, not here
							 */
							std::get<0>(c.value().second).deconstitute();
						}
					}
				}
			}

		public:
			using bucket_type = Bucket;
			using bucket_aligned_t = bucket_aligned<Bucket>;
			using six_t = std::size_t;
			using bix_t = std::size_t;
		private:
			six_t _index;
			std::size_t _bi_mask;
		public:
			bucket_control_unlocked<Bucket> *_prev; /* public only for initial linknig by hop_hash_base */
			bucket_control_unlocked<Bucket> *_next;
			bucket_aligned_t *_buckets; /* public only so that transform can access the element */
		private:
			bucket_aligned_t *_buckets_end;
			bool _can_reconstitute;


			unsigned log_segment_size() const
			{
				return unsigned( segment_layout::log2_base_segment_size + (_index == 0U ? 0U : (_index-1U) ) );
			}

			std::size_t segment_size_from_log() const {
				return std::size_t(1) << log_segment_size();
			}
		public:

			bucket_control_unlocked(
				six_t index_
				, bucket_aligned_t *buckets_
				, bool can_reconstitute_
			)
				: _index(index_)
				, _bi_mask(segment_size_from_log()-1U)
				, _prev(this)
				, _next(this)
				, _buckets(buckets_)
				, _buckets_end(
					_buckets
					? _buckets + segment_size()
					: nullptr
				)
				, _can_reconstitute(can_reconstitute_)
			{
			}
			bucket_control_unlocked(const bucket_control_unlocked &) = delete;
			bucket_control_unlocked(bucket_control_unlocked &&) noexcept = default;

			~bucket_control_unlocked()
			{
#if HSTORE_TRACE_MANY
				/* report statistics for in-use segments */
				if ( _buckets != _buckets_end )
				{
					report();
				}
#endif
				deconstitute();
			}

			bucket_control_unlocked operator=(const bucket_control_unlocked &) = delete;

			/* Should be replaced by a placement constructor */
			void extend(
				bucket_aligned_t *more_
				, bucket_control_unlocked *prev_
				, bucket_control_unlocked *next_
				, six_t index_
			)
			{
				_buckets = more_;
				_prev = prev_;
				_next = next_;
				_index = index_;
				_bi_mask = segment_size_from_log()-1U;
				_buckets_end = _buckets + segment_size();
			}

			std::size_t bi_mask() const
			{
				return _bi_mask;
			}
			six_t index() const { return _index; }
			bucket_aligned_t *buckets() const { return _buckets; }
			bucket_aligned_t *buckets_end() const { return _buckets_end; }
			bucket_control_unlocked<Bucket> *prev() const { return _prev; }
			bucket_control_unlocked<Bucket> *next() const { return _next; }
			std::size_t segment_size() const { return bi_mask() + 1U; }
			bucket_aligned_t &deref(bix_t bi) const { return _buckets[bi]; }

			template <typename Allocator>
				void reconstitute(
					Allocator av_
				)
				{
					for ( auto it = _buckets; it != _buckets_end; ++it )
					{
						typename bucket_type::owner_type &w = *it;
						typename bucket_type::content_type &c = *it;
						/* reconsititute key and value */
						if ( w.is_adjacent_content_in_use() )
						{
							/* ERROR: depends on the types of first and second,
							 * so should be handled by the session level, not here
							 */
							const_cast<
								typename std::remove_const<typename bucket_type::content_type::key_t>::type &
							>(c.value().first).reconstitute(av_);
							std::get<0>(c.value().second).reconstitute(av_);
						}
					}
				}
		};
}

#endif
