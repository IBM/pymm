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


#ifndef _MCAS_HSTORE_OWNER_H
#define _MCAS_HSTORE_OWNER_H


#include "hstore_config.h"
#include "persistent.h"
#include "trace_flags.h"
#if TRACED_OWNER
#include "hop_hash_debug.h"
#endif

#include <gsl/pointers>
#include <cassert>
#include <cstddef> /* size_t */
#include <cstdint> /* uint64_t */
#include <limits> /* numeric_limits */
#include <string>

/*
 * The "owner" part of a hash bucket
 */

namespace impl
{
	struct allocation_state_emplace;

	template <typename Bucket, typename Referent, typename Lock>
		struct bucket_shared_lock;
	template <typename Bucket, typename Referent, typename Lock>
		using bucket_shared_ref = bucket_shared_lock<Bucket, Referent, Lock> &;

	template <typename Bucket, typename Referent, typename Lock>
		struct bucket_unique_lock;
	template <typename Bucket, typename Referent, typename Lock>
		using bucket_unique_ref = bucket_unique_lock<Bucket, Referent, Lock> &;

	template <bool Trace>
		struct pos_trace;

	template <>
		struct pos_trace<true>
		{
		protected:
			std::size_t _pos;
			pos_trace();
			void set_pos(std::size_t pos_, unsigned p_);
		};

	template <>
		struct pos_trace<false>
		{
		protected:
			pos_trace() {}
			void set_pos(std::size_t, unsigned) {}
		};

	struct owner
		: public pos_trace<TRACED_OWNER>
	{
		using index_type = unsigned;
		static constexpr index_type size = 63U;
		using value_type = std::uint64_t; /* sufficient for size not over 64U */
		static constexpr auto pos_undefined = std::numeric_limits<std::size_t>::max();
		static constexpr char lock_id = 'w';
	private:
		persistent_atomic_t<value_type> _value; /* at least owner::size bits */
		static constexpr auto mask_from_pos(index_type pos) -> value_type { return value_type(1U) << pos; }
		void insert(
			const std::size_t pos_
			, const index_type p_
		)
		{
			set_pos(pos_, p_);
			_value |= mask_from_pos(p_);
		}
		void set_adjacent_content_free() { _value &= ~ adjacent_content_in_use_mask(); }
	public:
		explicit owner()
			: _value(0)
		{}

		static constexpr value_type ownership_bit_mask() { return mask_from_pos(size) - 1U; }
		static constexpr value_type adjacent_content_in_use_mask() { return mask_from_pos(size); }
		static index_type nearest_owned_content_offset(value_type c)
		{
			return index_type(__builtin_ctzll(c & ownership_bit_mask()));
		}

		/* ERROR: it is being a bit lazy to make PersistController a template parameter.
		 * Chances are that what we need of the persist_map_controller_t could be provided
		 * in a more limited manner.
		 */
		template<typename Bucket, typename Referent, typename SharedMutex, typename PersistController>
			void insert(
				const std::size_t pos_
				, const index_type p_
				, bucket_unique_ref<Bucket, Referent, SharedMutex>
				, gsl::not_null<PersistController *> pc_
			)
			{
				if ( pc_->pool()->is_crash_consistent() )
				{
					/* includes setting of doubt type to emplace */
					pc_->em_record_owner_addr_and_bitmask(&_value, mask_from_pos(p_));
				}
				insert(pos_, p_);
			}
		template<typename Bucket, typename Referent, typename SharedMutex, typename PersistController>
			void erase(
				index_type p_
				, bucket_unique_ref<Bucket, Referent, SharedMutex>
				, gsl::not_null<PersistController *> pc_
			)
			{
				if ( pc_->pool()->is_crash_consistent() )
				{
					/* includes setting of doubt type to emplace */
					pc_->er_record_owner_addr_and_bitmask(&_value, mask_from_pos(p_));
				}
				_value &= ~mask_from_pos(p_);
			}
		template<typename Bucket, typename Referent, typename SharedMutex>
			void move(
				index_type dst_
				, index_type src_
				, bucket_unique_ref<Bucket, Referent, SharedMutex>
			)
			{
				assert(dst_ < size);
				assert(src_ < size);
				_value = (_value | mask_from_pos(dst_)) & ~mask_from_pos(src_);
			}

		bool is_adjacent_content_in_use() const { return _value & adjacent_content_in_use_mask(); }
		void set_adjacent_content_in_use(bool in_use)
		{
			if ( in_use )
			{
				set_adjacent_content_in_use();
			}
			else
			{
				set_adjacent_content_free();
			}
		}
		void set_adjacent_content_in_use() { _value |= adjacent_content_in_use_mask(); }

		template <typename Lock>
			auto ownership_bits(Lock &) const -> value_type { return _value & ownership_bit_mask(); }

		template <typename Lock>
			auto is_in_use(Lock &, const index_type p_) const -> bool { return (_value >> p_) & value_type(1U); }
		auto is_in_use(const index_type p_) const -> bool { return (_value >> p_) & value_type(1U); }
		template <typename Lock>
			auto owned(std::size_t hop_hash_size, Lock &) const -> std::string;
		/* clear the senior owner of all the bits set in its new junior owner. */
		template <typename Bucket, typename Referent, typename SharedMutex>
			void clear_from(
				const owner &junior
				, bucket_unique_ref<Bucket, Referent, SharedMutex>
				, bucket_shared_ref<Bucket, Referent, SharedMutex>
			)
			{
				_value &= ~junior._value;
			}

#if TRACED_OWNER
		template <
			typename Lock
		>
			friend auto operator<<(
				std::ostream &o
				, const impl::owner_print<Lock> &
			) -> std::ostream &;
		template <typename Table>
			friend auto operator<<(
				std::ostream &o
				, const impl::owner_print<impl::bypass_lock<const typename Table::bucket_t, const impl::owner>> &
			) -> std::ostream &;
#endif
	};

	inline pos_trace<true>::pos_trace()
		: _pos(owner::pos_undefined)
	{}

	inline void pos_trace<true>::set_pos(std::size_t pos_,
                                       unsigned p_
                                       )
	{
		assert(_pos == owner::pos_undefined || _pos == pos_);
		(void)p_;
		assert(p_ < owner::size);
		_pos = pos_;
	}
}

#endif
