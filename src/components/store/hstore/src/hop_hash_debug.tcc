/*
   Copyright [2018-2019] [IBM Corporation]
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

/*
 * Hopscotch hash hop_hash debug
 */

#include <cassert>

#include "owner.h"
#include "owner_debug.tcc"
#include "content_debug.tcc"
#include "bits_to_ints.h"
#include <boost/io/ios_state.hpp>
#include <algorithm> /* set_difference */
#include <iostream> /* ostream */

template <typename LockOwner, typename LockContent>
	impl::bucket_print<LockOwner, LockContent>::bucket_print(
		const std::size_t ct_
		, LockOwner &c_
		, LockContent &i_
	)
		: _ct(ct_)
		, _c{&c_}
		, _i{&i_}
	{
		assert(c_.index() == i_.index());
	}

/*
 * ===== hash_bucket =====
 */

template <typename LockOwner, typename LockContent>
	auto impl::operator<<(
		std::ostream &o_
		, const bucket_print<LockOwner, LockContent> &p_
	) -> std::ostream &
	{
		const auto &b = p_.sb().deref();
		return o_
			<< "( "
			<< dump<true>::make_owner_print(p_.bucket_count(), p_.lock_owner())
			<< " "
			<< b
			<< " )";
	}

#if 0
template <typename TableBase>
	auto impl::operator<<(
		std::ostream &o_
		, const bucket_print
			<
				bypass_lock<typename TableBase::bucket_t, owner>
				, bypass_lock<typename TableBase::bucket_t, content<typename TableBase::bucket_t>>
			> &p_
	) -> std::ostream &
	{
		const auto &b = p_.sb().deref();
		auto lk_shared_owner = bypass_lock<typename TableBase::bucket_t, owner>(p_.index());
		return o_
			<< "( "
			<< dump<true>::make_owner_print(p_.bucket_count(), lk_shared_owner)
			<< " "
			<< b
			<< " )";
	}
#endif

#include "cond_print.h"

#include <cstddef> /* size_t */
#include <set>
#include <vector>

/*
 * ===== hop_hash =====
 */

template <typename TableBase>
	auto impl::operator<<(
		std::ostream &o_
		, const hop_hash_print<TableBase> &t_
	) -> std::ostream &
	{
		auto &tbl = t_.get_hop_hash();
		for ( const auto &k : tbl )
		{
			o_ << cond_print(k.first, "(key)") << " -> "
				<< cond_print(k.second, "(mapped)") << "\n";
		}
		return o_;
	}

template <typename TableBase>
	auto impl::operator<<(
		std::ostream &o_
		, const dump<false>::hop_hash_dump<TableBase> & // t_
	) -> std::ostream &
	{
		return o_;
	}
#if 0
inline std::vector<int> bits_to_ints(std::uint64_t b, int offset)
{
	std::vector<int> v;
	for ( int ix = 0; b != 0; b>>=1, ++ix)
	{
		if ( b & 1 )
		{
			v.push_back(ix + offset);
		}
	}
	return v;
}

inline std::string ints_to_string(std::vector<int> v)
{
	std::string s = "(";
	for ( auto i : v )
	{
		s += std::to_string(i) + " ";
	}
	return s + ")";
}
#endif
template <typename TableBase>
	auto impl::operator<<(
		std::ostream &o_
		, const dump<true>::hop_hash_dump<TableBase> &t_
	) -> std::ostream &
	{
		auto &tbl_base = t_.get_hop_hash();
		o_ << "Buckets\n";
		std::set<std::size_t> owners;
		std::set<std::size_t> contents;
		std::size_t in_use_count = 0;
		for ( std::size_t k = 0; k != tbl_base.bucket_count(); ++k )
		{
			auto sb = tbl_base.make_segment_and_bucket(k);
			bypass_lock<typename TableBase::bucket_t, const owner> owner_lk(tbl_base.locate_owner(sb), sb);
			bypass_lock<typename TableBase::bucket_t, const content<typename TableBase::value_type>>
				content_lk(
					static_cast<const typename TableBase::content_t &>(sb.deref())
					, sb
				);

			auto v = owner_lk.ref().ownership_bits(owner_lk);
			if (
				v != 0
				||
				owner_lk.ref().is_adjacent_content_in_use()
			)
			{
				{
					boost::io::ios_flags_saver s(o_);
					o_ << k << ": " << std::hex << v << "=" << ints_to_string(bits_to_ints(v, k));
				}

				for ( auto m = k; v != 0; v >>= 1, m = (m + 1) % tbl_base.bucket_count() )
				{
					/* v claims ownership of m */
					if ( v & 1 )
					{
						if ( ! owners.insert(m).second )
						{
							o_ << "MULTIPLE OWNERS for " << m << " ";
						}
						else
						{
						}
					}
				}
				o_ << ": "
					<< make_bucket_print(tbl_base.bucket_count(), owner_lk, content_lk)
					<< "\n";
				if ( owner_lk.ref().is_adjacent_content_in_use() ) { contents.insert(k); }
			}
		}

		if ( contents.size() != owners.size() )
		{
			o_ << "MISMATCH: content count " << in_use_count << " owner count " << owners.size();
			{
				o_ << " extra content ";
				std::vector<std::size_t> ec;
				std::set_difference(contents.begin(), contents.end(), owners.begin(), owners.end(), std::back_inserter(ec));
				for ( auto e : ec )
				{
					o_ << e << " ";
				}
			}
			{
				o_ << " extra owners ";
				std::vector<std::size_t> ec;
				std::set_difference(owners.begin(), owners.end(), contents.begin(), contents.end(), std::back_inserter(ec));
				for ( auto e : ec )
				{
					o_ << e << " ";
				}
			}

			o_ << "\n";
		}

		if ( ! tbl_base.segment_count_actual().is_stable() )
		{
			auto &loc = tbl_base._bc[tbl_base.segment_count_not_stable()];
			if ( loc._buckets )
			{
				o_ << "Pending buckets\n";
				for ( std::size_t ks = 0; ks != tbl_base.bucket_count(); ++ks )
				{
					const auto kj = tbl_base.bucket_count() + ks;
					const auto sbj = tbl_base.make_segment_and_bucket_unsafe(kj);
					bypass_lock<typename TableBase::bucket_t, const owner> owner_lk(loc._buckets[ks], sbj);
					bypass_lock<typename TableBase::bucket_t, const content<typename TableBase::value_type>>
						content_lk(
							loc._buckets[ks]
							, sbj
						);
					if (
						owner_lk.ref().ownership_bits(owner_lk) != 0
						||
						owner_lk.ref().is_adjacent_content_in_use()
					)
					{
						o_ << kj << ": "
							<< make_bucket_print(tbl_base.bucket_count(), owner_lk, content_lk)
							<< "\n";
					}
				}
			}
			else
			{
				o_ << "Resize in progress but no pending buckets\n";
			}
		}
		return o_;
	}
