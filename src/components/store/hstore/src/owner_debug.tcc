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
 * Hopscotch hash owner debug
 */

/*
 * ===== owner =====
 */

template <typename Lock>
	auto impl::operator<<(
		std::ostream &o_
		, const owner_print<Lock> &op_
	) -> std::ostream &
	{
		const auto &w = static_cast<const owner &>(op_.sb().deref());
		return o_
			<< "(owner "
			<< w.owned(op_.bucket_count(), op_.lock())
			<< ") adj " << (w.is_adjacent_content_in_use() ? "IN_USE" : "FREE");
	}

template <typename TableBase>
	auto impl::operator<<(
		std::ostream &o_
		, const owner_print<bypass_lock<typename TableBase::bucket_t, const owner>> &op_
	) -> std::ostream &
	{
		const auto &w = static_cast<const owner &>(op_.sb().deref());
		bypass_lock<typename TableBase::bucket_t, const owner> lk(w, op_.sb());
		return o_
			<< "(owner owns"
			<< w.owned(op_.get_table().bucket_count(), lk)
			<< ") adj " << (w.is_adjacent_content_in_use() ? "IN_USE" : "FREE");
	}

