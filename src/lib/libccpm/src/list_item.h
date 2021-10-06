/*
   Copyright [2019, 2020] [IBM Corporation]
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

#ifndef CCPM_LIST_ITEM_H
#define CCPM_LIST_ITEM_H

#include <cassert>

#define MCAS_CCPM_LIST_CHECK 0
#define MCAS_CCPM_IN_LIST_CHECK 0

namespace ccpm
{
	/*
	 * Doubly linked list.
	 */
	struct list_item
	{
	private:
		list_item *_prev;
		list_item *_next;
		std::size_t count_next() const
		{
			std::size_t ct = 0;
			for ( auto i = this; i->_next != this; ++ct, i = i->_next )
			{
				assert(ct < 100000);
			}
			return ct;
		}
#if MCAS_CCPM_LIST_CHECK || MCAS_CCPM_IN_LIST_CHECK
		std::size_t count_prev() const
		{
			std::size_t ct = 0;
			for ( auto i = this; i->_prev != this && ct < 100000; ++ct, i = i->_prev )
			{
			}
			assert(ct < 100000);
			return ct;
		}
#endif
#if MCAS_CCPM_IN_LIST_CHECK
		/* inefficient function, for assert use only */
		bool is_in_list() const { return count() != 0; }
#endif
	public:
		list_item() : _prev(this), _next(this) {}
		list_item(const list_item &) = delete;
		list_item &operator=(const list_item &) = delete;

		/* forcibly remove from a list */
		void force_reset()
		{
			_prev = this;
			_next = this;
		}

		std::size_t count() const
		{
			auto cn = count_next();
#if MCAS_CCPM_LIST_CHECK
			auto cp = count_prev();
			assert(cn == cp);
#endif
			return cn;
		}

		/* insert i after this item */
		void insert_after(list_item *i)
		{
#if MCAS_CCPM_IN_LIST_CHECK
			assert( ! i->is_in_list() );
#endif
#if MCAS_CCPM_LIST_CHECK
			this->count();
#endif
			const auto n = this->_next;
			i->_next = n;
			i->_prev = this;
			this->_next = i;
			n->_prev = i;
#if MCAS_CCPM_IN_LIST_CHECK
			assert( i->is_in_list() );
			assert(this->count() == i->count());
#endif
		}

		/* true iff list contains i */
		bool contains(list_item *i) const
		{
#if MCAS_CCPM_LIST_CHECK
			this->count();
#endif
			auto e = this;
			while ( e != i && e->_next != this )
			{
				e = e->_next;
			}
			return e == i;
		}

		/* insert this item before i */
		void insert_before(list_item *i)
		{
#if MCAS_CCPM_IN_LIST_CHECK
			assert( ! is_in_list() );
#endif
#if MCAS_CCPM_LIST_CHECK
			i->count();
#endif
			const auto p = this->_prev;
			i->_prev = p;
			i->_next = this;
			this->_prev = i;
			p->_next = i;
#if MCAS_CCPM_IN_LIST_CHECK
			assert( i->is_in_list() );
			assert(this->count() == i->count());
#endif
		}

		void remove()
		{
#if MCAS_CCPM_IN_LIST_CHECK
			assert( is_in_list() );
#endif
			_prev->_next = _next;
			_next->_prev = _prev;
#if MCAS_CCPM_IN_LIST_CHECK
			assert(_next->count() == _prev->count());
#endif
			_prev = this;
			_next = this;
#if MCAS_CCPM_IN_LIST_CHECK
			assert( ! is_in_list() );
#endif
		}

		list_item *prev() { return _prev; }
		const list_item *prev() const { return _prev; }
		list_item *next() { return _next; }
		const list_item *next() const { return _next; }

		/* for list head: empty check */
		bool empty() const { return this == _next; }

		/* returns true if element e is in the list.
		 * "this" is assumed to be a list anchor;
		 * e is not tested for equality with this.
		 */
		bool contains(const list_item *e) const
		{
			for ( auto p = this ; p->next() != this ; p = p->next() )
			{
				if ( p->next() == e ) { return true; }
			}
			return false;
		}
	};
}

#endif
