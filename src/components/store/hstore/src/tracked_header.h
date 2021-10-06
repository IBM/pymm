/*
   Copyright [2020-2021] [IBM Corporation]
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

#ifndef MCAS_HSTORE_TRACKED_HEADER_H
#define MCAS_HSTORE_TRACKED_HEADER_H

#include <cstddef> /* size_t, ptrdiff_t */

struct injectee;

/* The layout of tracked storage:
 *
 * | ...................... | tracked_header | tracked data |........................|
 * |<--------- tracked_header._align ------->|<-(multiple of tracked_header._align) ->
 * |<-------------- tracked_header._size ------------------------------------------->|
 *
 * Note that tracked_header could perhaps be packed into a smaller space: _prev and
 * _next are at least 16-byte aligned, _align is a power if 2, and _size is a
 * multiple of _align.
 */

struct tracked_header
{
	tracked_header *_prev;
	tracked_header *_next;
	std::size_t _size;
	std::size_t _align;

	explicit tracked_header(unsigned debug_level, tracked_header *prev, tracked_header *next, std::size_t size, std::size_t align);
	void recover(unsigned debug_level, injectee *);
};

#endif
