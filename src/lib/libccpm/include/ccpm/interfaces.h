/*
   Copyright [2019-2021] [IBM Corporation]
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

#ifndef __CCPM_INTERFACES_H__
#define __CCPM_INTERFACES_H__

#include <common/byte_span.h>
#include <common/types.h>
#include <gsl/pointers>
#include <cstddef>
#include <functional>
#include <vector>

namespace ccpm
{
	struct persister
	{
		using byte_span = common::byte_span;
		virtual void persist(byte_span span) = 0;
	protected:
		~persister() {}
	};

/*
 * Ownership callback can resolve the ambiguity about area
 *        ownership which occurs during phase (b) of allocate() and free()
 *        Intended to be used by IHeap::reconstitute when a crash during
 *        an IHeap::allocate or IHeap::free call has left area ownership
 *        in doubt.
 *
 * In order to support ownership resolution, it is expected that a caller to
 * allocate or free will
 *   (1) store and persist the initial value of ptr
 *   (2) call allocate or free
 *
 * It is expected that IHeap::allocate() will
 *   (1) determine an appropriate new value for ptr
 *   (2) persist a note indicating that the allocation status of the new value
 *       of ptr is indeterminate.
 *   (3) store and persist the new value of ptr
 *   (4) persist an invalidation of the note written in step 2.
 *
 * It is expected that IHeap::free() will
 *   (1) persist a note indicating that the allocation status of the value
 *       of ptr is indeterminate.
 *   (2) store and persist nullptr as the new value of ptr
 *   (3) return the area located by the original value of ptr to free status
 *   (4) persist an invalidation of the note written in step 1.
 */

using ownership_callback_type = std::function<bool(const void * ptr)>;
using ownership_callback_t = ownership_callback_type;

inline bool accept_all(const void *) { return true; }
/*
 * Allocators can expand to more than one coarse-grained region of
 * memory.
 */
struct region_vector_t : public std::vector<common::byte_span>
{
	using base = std::vector<common::byte_span>;
  explicit region_vector_t(void * ptr_, std::size_t size_)
    : region_vector_t(common::make_byte_span(static_cast<common::byte *>(ptr_), size_))
  {}
  explicit region_vector_t(const value_type &v) {
    push_back(v);
  }
  region_vector_t() {
  }
	const base &cbase() const { return *this; }
};

using region_span = gsl::span<common::byte_span>;

enum class Type_id : int64_t
  {
    None        = 0,
    Fixed_array = 0xF0,
  };

/**
 * Heap allocator (variable sized allocations)
 */
class IHeap
{
public:
  virtual ~IHeap() {}

  /* Reconstitute/initialize slab from existing memory
   *
   * @param regions Pointer/length regions of contiguous memory
   * @param resolver Access to an object which can resolve the ambiguity
   *        over area ownership which occurs during a phase of allocate()
   *        and free() calls. If the callee of the ownership_callback_t
   *        function might own the area located by the arument, the callee
   *        must return true. If the callee does not own the area, it should
   *        return false.
   * @param force_init If true, force re-setting to empty
   *
   *
   * @return : True if memory was reset to empty
   **/
  virtual bool reconstitute(const region_span regions,
                            ownership_callback_t resolver = [] (const void *) -> bool { return true; },
                            const bool force_init = false) = 0;

  /* Allocate memory
   *
   * @param ptr in/out: nullptr -> pointer to newly allocated memory
   *                    On a successful allocation the allocator will
   *                    first write and then persist ptr. Allocation
   *                    has three phases:
   *                    (a) ptr is not yet written; the allocator unambiguously
   *                        owns the free "area" the address of which it will
   *                        later write into ptr.
   *                    (b) allocator has written ptr, but has not yet persisted
   *                        ptr. Caller must be able to tell the allocator (see
   *                        reconstitute) whether caller has accepted ownership
   *                        of the allocated area. Acceptance is implicit.
   *                        Visibility of a non-null value in ptr, which happens
   *                        upon write in the non-crash case and upon persist in
   *                        the crash+reconstitute case, consititutes acceptance.
   *                    (c) allocator has persisted the ptr. Caller will know upon
   *                        successful return (normal case) or discovery of a
   *                        non-null value (crash+reconstitute case) that it owns
   *                        the area.
   *
   *                    The pointer shall be altered iff the function returns S_OK.
   *
   * @param size Size of memory to allocate in bytes
   * @param alignment Alignment for memory
   *
   * @return : S_OK, E_INVAL, E_BAD_PARAM, E_EMPTY
   **/
  virtual status_t allocate(void * & ptr,
                            std::size_t bytes,
                            std::size_t alignment) = 0;

  /* Free previously allocated memory.
   * @param ptr in/out: Pointer to memory to free, which must have been previously
   *                    allocated with the same size and alignment.
   *                    On a successful free the allocator will first nullify ptr
   *                    and then persist ptr. Free has has three phases, similar
   *                    to allocation:
   *                    (a) ptr is not yet written; the caller unambiguously
   *                        owns the "area".
   *                    (b) allocator has nullified ptr, but has not yet persisted
   *                        ptr. Caller must be able to tell the allocator (see
   *                        reconstitute) whether caller has relinquished ownership
   *                        of the allocated area. Relinquishment is implicit.
   *                        Visibility of a null value in ptr, which happens upon
   *                        write in the non-crash case and upon persist in the
   *                        crash+reconstitute case, consititutes relinquishment.
   *                    (c) allocator has persisted the (null-valued) ptr. Caller
   *                        will know upon successful return (normal case) or
   *                        discovery of null value (crash+reconstitute case) that
   *                        the allocator now owns the area.
   *
   *                    The pointer shall be altered iff the function returns S_OK.
   *
   * @param size Size of memory to free in bytes
   *
   * @return : S_OK, E_INVAL
   **/
  virtual status_t free(void * & ptr,
                        std::size_t bytes = 0) = 0;

  /* Return remaining space in bytes
   * @param out_size Size remaining in bytes
   *
   * @return S_OK or E_NOT_IMPL
   **/
  virtual status_t remaining(std::size_t& out_size) const = 0;

  /* Return a vector of all regions
   *
   * @return vector containing the initial regions and all subsequently added
   * regions, if the implementation keeps track of regions. Otherwise, an
   * empty region_vector_t.
   **/
  virtual region_vector_t get_regions() const = 0;
};

/**
 * Heap allocator (variable sized allocations)
 */
class IHeap_expandable : public IHeap
{
public:
  /* Add an additional regions to the heap
   * @param regions Pointer/length regions of contiguous memory
   **/
  virtual void add_regions(const region_span regions) = 0;

  /** Test whether an address is in any heap region
   * @param addr the address to test
   *
   * @erturn true iff the address is in some heap region
   */
  virtual bool includes(const void *ptr) const = 0;
};

class ILog
{
public:
  virtual ~ILog() = default;

  /*
   * Record old value of a region to the log.
   */
  virtual void add(void *begin, std::size_t size) = 0;
  /*
   * Record an allocation to the log.
   */
  virtual void allocated(void *&p, std::size_t size) = 0;
  /*
   * Record a free to the log.
   */
  virtual void freed(void *&p, std::size_t size) = 0;

  /*
   * commit all previous add/allocated/freed commands
   */
  virtual void commit() = 0;
  /*
   * Restore all data areas added after initialization (or the most recent clear)
   * to the values present at their last add command.
   */
  virtual void rollback() = 0;

};
}
#endif //  __CCPM_INTERFACES_H__
