/*
   Copyright [2017-2020] [IBM Corporation]
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

#ifndef MCAS_CCPM_CONTAINER_CC_H
#define MCAS_CCPM_CONTAINER_CC_H

#include <ccpm/allocator_tl.h>
#include <ccpm/cca.h>
#include <ccpm/log.h>

#include <cstddef> // size_t

namespace ccpm
{
	/* Stuff needed for a crash-consistent container */
	template <typename Container>
		struct container_cc
#if 0
			: public cca /* proxy for a memory resource (in C++17 parlance), for more memory */
#endif
			: public log /* a log, which uses the memory resource, for rollback */
		{
		private:
			cca *_cca; /* not owned */
			/* an allocator + tracker, to translate memory resource callbacks to the log */
			allocator_tl _allocator;
			/* A container, which uses the memory resource and the log */
		public:
			/* The container could be included here directly.
			 * But to simplify the log "includes" filter, which determines which
			 * changes to record and roll back, the container space is instead allocated
			 * in the same memory region as its elements.
			 */
			Container *container;
		private:
			const cca &mr() const { return static_cast<const cca &>(*this); }
			log &lr() { return static_cast<log &>(*this); }
		public:
			container_cc(log::persister_type persister_, cca& cca_ /* void *ptr, std::size_t size */)
				/* Note: region_vector_t has a std::vector as a public base class.
				 * Best to avoid it.
				 */
				: log(persister_, &cca_)
				, _cca(&cca_)
				, _allocator(&mr(), &lr())
				, container(new (_allocator.allocate(sizeof *container)) Container(_allocator))
			{
			}
			container_cc(const container_cc &) = delete;
			container_cc &operator=(const container_cc &) = delete;
			/*
			 * The explanation for the log (base class) move constructor
			 * applies here too.
			 */
			container_cc(container_cc &&other) = default;
			container_cc(container_cc &&other, log::persister_type persister_, cca *cca)
				: log(std::move(static_cast<log &&>(other)), persister_, cca)
				, _cca(cca)
				, _allocator(std::move(other._allocator))
				, container(std::move(other.container))
			{
				PLOG("CCA at %p", static_cast<void *>(_cca));
			}

			cca& mr() { return *_cca; }
			void set_cca(ccpm::cca& cca_) { _cca = &cca_; }
		};
}

#endif
