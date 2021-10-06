
#include "../../mm_plugin_itf.h"
#include <ccpm/cca.h>
#include <common/logging.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
//#define DEBUG /* enable log output */

using Heap = ccpm::cca;
using IHeap = ccpm::IHeap_expandable;

PUBLIC status_t mm_plugin_init()
{
  return S_OK;
}

PUBLIC status_t mm_plugin_create(const char * params, void * root_ptr, mm_plugin_heap_t * out_heap)
{
	PLOG("%s (%s)", __func__, params);
	if (out_heap == nullptr) { return E_INVAL; }
	ccpm::cca::ctor_args *args = static_cast<ccpm::cca::ctor_args *>(root_ptr);
	if ( args->persister == nullptr ) { return E_INVAL; }
	auto heap =
		args->regions.size() == 0 && args->force_init
		? new Heap(static_cast<ccpm::persister *>(args->persister))
		: new Heap(
			args->persister
			, args->regions
			, args->callee_owns
		)
		;
	*out_heap = heap;
	return S_OK;
}

PUBLIC status_t mm_plugin_destroy(mm_plugin_heap_t heap)
{
	delete static_cast<Heap *>(heap);
	return S_OK;
}

PUBLIC status_t mm_plugin_add_managed_region(mm_plugin_heap_t heap,
                                      void * region_base,
                                      size_t region_size)
{
	PNOTICE("%s base=%p size=%zu",__func__, region_base, region_size);
	std::memset(region_base, 0, region_size);
	IHeap *h = static_cast<Heap *>(heap);
	
	ccpm::region_vector_t rv(common::make_byte_span(region_base, region_size));
	ccpm::region_span rs(rv);
	h->add_regions(rs);
	return S_OK;
}

PUBLIC status_t mm_plugin_query_managed_region(mm_plugin_heap_t heap,
                                        unsigned region_id,
                                        void** out_region_base,
                                        size_t* out_region_size)
{
  PLOG("%s",__func__);
  asm("int3");
  return E_NOT_IMPL;
}

PUBLIC status_t mm_plugin_register_callback_request_memory(mm_plugin_heap_t heap,
                                                    request_memory_callback_t callback,
                                                    void * param)
{
  PLOG("%s",__func__);
  asm("int3");
  return E_NOT_IMPL;
}

PUBLIC status_t mm_plugin_allocate(mm_plugin_heap_t heap, size_t n, void ** out_ptr)
{
  PLOG("%s (%lu)",__func__, n);
  *out_ptr = nullptr;
  return static_cast<Heap *>(heap)->allocate(*out_ptr, n, 1);
}

PUBLIC status_t mm_plugin_aligned_allocate(mm_plugin_heap_t heap, size_t n, size_t alignment, void ** out_ptr)
{
  PLOG("%s (%lu,%lu)",__func__, n, alignment);
  *out_ptr = nullptr;
  return static_cast<Heap *>(heap)->allocate(*out_ptr, n, alignment);
}

PUBLIC status_t mm_plugin_aligned_allocate_offset(mm_plugin_heap_t heap, size_t n, size_t alignment, size_t offset, void ** out_ptr)
{
  asm("int3");  
  *out_ptr = nullptr;
  return E_NOT_IMPL;
}

PUBLIC status_t mm_plugin_deallocate(mm_plugin_heap_t heap, void ** ptr, size_t)
{
	return mm_plugin_deallocate_without_size(heap, ptr);
}

PUBLIC status_t mm_plugin_deallocate_without_size(mm_plugin_heap_t heap, void ** ptr)
{
  PLOG("%s (%p)",__func__, ptr);
  if ( ptr )
  {
	static_cast<Heap*>(heap)->free(*ptr, 0);
  }
  return S_OK;
}

PUBLIC status_t mm_plugin_callocate(mm_plugin_heap_t heap, size_t n, void ** out_ptr)
{
	auto st = mm_plugin_allocate(heap, n, out_ptr);
	if ( st == S_OK )
	{
		::memset(*out_ptr, 0, n);
	}
	return st;
}

PUBLIC status_t mm_plugin_reallocate(mm_plugin_heap_t heap, void ** in_out_ptr, size_t n)
{
  return E_NOT_IMPL; /* we don't support reallocation */
}

PUBLIC status_t mm_plugin_usable_size(mm_plugin_heap_t heap, void * ptr, size_t * out_size)
{
  asm("int3");
  return E_NOT_IMPL;
}

PUBLIC status_t mm_plugin_inject_allocation(mm_plugin_heap_t heap, void * ptr, size_t size)
{
  return E_NOT_IMPL;
}

PUBLIC status_t mm_plugin_bytes_remaining(mm_plugin_heap_t heap, size_t *size)
{
  return static_cast<Heap*>(heap)->remaining(*size);
}

PUBLIC int mm_plugin_is_crash_consistent(mm_plugin_heap_t heap)
{
  return true;
}

PUBLIC int mm_plugin_can_inject_allocation(mm_plugin_heap_t heap)
{
  return false;
}

PUBLIC void mm_plugin_debug(mm_plugin_heap_t heap)
{
}
