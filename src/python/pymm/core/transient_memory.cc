#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL pymmcore_ARRAY_API

#include <sys/mman.h>
#include <dirent.h>
#include <sstream>
#include <map>

#include <common/logging.h>
#include <common/utils.h>
#include <Python.h>

static constexpr size_t LARGE_ALLOCATION_THRESHOLD = KiB(64);

#include "memory_providers.h"

static Transient_memory_provider * g_provider = nullptr;
static PyMemAllocatorEx            g_default_allocators;


/** 
 * Static clean up
 * 
 */
static void __attribute__((destructor)) __cleanup__() {
  if(g_provider)
    delete g_provider;
}

extern "C" void* Intercept_Malloc(void * ctx, size_t n)
{
  if(globals::debug_level > 2)
    PLOG("transient_memory: intercept malloc(%p,%ld)", ctx, n);
  
  if(n < LARGE_ALLOCATION_THRESHOLD)
    return malloc(n);

  void * p = g_provider->malloc(n); /* n > 0 --> p != null */
  if(p == nullptr) {
    perror("");
    throw General_exception("malloc() call in Intercept_Malloc failed (errno=%d)", errno);
  }
  return p;
}

extern "C" void* Intercept_Calloc(void * ctx, size_t nelem, size_t elsize)
{
  if(globals::debug_level > 2)
    PLOG("transient_memory: intercept calloc(%p,%ld,%ld)", ctx, nelem, elsize);
  
  if(nelem*elsize < LARGE_ALLOCATION_THRESHOLD)
    return calloc(nelem,elsize); 
    
  void * p = g_provider->calloc(nelem,elsize);
  if(p == nullptr) {
    perror("");
    throw General_exception("malloc() call in Intercept_Calloc failed (errno=%d)", errno);
  }
  return p;
}

extern "C" void* Intercept_Realloc(void * ctx, void * p, size_t n)
{
  if(globals::debug_level > 2)
    PLOG("transient_memory: intercept remalloc(%p,%p,%ld)", ctx, p, n);
  
  if(n < LARGE_ALLOCATION_THRESHOLD)
    return realloc(p, n);
  
  void * np = g_provider->realloc(p,n);
  return np;
}

extern "C" void Intercept_Free(void * ctx, void * p)
{
  if(globals::debug_level > 2)
    PLOG("transient_memory: intercept free(%p,%p)", ctx, p);
  
  g_provider->free(p);
}

PyObject * pymmcore_disable_transient_memory(PyObject * self,
                                             PyObject * args,
                                             PyObject * kwds)
{
  if(PY_MINOR_VERSION == 9) {
    PWRN("Python3.9 does not yet support disabling transient memory");
  }
  else {
    PyMem_SetAllocator(PYMEM_DOMAIN_RAW, &g_default_allocators);
    delete g_provider;
    g_provider = nullptr;
  }
  
  Py_RETURN_NONE;
}


PyObject * pymmcore_enable_transient_memory(PyObject * self,
                                            PyObject * args,
                                            PyObject * kwds)
{
  static const char *kwlist[] = {"backing_directory",
                                 "pmem_file",
                                 "pmem_file_size_gb",
                                 NULL};

  char * p_backing_directory = nullptr;
  char * p_pmem_file = nullptr;
  unsigned long p_pmem_file_size_gb = 0;
  
  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "s|sk",
                                    const_cast<char**>(kwlist),
                                    &p_backing_directory,
                                    &p_pmem_file,
                                    &p_pmem_file_size_gb)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  assert(p_backing_directory);

  try {
    if(p_pmem_file == nullptr) {
      /* single mmap'ed file tier */
      g_provider = new Mmap_memory_provider(p_backing_directory);
    }
    else {
      //    g_provider = new Pmem_memory_provider(p_pmem_file, p_pmem_file_size_gb);
      g_provider = new Tiered_memory_provider(p_backing_directory, p_pmem_file, p_pmem_file_size_gb);
    }
  }
  catch(...) {
    PWRN("transient memory unable to initialize (resource creation failed)");
    Py_RETURN_NONE;
  }
      
  
  /* modify allocator */
  PyMemAllocatorEx allocator;
  PyMem_GetAllocator(PYMEM_DOMAIN_RAW, &g_default_allocators);
  PyMem_GetAllocator(PYMEM_DOMAIN_RAW, &allocator);

  allocator.malloc = &Intercept_Malloc;
  allocator.realloc = &Intercept_Realloc;
  allocator.calloc = &Intercept_Calloc;
  allocator.free = &Intercept_Free;
  
  PyMem_SetAllocator(PYMEM_DOMAIN_RAW, &allocator);
  
  Py_RETURN_NONE;
}

