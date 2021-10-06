#ifndef __MEMORY_PROVIDERS__
#define __MEMORY_PROVIDERS__

#include <common/exceptions.h>
#include <libpmem.h>
#include "mm_plugin_itf.h"
#include "pymm_config.h"

class Mmap_memory_provider;

/** 
 * Interface class for memory providers
 * 
 */
class Transient_memory_provider
{
public:
  virtual ~Transient_memory_provider() {}
  virtual void * malloc(size_t n) = 0;
  virtual void * calloc(size_t nelem, size_t elsize) = 0;
  virtual void * realloc(void * p, size_t n) = 0;
  virtual void   free(void * p) = 0;
};

inline bool ends_with(std::string const & value, std::string const & ending)
{
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

/** 
 * Provider for tiering between persistent memory and mmap
 * 
 */
class Pmem_memory_provider : public Transient_memory_provider
{
private:
  void *              _mapped_memory;
  size_t              _mapped_memory_size;
  MM_plugin_wrapper * _heap;
  addr_t _base;
  addr_t _limit;

public:

  Pmem_memory_provider(const std::string& pmem_file,
                       const unsigned long pmem_file_size_gb) {
    PLOG("using tiered memory provider (pmem_file=%s)(pmem_file_size_gb=%lu)",
         pmem_file.c_str(), pmem_file_size_gb);
    assert(!backing_directory.empty());
    assert(!pmem_file.empty());

    remove(pmem_file.c_str());

    /* set up pmem file and mapping */
    size_t mapped_lenp = 0;
    int is_pmem = 0;
    _mapped_memory_size = GiB(pmem_file_size_gb);
    _mapped_memory = pmem_map_file(pmem_file.c_str(),
                                   _mapped_memory_size,
                                   PMEM_FILE_CREATE,
                                   O_RDWR,
                                   &mapped_lenp,
                                   &is_pmem);

    if(_mapped_memory == nullptr)
      throw General_exception("unable to create transient mapped memory");
    
    if(mapped_lenp != _mapped_memory_size)
      throw General_exception("bad mapped size");

    assert(is_pmem);
    
    _base = reinterpret_cast<addr_t>(_mapped_memory);
    _limit = _base + _mapped_memory_size;

    /* set up pluggable heap allocator */
    _heap = new MM_plugin_wrapper(RCA_MM_PLUGIN_PATH);
    assert(_heap);
    
    if(_heap->init() != S_OK)
      throw Constructor_exception("heap init failed");

    if(_heap->add_managed_region(_mapped_memory, _mapped_memory_size) != S_OK)
      throw Constructor_exception("heap add region failed");
    
    PLOG("tiered memory provider checks OK (mapped len=%luGiB)", REDUCE_GiB(_mapped_memory_size));      
  }

  virtual ~Pmem_memory_provider() {
    delete _heap;
    pmem_unmap(_mapped_memory, _mapped_memory_size);
  }
  
  void * malloc(size_t n) {
    void * p = nullptr;
    if(_heap->allocate(n, &p) != S_OK)
      throw Out_of_memory("_heap->allocate failed");

    assert(p);
    PLOG("[PyMM]: using pmem allocation for transient memory");
    return p;
  }
  
  void * calloc(size_t nelem, size_t elsize) {
    void * p = nullptr;
    if(_heap->callocate(nelem * elsize, &p) != S_OK)
      throw General_exception("_heap->callocate failed");
    PLOG("transient pmem callocation (%p)", p);
    return p;
  }
  
  void * realloc(void * p, size_t n) {
    if(belongs(p)) {
      PLOG("[PyMM]: reallocate attempt (%p,%lu)", p, n);
      return nullptr;
    }

    return ::realloc(p,n);
  }

  inline bool belongs(void * addr)  {
    return (reinterpret_cast<uint64_t>(addr) >= _base) &&
      (reinterpret_cast<uint64_t>(addr) < _limit);
  }

  void free(void * p) {
    if(p==nullptr || belongs(p)==false) {
      ::free(p);
      return;
    }

    PLOG("[PyMM]: transient pmem free (%p)", p);
    void * q = p;
    if(_heap->deallocate_without_size(&q) != S_OK)
      throw General_exception("free failed in Pmem_memory_provider");
  }
};

/** 
 * Use mmap'ed file for each allocation
 * 
 * @param file_directory Directory for temporary files
 * 
 */
class Mmap_memory_provider : public Transient_memory_provider
{
private:
  static constexpr size_t BASE_ADDR = 0x0FAB00000000ULL;

  const std::string _dir;
  const uint64_t _base = BASE_ADDR;
  uint64_t       _addr = BASE_ADDR;
  
  std::map<void *, std::string> _filemap;
public:
  Mmap_memory_provider(const std::string& file_directory = "/tmp") : _dir(file_directory) {

    PLOG("mmap memory provider (%s)", file_directory.c_str());
    struct stat st;
    if((stat(file_directory.c_str(),&st) != 0) ||
       (st.st_mode & (S_IFDIR == 0))) {
      throw API_exception("invalid directory (%s)", file_directory.c_str());
    }

    { /* clean up prior .mem files */
      DIR * folder = opendir(file_directory.c_str());
      struct dirent *next_file;
      char filepath[256];

      while((next_file = readdir(folder)) != NULL )  {
        // build the path for each file in the folder
        snprintf(filepath, 255, "%s/%s", file_directory.c_str(), next_file->d_name);
        if(ends_with(std::string(filepath),".mem")) {
          remove(filepath);
        }
      }
      closedir(folder);
    }
  }

  bool belongs(void * addr)  {
    return (reinterpret_cast<uint64_t>(addr) >= _base) &&
      (reinterpret_cast<uint64_t>(addr) < _addr);
  }

  void * malloc(size_t n) {

    PLOG("[PyMM]: using transient mmap'ed file allocator for size (%lu)", n);
    const mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
    size_t rounded_n = round_up_page(n);
    std::stringstream ss;
    void * p;
    
    ss << _dir << "/mmap_transient_memory_" << _addr << ".mem";
    std::string filename = ss.str();
    int fdout;
    if((fdout = open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, mode)) >= 0) {
      /* create space in file */
      if(ftruncate(fdout, rounded_n) == 0) {
        p = mmap(reinterpret_cast<void*>(_addr), /* help debugging */
                 rounded_n,
                 PROT_READ | PROT_WRITE,
                 MAP_SHARED,
                 fdout, /* file */
                 0 /* offset */);
        _addr += rounded_n;

        close(fdout);
        _filemap[p] = filename;
        PLOG("[PyMM]: allocated %p", p);
        return p;
      }
    }
    throw Out_of_memory("transient_memory: malloc failed");
    return nullptr;
  }

  void * calloc(size_t nelem, size_t elsize) {
    size_t size = nelem * elsize;
    void * p = this->malloc(size);
    PLOG("[PyMM]: callocated %p", p);
    memset(p, 0, size);
    return p;
  }
  
  void * realloc(void * p, size_t n) {

    if(belongs(p)) {
      PLOG("[PyMM]: reallocate attempt (%p,%lu)", p, n);
      return nullptr;
    }

    return ::realloc(p,n);
  }
  
  void free(void * p) {

    if(p==nullptr || belongs(p)==false) {
      ::free(p);
      return;
    }

    if(_filemap.find(p) == _filemap.end())
      throw General_exception("trying to free bad address");

    const char * fname = _filemap[p].c_str();
    struct stat stat_buf;
    stat(fname, &stat_buf);
    assert(rc == 0);
    ::munmap(p, stat_buf.st_size);
    PLOG("[PyMM]: freeing (%p,%lu)", p, stat_buf.st_size);
    remove(fname);
    _filemap.erase(p);
  }
};


/** 
 * Provider for persistent memory
 * 
 */
class Tiered_memory_provider : public Transient_memory_provider
{
private:
  Pmem_memory_provider _prov_pmem;
  Mmap_memory_provider _prov_mmap;
  
public:
  Tiered_memory_provider(const std::string& file_directory,
                         const std::string& pmem_file,
                         const unsigned long pmem_file_size_gb)
    : _prov_pmem(pmem_file, pmem_file_size_gb),
      _prov_mmap(file_directory) {
  }
  
  void * malloc(size_t n) {
    void * p = nullptr;
    try {
      p = _prov_pmem.malloc(n);
    }
    catch(...) {
      p = _prov_mmap.malloc(n);
    }
    return p;
  }
  
  void * calloc(size_t nelem, size_t elsize) {
    void * p = nullptr;
    size_t n = nelem * elsize;
    try {
      p = _prov_pmem.malloc(n);
    }
    catch(...) {
      p = _prov_mmap.malloc(n);
    }
    return p;
  }
  
  void * realloc(void * p, size_t n) {
    void * q = nullptr;
    q = _prov_pmem.realloc(p, n);
    if(q) return q;
    q = _prov_mmap.realloc(p, n);
    if(q) return q;
    
    return ::realloc(p, n);
  }
  
  void free(void * p) {
    if(p==nullptr) {
      ::free(p);
      return;
    }
    if(_prov_mmap.belongs(p)) {
      _prov_mmap.free(p);      
    }
    else if(_prov_pmem.belongs(p)) {
      _prov_pmem.free(p);
    }
    else {
      ::free(p);
    }
  }
};


#endif // __MEMORY_PROVIDERS__

