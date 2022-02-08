#ifndef __COMMON_RWLOCK_H__
#define __COMMON_RWLOCK_H__

#include <common/common.h>
#define _MULTI_THREADED
#include <common/exceptions.h>
#include <pthread.h>
#include <stdexcept>

/* enables lock checking and debugging output */
//#define DEBUG_LOCKING

namespace common
{
class RWLock {
 public:
  RWLock() : _lock{} {
    if (pthread_rwlock_init(&_lock, NULL))
      throw General_exception("unable to initialize RW lock");
  }

  int read_lock() {
#ifdef DEBUG_LOCKING    
    _read_lock_count++;
#endif
    return pthread_rwlock_rdlock(&_lock);
  }

  int read_trylock() {
#ifdef DEBUG_LOCKING
    int rc = pthread_rwlock_tryrdlock(&_lock);
    if(rc == 0) _read_lock_count++;
    return rc;
#else
    return pthread_rwlock_tryrdlock(&_lock);
#endif
  }

  int write_lock() {
#ifdef DEBUG_LOCKING
    _write_lock_count++;
#endif
    return pthread_rwlock_wrlock(&_lock);
  }

  int write_trylock() {
#ifdef DEBUG_LOCKING
    PLOG("common::RWLock::write-trylock (%p)", reinterpret_cast<void*>(this));
    int rc = pthread_rwlock_trywrlock(&_lock);
    if(rc == 0) _write_lock_count++;
    show();
    return rc;
#else
    return pthread_rwlock_trywrlock(&_lock);
#endif
  }

  int unlock() {
#ifdef DEBUG_LOCKING
    PLOG("common::RWLock::unlock (%p)", reinterpret_cast<void*>(this));
    if(_read_lock_count > 0)
      _read_lock_count--;
    else if(_write_lock_count == 1)
      _write_lock_count = 0;
    else {
      PWRN("bad lock state, nothing to unlock");
      asm("int3");
    }
    //    else throw Logic_exception("bad lock state");
    show();
#endif
    return pthread_rwlock_unlock(&_lock);
  }

#ifdef DEBUG_LOCKING
  void show() {
    PLOG("info RWLock (%p wr=%d rd=%d)",
         reinterpret_cast<void*>(this), _write_lock_count, _read_lock_count);
  }
#endif

  ~RWLock() { pthread_rwlock_destroy(&_lock); }

 private:
  pthread_rwlock_t _lock;
  int              _read_lock_count = 0;
  int              _write_lock_count = 0;
};

class RWLock_guard {
 public:
  enum {
    WRITE = 0,
    READ = 1,
  };

 public:
  RWLock_guard(RWLock &lock, int mode = READ) : _lock(lock) {
    if (mode == WRITE) {
      if (_lock.write_lock() != 0)
        throw std::range_error("failed to take write lock");
    }
    else if (mode == READ) {
      if (_lock.read_lock() != 0)
        throw std::range_error("failed to take read lock");
    }
    else throw Logic_exception("unexpected RWLock_guard mode");
  }

  ~RWLock_guard() noexcept(false) {
    if (_lock.unlock() != 0)
    {
      PLOG("%s: failed to release lock %p", __func__, common::p_fmt(this));
#ifdef DEBUG_LOCKING
      abort();
#endif
    }
  }

 private:
  RWLock &_lock;
};
}  // namespace common

#endif  // __COMMON_RWLOCK_H__
