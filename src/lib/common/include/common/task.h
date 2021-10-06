/*
   IBM Corporation Copyright (C) 2017

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.

   As a special exception, if you link the code in this file with
   files compiled with a GNU compiler to produce an executable,
   that does not cause the resulting executable to be covered by
   the GNU Lesser General Public License.  This exception does not
   however invalidate any other reasons why the executable file
   might be covered by the GNU Lesser General Public License.
   This exception applies to code released by its copyright holders
   in files containing the exception.
*/

/*
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef __COMMON_TASK_H__
#define __COMMON_TASK_H__

#include <common/common.h>
#include <common/cpu.h>
#include <common/errors.h>
#include <common/exceptions.h>
#include <common/spinlocks.h>
#include <common/utils.h>
#include <numa.h>
#include <sched.h>
#include <unistd.h>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wunused-parameter"

namespace common
{
/**
 * Basic unit of work
 *
 */
class Tasklet {
 public:
  virtual ~Tasklet() {}
  virtual void initialize(unsigned core) = 0; /*< called once */
  virtual bool do_work(unsigned core) = 0; /*< called in tight loop; return false to exit */
  virtual void cleanup(unsigned core) {}; /*< called once */
  virtual bool ready() { return _ready; }
  void mark_ready() { _ready = true; }
private:
  bool _ready = false;
};

/**
 * Manages per-core threads executing tasklets.
 *
 */
template <typename __Tasklet_t, typename __Arg_t>
class Per_core_tasking {
  static constexpr unsigned MAX_CORES = 256;
  static constexpr bool option_DEBUG = false;

 public:
  
  Per_core_tasking(cpu_mask_t& cpus, __Arg_t arg, bool pin=true, unsigned int us_stagger=0) : _pin(pin) {
    
    for (unsigned c = 0; c < sysconf(_SC_NPROCESSORS_ONLN); c++) {

      if(us_stagger >  0)
        usleep(us_stagger);
      
      if (cpus.check_core(c)) {
        _tasklet[c] = new __Tasklet_t(arg);
        _threads[c] = new std::thread(&Per_core_tasking::thread_entry, this, c);
      }
      else {
        _tasklet[c] = nullptr;
        _threads[c] = nullptr;
      }
    }

    /* wait for tasklet to signal its readiness */
    for (unsigned c = 0; c < sysconf(_SC_NPROCESSORS_ONLN); c++) {
      if (cpus.check_core(c)) {
        while (!_tasklet[c]->ready()) usleep(100000);
      }
    }

    _start_flag = true;
  }

  virtual ~Per_core_tasking() {
    _exit_flag = true;
    wait_for_all();
    if (option_DEBUG) PLOG("Per_core_tasking: threads joined");
  }

  void wait_for_all() {
    unsigned remaining;
    do {
      remaining = 0;
      
      for (unsigned c = 0; c < sysconf(_SC_NPROCESSORS_ONLN); c++) {
        if (_threads[c]) {
          void * rv;
          if(pthread_tryjoin_np(_threads[c]->native_handle(),&rv) == 0) {
            _threads[c] = nullptr;
            delete _tasklet[c];
          }
          else remaining ++;
        }
      }
      if(remaining > 0)
        usleep(100000);
      
    } while(remaining > 0);
  }

  __Tasklet_t* tasklet(unsigned core) {
    if (core >= MAX_CORES) throw General_exception("out of bounds");
    return _tasklet[core];
  }

 private:
  void thread_entry(unsigned core) {
    if(_pin) {
      cpu_mask_t mask;
      mask.add_core(core);
      set_cpu_affinity_mask(mask);
    }

    try {
      _tasklet[core]->initialize(core);
      _tasklet[core]->mark_ready();
      
      while (!_start_flag) usleep(1000);

      while (!_exit_flag) {
        if (!(_tasklet[core]->do_work(core)))
          break; /* if do_work return false, we exit the thread */
      }

    } catch ( const Exception &e ) {
      PERR("thread work threw exception: %s", e.cause());
    } catch ( const std::exception &e ) {
      PERR("thread work threw exception %s", e.what());
    } catch (...) {
      PERR("thread work threw exception");
    }

    try {
      _tasklet[core]->cleanup(core);
    } catch ( const Exception &e ) {
      PERR("thread cleanup threw exception: %s", e.cause());
    } catch ( const std::exception &e ) {
      PERR("thread cleanup threw exception %s", e.what());
    } catch (...) {
      PERR("thread cleanup threw exception");
    }
  }

 private:
  bool _start_flag = false;
  bool _exit_flag  = false;
  const bool _pin  = true;
  
  std::thread*      _threads[MAX_CORES];
  __Tasklet_t*      _tasklet[MAX_CORES];
};

}  // namespace common

#pragma GCC diagnostic pop
#endif
