/*
  eXokernel Development Kit (XDK)

  Samsung Research America Copyright (C) 2013

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
  Authors:
  Copyright (C) 2020, Daniel G. Waddington <daniel.waddington@ibm.com>
*/

#ifndef __COMMON_EXCEPTIONS_H__
#define __COMMON_EXCEPTIONS_H__

#include <assert.h>
#include <common/types.h>
#include <common/stack_trace.h>
#include <cstdarg>
#include <string>
#include <utility> /* forward */
#include "errors.h"
#include "logging.h"

#ifndef STRINGIFY
#define STRINGIFY(x) #x
#endif

#define TOSTRING(x) STRINGIFY(x)
#define ADD_LOC(X) X __FILE__ ":" TOSTRING(__LINE__)

//#define INTERRUPT_ON_EXCEPTION
//#define STACKTRACE_ON_EXCEPTION

class Exception {
 protected:
  Exception() {}

  Exception(const char *fmt, ...)
  {
    char cause[512];
    va_list args;
    va_start(args, fmt);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#if 7 <= __GNUC__
#pragma GCC diagnostic ignored "-Wformat-truncation"
#endif
    vsnprintf(cause, (sizeof cause)-2, fmt, args);
    va_end(args);
    snprintf(_cause, 512, "%s<< EXCEPTION - %s >>%s", ESC_ERR, cause, ESC_END);
#pragma GCC diagnostic pop

#ifdef STACKTRACE_ON_EXCEPTION
    print_stacktrace();
#endif
#ifdef INTERRUPT_ON_EXCEPTION
    asm("int3");
#endif
  }
 public:
  const char *cause() const { return _cause; }

  void set_cause(const char *cause) {
    __builtin_strncpy(_cause, cause, sizeof _cause);
  }

private:
  char _cause[512];
};

#include <common/logging.h>

/* A "coded" excpetion has a status_t error code */
class Coded_exception : public Exception {
protected:
  Coded_exception() : _err_code(E_FAIL) {}

  template <typename ... Args>
  Coded_exception(int err, const char *fmt, Args&& ... args)
    : Exception(fmt, std::forward<Args>(args)...)
    , _err_code(err)
  {
    PLOG("Coded_exception code %d cause %s", error_code(), cause());
  }

public:
  status_t error_code() const { return _err_code; }

private:
  status_t _err_code;
};

/*
 * Possibly the equivalent of a std::runtime_exception when that exception
 * occurs in a constructor. (Might replace with General_exception, or  the
 * proposed resource_exception subclass General_exception).
 */
class Constructor_exception : public Coded_exception {
public:
  Constructor_exception(int err)
    : Coded_exception(err, "Constructor failed") {}

  Constructor_exception()
    : Constructor_exception(E_FAIL) {}

  template <typename ... Args>
  __attribute__((__format__(__printf__, 2, 0)))
  Constructor_exception(const char *fmt, Args&& ... args)
    : Coded_exception(E_FAIL, fmt, std::forward<Args>(args)...) {
  }
};

/*
 * Possibly the equivalent of a std::runtime_exception: typically a resource error.
 * But due to the name, likely used for other things.
 * Might be good to have a subclass of General_excepton for resource exceptions.
 */
class General_exception : public Coded_exception {
public:
  General_exception(int err) : Coded_exception(err, "General exception") {}

  General_exception() : General_exception(E_FAIL) { }

  template <typename ... Args>
  __attribute__((__format__(__printf__, 2, 0)))
  General_exception(const char *fmt, Args&& ... args)
    : Coded_exception(E_FAIL, fmt, std::forward<Args>(args)...)
  {
  }
};

/*
 * (Presumed) failure of a caller to adhere to limitations described in an API
 */
class API_exception : public Coded_exception {
public:
  API_exception(int err) : Coded_exception(err, "API error") {}

  API_exception() : API_exception(E_FAIL) {}

  template <typename ... Args>
  __attribute__((__format__(__printf__, 2, 0)))
  API_exception(const char *fmt, Args&& ... args)
    : Coded_exception(E_FAIL, fmt, std::forward<Args>(args)...) {}
};

/*
 * std::logic_exeception means a program error which should have been detectable before execution.
 * Indicates a coding error, and arguably cannot be handled, as it indicates that the program
 * has lost its integrity cannot be handled, as it indicates that the program
 * has lost its integrity.
 *
 * This use of Logic_exception is much broader. It has become a catch-all exception, and so general
 * as to convey little meaning. Seven meanings are desribed below:
 *
 * 1. Failure of a request when no other information is avaiable (e.g. nullptr to request for a resource)
 * Might be due to a lack of resource, or migth be a permanent error. We cannot tell because not enough information was provided.
 *
 * Typical patterns
 *   if ( !ptr ) throw ...
 *  if ( mmap() != page_start ) throw
 *   throw std::range_error("failed to take read lock");
 *   throw Logic_exception("bad clock_gettime call");
 *   throw Logic_exception("nupm::expose_memory failed unexpectedly");
 *   throw Logic_exception("xpmem_make failed unexpectedly");
 *   throw Logic_exception("get_attributes failed on storage engine");
 *   throw Logic_exception("pool info JSON creation failed");
 *   throw Logic_exception("unable to create Zyre factory");
 *   throw Logic_exception("unable to create Zyre component instance");
 *   throw Logic_exception("%s: network posting failed unexpectedly.", __func__);
 *   throw Logic_exception("ADO_proxy execv failed (%s)", c.data());
 *   throw Logic_exception("%s lock failed", __func__);
 *   throw Logic_exception("%s unlock failed", __func__);
 *   throw Logic_exception("%s swap_keys failed", __func__);
 *   throw Logic_exception("%s erase failed", __func__);
 *
 * 2. std::logic_error: Errors caused within a component and which could have been avoided at compile time (by error-free coding)
 *   a. Error in dynamic_cast, or in cnverting an integer to an enumeration (std::domain_error)
 *      Such errors between components on the same call stack have a separate type: API_excepions.
 *      Such errors appearing in message data also have a separate type: Protocol_exception.
 *
 *     Typical patterns
 *       switch ... default: throw (ir if-else equivalent)
 *       throw Logic_exception("unexpected condition");
 *       throw Logic_exception("SIGSEGV handler received non-SEGV signal");
 *       throw Logic_exception("bad downsize type");
 *       throw Logic_exception("unexpected RWLock_guard mode");
 *       throw Logic_exception("unexpected task condition");
 *
 *   b. other things which could have been avoided at compile time. These include "can't happen" conditions for which
 *      the codeer has nevertheless chosen to check.
 *       throw Logic_exception(" ... conflict detected."
 *       throw Logic_exception(" ... is nullptr") in a request for an existing resource, not a new resource
 *       throw Logic_exception("unexpected condition");
 *       throw Logic_exception("unable to unlock after lock");
 *       throw Logic_exception("Shard_ado: unlock for KV after ADO work completion failed");
 *       throw Logic_exception("deferred unlock failed");
 *       throw Logic_exception("unable to delete pool after POOL DELETE op event");
 *       throw Logic_exception("failed to close pool");
 *       throw Logic_exception("unexpected pool reference count");
 *       throw Logic_exception("release_life_locks: pool unlock failed (%d)", rc);
 *
 * 3. Meaning: Length error (std::length_error)
 *   if ( f(x) > buffer_size )
 *   throw Logic_exception("%s::%s - insufficient buffer for Message_handshake_reply", +description, __func__);
 *
 * 4. Protocol_exception: Several uses of Logic_exceptions should probably use Protocol_exception:
 * Unsupported request, unexpected response, conversion erros of values contained in messages.
 *
 * Typical patterns
 *   throw Logic_exception(" ... got something else")
 *   throw Logic_exception("invalid ... message")
 *   throw Logic_exception("ADO_process: unknown mcas::ipc message type");
 *   throw Logic_exception("unknown table op code");
 *   throw Logic_exception("unknown op event (%d)", op);
 *   throw Logic_exception("Shard_ado: bad op request from ADO plugin");
 *   throw Logic_exception("unhandled message (type:%x)", int(msg->type_id()));
 *   throw Logic_exception("unknown action type");
 *
 * 5. std::runtime_error: Errors in which neither the requestor nor the provider is necessariily at fault. Both are victims fo bad circumstance: something full, or something missing
 *
 * Typical patterns
 *   throw Logic_exception("lookup_dax_device could not find path for region (%d)",
 *   throw Logic_exception("reopened root incorrect size");
 *   throw Logic_exception("no MCAS kernel module");
 *   throw Logic_exception("put_ado_invoke: put failed");
 *   throw Logic_exception("cluster signal queue full");
 *   throw Logic_exception("Ado_map: pool (%s) already assigned proxy (%p)", pool_name.c_str(), proxy);

 * 6. API_Exception (either std::invalid_argument or std::domain_error, depending on whether the failure is a flag or an arithmetic value)
 *
 * Typical patterns
 *   throw Logic_exception("bad iterator ctor param");
 *   throw Logic_exception("add reference to pool that is not open");
 *   throw Logic_exception("%s: bad target; value never locked? (%p)", __func__, target);
 *   throw Logic_exception("bad iterator ctor param"), bool()),
 *   throw Logic_exception("bad mask parameter");
 */
class Logic_exception : public Coded_exception {
public:
  Logic_exception(int err) : Coded_exception(err, "Logic error") {}

  Logic_exception() : Logic_exception(E_FAIL) {}

  template <typename ... Args>
  __attribute__((__format__(__printf__, 2, 0)))
  Logic_exception(const char *fmt, Args&& ... args)
    : Coded_exception(E_FAIL, fmt, std::forward<Args>(args)...)
  {
  }
};

/*
 * Two uses:
 *   Out of buffers (a runtime exception)
 *   Timeout (a runtime exception)
 */
class Program_exception : public Coded_exception {
public:
  Program_exception(int err) : Coded_exception(err, "Program error") {}

  Program_exception() : Program_exception(E_FAIL) {}

  template <typename ... Args>
  __attribute__((__format__(__printf__, 2, 0)))
  Program_exception(const char *fmt, Args&& ... args)
    : Coded_exception(E_FAIL, fmt, std::forward<Args>(args)...) {
  }
};

/*
 * Used to indicate memory exhaustion or inability to allocate requested memory
 */
class Out_of_memory : public Exception {
public:
  Out_of_memory() {}

  template <typename ... Args>
  Out_of_memory(const char *fmt, Args&& ... args)
    : Exception(fmt, std::forward<Args>(args)...)
  {
  }
};


/*
 * (Presumed) failure of a client or server to adhere to a protocoli: 6 uses, all in MCAS server
 */
class Protocol_exception : public Coded_exception {
public:
  Protocol_exception(int err) : Coded_exception(err, "Protocol error") {}

  Protocol_exception() : Protocol_exception(E_FAIL) {}

  template <typename ... Args>
  __attribute__((__format__(__printf__, 2, 0)))
  Protocol_exception(const char *fmt, Args&& ... args)
    : Coded_exception(E_FAIL, fmt, std::forward<Args>(args)...) {}
};

#endif
