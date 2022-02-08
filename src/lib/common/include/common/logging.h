/*
   eXokernel Development Kit (XDK)

   Samsung Research America Copyright (C) 2013
   IBM Corporation 2019-2021

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
  Author(s):
  Copyright (C) 2016,2019, Daniel G. Waddington <daniel.waddington@ibm.com>
  Copyright (C) 2014, Daniel G. Waddington <daniel.waddington@acm.org>
*/

#ifndef __COMMON_LOGGING_H__
#define __COMMON_LOGGING_H__

#if defined(__cplusplus)
#include <cstdio>
#include <cstdarg>
#include <sstream> /* ostringstream */
#include <thread> /* this_thread */
#include <utility> /* forward */

/* optional: print a timestamp */
#ifndef LOG_PRINT_TIMESTAMP
#define LOG_PRINT_TIMESTAMP 0
// #define FLOGM(fmt, ...) (FLOG)("<{} {:x}> {}::{} " fmt, DRD_GET_VALGRIND_THREADID, std::this_thread::get_id(), type_of(*this), __func__, __VA_ARGS__)
#endif

#if LOG_PRINT_TIMESTAMP && defined CONFIG_DEBUG
#if __has_include(<valgrind/drd.h>)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <valgrind/drd.h>
#pragma GCC diagnostic pop
#endif
#include <chrono>

namespace
{
  double log_ts()
  {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }
}

#define LOG_TS_FMT " %f"
#define LOG_TS_ARG log_ts(),
#else /* ! ( LOG_PRINT_TIMESTAMP && defined CONFIG_DEBUG ) */
#define LOG_TS_FMT ""
#define LOG_TS_ARG
#endif

namespace common
{
  struct log_source
  {
  private:
	unsigned _debug_level;
  public:
    explicit log_source(const unsigned debug_level_)
      : _debug_level(debug_level_)
    {}
    unsigned debug_level() const { return _debug_level; }
    log_source(const log_source &other) = default;
  };
  /* g++ formast checking wants an argument formatted with %p to be a void *,
   * or at least not a char *. Function to turn any pointer into an type
   * acceptable to %p/
   */
  template <typename T> const void *p_fmt(const T t) { return static_cast<const void *>(t); }
}

#else /* ! defined __cplusplus */
#include <stdarg.h> /* va_list, va_start, va_end */
#include <stdio.h> /* vsnprint, fprintf */
#endif

#define NORMAL_RED "\033[31m"
#define NORMAL_GREEN "\033[32m"
#define NORMAL_YELLOW "\033[33m"
#define NORMAL_BLUE "\033[34m"
#define NORMAL_MAGENTA "\033[35m"
#define NORMAL_CYAN "\033[36m"

#define BRIGHT "\033[1m"
#define NORMAL_XDK "\033[0m"
#define RESET "\033[0m"

#define BRIGHT_CYAN BRIGHT NORMAL_CYAN
#define BRIGHT_MAGENTA BRIGHT NORMAL_MAGENTA
#define BRIGHT_BLUE BRIGHT NORMAL_BLUE
#define BRIGHT_YELLOW BRIGHT NORMAL_YELLOW
#define BRIGHT_GREEN BRIGHT NORMAL_GREEN
#define BRIGHT_RED BRIGHT NORMAL_RED

#define WHITE_ON_RED "\033[41m"
#define WHITE_ON_GREEN "\033[42m"
#define WHITE_ON_YELLOW "\033[43m"
#define WHITE_ON_BLUE "\033[44m"
#define WHITE_ON_MAGENTA "\033[45m"

#define ESC_LOG NORMAL_GREEN
#define ESC_DBG NORMAL_YELLOW
#define ESC_INF NORMAL_CYAN
#define ESC_WRN NORMAL_RED
#define ESC_ERR BRIGHT_RED
#define ESC_END "\033[0m"

#define PR_MAX_BUFFER 2048

inline void pr_inner(const char *format, const char *intro, va_list args)
{
#ifdef CONFIG_DEBUG
  char buffer[PR_MAX_BUFFER];
  vsnprintf(buffer, PR_MAX_BUFFER, format, args);
  fprintf(stderr, "%s:" LOG_TS_FMT " %s %s\n", intro, LOG_TS_ARG buffer, ESC_END);
#else
  (void)format;
  (void)intro;
  (void)args;
#endif
}

/* Somehow, pr_info and PLOG ended up identical */
void pr_info(const char * format, ...) __attribute__((format(printf, 1, 2)));
inline void pr_info(const char * format, ...)
{
  va_list args; va_start(args, format);
  pr_inner(format, ESC_LOG "[LOG]", args);
  va_end(args);
}

/* Somehow, pr_error and PERR ended up identical */
void pr_error(const char * format, ...) __attribute__((format(printf, 1, 2)));
inline void pr_error(const char * format, ...)
{
  va_list args; va_start(args, format);
  pr_inner(format, ESC_ERR "[LOG]", args);
  va_end(args);
}

void PLOG(const char * format, ...) __attribute__((format(printf, 1, 2)));
inline void PLOG(const char * format, ...)
{
  va_list args; va_start(args, format);
  pr_inner(format, ESC_LOG "[LOG]", args);
  va_end(args);
}

void PDBG(const char * format, ...) __attribute__((format(printf, 1, 2)));
inline void PDBG(const char * format, ...)
{
  va_list args; va_start(args, format);
  pr_inner(format, ESC_DBG "[DBG]", args);
  va_end(args);
}

void PINF(const char * format, ...) __attribute__((format(printf, 1, 2)));
inline void PINF(const char * format, ...)
{
  va_list args; va_start(args, format);
#ifdef CONFIG_DEBUG
  char buffer[PR_MAX_BUFFER];
  vsnprintf(buffer, PR_MAX_BUFFER, format, args);
  fprintf(stderr, "%s %s %s\n", ESC_INF, buffer, ESC_END);
#else
  (void)format;
  (void)args;
#endif
  va_end(args);
}

void PWRN(const char * format, ...) __attribute__((format(printf, 1, 2)));
inline void PWRN(const char * format, ...)
{
  va_list args; va_start(args, format);
  pr_inner(format, ESC_WRN "[WRN]", args);
  va_end(args);
}

void PERR(const char * format, ...) __attribute__((format(printf, 1, 2)));
inline void PERR(const char * format, ...)
{
  va_list args; va_start(args, format);
  pr_inner(format, ESC_ERR "Error", args);
  va_end(args);
}

void PEXCEP(const char * format, ...) __attribute__((format(printf, 1, 2)));
inline void PEXCEP(const char * format, ...)
{
  va_list args; va_start(args, format);
  pr_inner(format, ESC_ERR "Exception", args);
  va_end(args);
}

void PNOTICE(const char * format, ...) __attribute__((format(printf, 1, 2)));
inline void PNOTICE(const char * format, ...)
{
  va_list args; va_start(args, format);
  pr_inner(format, BRIGHT_RED "NOTICE", args);
  va_end(args);
}

void PMAJOR(const char * format, ...) __attribute__((format(printf, 1, 2)));
inline void PMAJOR(const char * format, ...)
{
  va_list args; va_start(args, format);
  pr_inner(format, NORMAL_BLUE "[+]", args);
  va_end(args);
}

/* one-line conditional PLOG */

// clang-format off

#define GET_MACRO(_1,_2,_3,_4,_5,_6,_7,_8,_9,_A,_B,_C,_D,_E,NAME,...) NAME

#define CPLOG_E(level,format,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,pa,pb) ( ( (level) < this->debug_level() ) && (PLOG(format,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,pa,pb), true) )
#define CPLOG_D(level,format,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,pa) ( ( (level) < this->debug_level() ) && (PLOG(format,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,pa), true) )
#define CPLOG_C(level,format,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9) ( ( (level) < this->debug_level() ) && (PLOG(format,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9), true) )
#define CPLOG_B(level,format,p0,p1,p2,p3,p4,p5,p6,p7,p8) ( ( (level) < this->debug_level() ) && (PLOG(format,p0,p1,p2,p3,p4,p5,p6,p7,p8), true) )
#define CPLOG_A(level,format,p0,p1,p2,p3,p4,p5,p6,p7) ( ( (level) < this->debug_level() ) && (PLOG(format,p0,p1,p2,p3,p4,p5,p6,p7), true) )
#define CPLOG_9(level,format,p0,p1,p2,p3,p4,p5,p6) ( ( (level) < this->debug_level() ) && (PLOG(format,p0,p1,p2,p3,p4,p5,p6), true) )
#define CPLOG_8(level,format,p0,p1,p2,p3,p4,p5) ( ( (level) < this->debug_level() ) && (PLOG(format,p0,p1,p2,p3,p4,p5), true) )
#define CPLOG_7(level,format,p0,p1,p2,p3,p4) ( ( (level) < this->debug_level() ) && (PLOG(format,p0,p1,p2,p3,p4), true) )
#define CPLOG_6(level,format,p0,p1,p2,p3) ( ( (level) < this->debug_level() ) && (PLOG(format,p0,p1,p2,p3), true) )
#define CPLOG_5(level,format,p0,p1,p2) ( ( (level) < this->debug_level() ) && (PLOG(format,p0,p1,p2), true) )
#define CPLOG_4(level,format,p0,p1) ( ( (level) < this->debug_level() ) && (PLOG(format,p0,p1), true) )
#define CPLOG_3(level,format,p0) ( ( (level) < this->debug_level() ) && (PLOG(format,p0), true) )
#define CPLOG_2(level,msg) ( ( (level) < this->debug_level() ) && (PLOG(msg), true) )
#define CPLOG_1(level) ( (level) < this->debug_level() )
#define CPLOG(...) GET_MACRO(__VA_ARGS__,CPLOG_E,CPLOG_D,CPLOG_C,CPLOG_B,CPLOG_A,CPLOG_9,CPLOG_8,CPLOG_7,CPLOG_6,CPLOG_5,CPLOG_4,CPLOG_3,CPLOG_2,CPLOG_1,CPLOG_0)(__VA_ARGS__)

#define CPINF_E(level,format,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,pa,pb) if ( (level) < this->debug_level() ) { PINF(format,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,pa,pb); }
#define CPINF_D(level,format,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,pa) if ( (level) < this->debug_level() ) { PINF(format,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,pa); }
#define CPINF_C(level,format,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9) if ( (level) < this->debug_level() ) { PINF(format,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9); }
#define CPINF_B(level,format,p0,p1,p2,p3,p4,p5,p6,p7,p8) if ( (level) < this->debug_level() ) { PINF(format,p0,p1,p2,p3,p4,p5,p6,p7,p8); }
#define CPINF_A(level,format,p0,p1,p2,p3,p4,p5,p6,p7) if ( (level) < this->debug_level() ) { PINF(format,p0,p1,p2,p3,p4,p5,p6,p7); }
#define CPINF_9(level,format,p0,p1,p2,p3,p4,p5,p6) if ( (level) < this->debug_level() ) { PINF(format,p0,p1,p2,p3,p4,p5,p6); }
#define CPINF_8(level,format,p0,p1,p2,p3,p4,p5) if ( (level) < this->debug_level() ) { PINF(format,p0,p1,p2,p3,p4,p5); }
#define CPINF_7(level,format,p0,p1,p2,p3,p4) if ( (level) < this->debug_level() ) { PINF(format,p0,p1,p2,p3,p4); }
#define CPINF_6(level,format,p0,p1,p2,p3) if ( (level) < this->debug_level() ) { PINF(format,p0,p1,p2,p3); }
#define CPINF_5(level,format,p0,p1,p2) if ( (level) < this->debug_level() ) { PINF(format,p0,p1,p2); }
#define CPINF_4(level,format,p0,p1) if ( (level) < this->debug_level() ) { PINF(format,p0,p1); }
#define CPINF_3(level,format,p0) if ( (level) < this->debug_level() ) { PINF(format,p0); }
#define CPINF_2(level,msg) if ( (level) < this->debug_level() ) { PINF(msg); }
#define CPINF_1(level)
#define CPINF(...) GET_MACRO(__VA_ARGS__,CPINF_E,CPINF_D,CPINF_C,CPINF_B,CPINF_A,CPINF_9,CPINF_8,CPINF_7,CPINF_6,CPINF_5,CPINF_4,CPINF_3,CPINF_2,CPINF_1,CPINF_0)(__VA_ARGS__)

#define CPWRN_E(level,format,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,pa,pb) if ( (level) < this->debug_level() ) { PWRN(format,p0,p1,p2,p3,p4,p5,p6,p7p8,p9,pa,pb); }
#define CPWRN_D(level,format,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,pa) if ( (level) < this->debug_level() ) { PWRN(format,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,pa,pb); }
#define CPWRN_C(level,format,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9) if ( (level) < this->debug_level() ) { PWRN(format,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9); }
#define CPWRN_B(level,format,p0,p1,p2,p3,p4,p5,p6,p7,p8) if ( (level) < this->debug_level() ) { PWRN(format,p0,p1,p2,p3,p4,p5,p6,p7,p8); }
#define CPWRN_A(level,format,p0,p1,p2,p3,p4,p5,p6,p7) if ( (level) < this->debug_level() ) { PWRN(format,p0,p1,p2,p3,p4,p5,p6,p7); }
#define CPWRN_9(level,format,p0,p1,p2,p3,p4,p5,p6) if ( (level) < this->debug_level() ) { PWRN(format,p0,p1,p2,p3,p4,p5,p6); }
#define CPWRN_8(level,format,p0,p1,p2,p3,p4,p5) if ( (level) < this->debug_level() ) { PWRN(format,p0,p1,p2,p3,p4,p5); }
#define CPWRN_7(level,format,p0,p1,p2,p3,p4) if ( (level) < this->debug_level() ) { PWRN(format,p0,p1,p2,p3,p4); }
#define CPWRN_6(level,format,p0,p1,p2,p3) if ( (level) < this->debug_level() ) { PWRN(format,p0,p1,p2,p3); }
#define CPWRN_5(level,format,p0,p1,p2) if ( (level) < this->debug_level() ) { PWRN(format,p0,p1,p2); }
#define CPWRN_4(level,format,p0,p1) if ( (level) < this->debug_level() ) { PWRN(format,p0,p1); }
#define CPWRN_3(level,format,p0) if ( (level) < this->debug_level() ) { PWRN(format,p0); }
#define CPWRN_2(level,msg) if ( (level) < this->debug_level() ) { PWRN(msg); }
#define CPWRN_1(level)
#define CPWRN(...) GET_MACRO(__VA_ARGS__,CPWRN_E,CPWRN_D,CPWRN_C,CPWRN_B,CPWRN_A,CPWRN_9,CPWRN_8,CPWRN_7,CPWRN_6,CPWRN_5,CPWRN_4,CPWRN_3,CPWRN_2,CPWRN_1,CPWRN_0)(__VA_ARGS__)

#if defined __cplusplus

#include <common/string_view.h>
#include <common/type_name.h>
#include <stdexcept>
namespace
{
	inline void faccrete(std::ostringstream &os_, common::string_view fmt_)
	{
		os_ << fmt_;
	}

	template <typename T, typename ... Args>
		void faccrete(std::ostringstream &os_, common::string_view fmt_, const T &t_, Args && ... args_)
		{
			auto c = fmt_.data();
			auto e = c + fmt_.size();
			while ( c != e && *c != '{' )
			{
				os_ << *c++;
			}
			if ( c != e )
			{
				++c; /* skip over '{' */
				if ( c != e && *c == '}' )
				{
					os_ << t_;
				}
				else if ( c != e && *c == ':' && ++c != e && *c == 'x' && ++c != e && *c == '}' )
				{
					/* our sole support of non-trivial formatting: {:x} */
					os_ << std::hex << t_ << std::dec;
				}
				else
				{
					std::string er("Ill formed : ");
					er.append(fmt_.data(), fmt_.size());
					throw std::runtime_error(er);
				}
				++c; /* skip over '}' */
				faccrete(os_, common::string_view(c, e-c), std::forward<Args>(args_) ...);
			}
		}
}

/* Simple support for formatting similar to std::format */
namespace common
{
	template <typename ... Args>
		std::string format(common::string_view fmt, Args && ... args)
		{
			std::ostringstream os;
			faccrete(os, fmt, std::forward<Args>(args) ...);
			return os.str();
		}
}

using pr_fn = void (*)(const char * format, ...);

/* Simple support for logging similar to std::format */
template <typename ... Args>
	void EPRINT(pr_fn pr, common::string_view fmt, Args && ... args)
	{
		pr("%s", common::format(fmt, std::forward<Args>(args) ...).c_str());
	}

#if defined DRD_GET_VALGRIND_THREADID
static inline auto vg_thread_id()
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
	return DRD_GET_VALGRIND_THREADID;
#pragma GCC diagnostic pop
}
#endif

#if LOG_PRINT_TIMESTAMP
# if defined DRD_GET_VALGRIND_THREADID
/* valgrind thread id, C++ thread id */
#  define ELOG_PFX_FMT "<{} {:x}> "
#  define ELOG_PFX_ARGS vg_thread_id(), std::this_thread::get_id()
# else
/* C++ thread id */
#  define ELOG_PFX_FMT "<{:x}> "
#  define ELOG_PFX_ARGS std::this_thread::get_id()
# endif
#else
/* no thread id */
# define ELOG_PFX_FMT "{}"
# define ELOG_PFX_ARGS ""
#endif
/* function name */
#define FLOG_FMT ELOG_PFX_FMT "{} "
#define FLOG_ARGS ELOG_PFX_ARGS, __func__
#define FDBG(fmt, ...) EPRINT(PDBG, FLOG_FMT fmt, FLOG_ARGS, ##__VA_ARGS__)
#define FLOG(fmt, ...) EPRINT(PLOG, FLOG_FMT fmt, FLOG_ARGS, ##__VA_ARGS__)
#define FINF(fmt, ...) EPRINT(PINF, FLOG_FMT fmt, FLOG_ARGS, ##__VA_ARGS__)
#define FWRN(fmt, ...) EPRINT(PWRN, FLOG_FMT fmt, FLOG_ARGS, ##__VA_ARGS__)
#define FERR(fmt, ...) EPRINT(PERR, FLOG_FMT fmt, FLOG_ARGS, ##__VA_ARGS__)
/* ELOG macro: less than FLOG macro. Does not include the function name. */
#define ELOG(fmt, ...) EPRINT(PLOG, ELOG_PFX_FMT fmt, ELOG_PFX_ARGS, ##__VA_ARGS__)
#define CELOG(level, fmt, ...)
/* conditional (by debug_level()) messages */
#define ECOND(level, mac, fmt, ...) ( (level) < this->debug_level() && (mac(fmt, ##__VA_ARGS__), true) )
#define CFLOG(level, fmt, ...) ECOND((level), FLOG, fmt, ##__VA_ARGS__)
#define CFWRN(level, fmt, ...) ECOND((level), FWRN, fmt, ##__VA_ARGS__)
/*
 * Messages with a class name. Code in which 'this' is unavailable (non-member
 * functions, static member functions) should use [C]FLOG instead.
 */
/* object address, class name, function name */
#define FLOGM_FMT ELOG_PFX_FMT "({}) {}::{} "
#define FLOGM_ARGS ELOG_PFX_ARGS, this, type_of(*this), __func__
#define FDBGM(fmt, ...) EPRINT(PDBG, FLOGM_FMT fmt, FLOGM_ARGS, ##__VA_ARGS__)
#define FLOGM(fmt, ...) EPRINT(PLOG, FLOGM_FMT fmt, FLOGM_ARGS, ##__VA_ARGS__)
#define FINFM(fmt, ...) EPRINT(PINF, FLOGM_FMT fmt, FLOGM_ARGS, ##__VA_ARGS__)
#define FWRNM(fmt, ...) EPRINT(PWRN, FLOGM_FMT fmt, FLOGM_ARGS, ##__VA_ARGS__)
#define FERRM(fmt, ...) EPRINT(PERR, FLOGM_FMT fmt, FLOGM_ARGS, ##__VA_ARGS__)
#define CFLOGM(level, fmt, ...) ECOND((level), FLOGM, fmt, ##__VA_ARGS__)
#define CFWRNM(level, fmt, ...) ECOND((level), FWRNM, fmt, ##__VA_ARGS__)
#endif

// clang-format on


#endif  // __COMMON_LOGGING_H__
