/*
   Copyright [2020] [IBM Corporation]

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
  Copyright (C) 2020, Daniel G. Waddington <daniel.waddington@ibm.com>
*/

#ifndef __COMMON_TIME_H__
#define __COMMON_TIME_H__

#include <common/common.h>
#include <limits.h>
#include <time.h>
#include <string.h>
#include <chrono>
#include <ctime>
#include <cmath>
#include <sstream>
#include <common/types.h>
#include <common/exceptions.h>

namespace common
{

/* timespec operations */
struct timespec timespec_subtract(const struct timespec& x,
				  const struct timespec& y);
struct timespec timespec_add(const struct timespec& x,
			     const struct timespec& y);

class Timepoint;

/** 
 * -------------------------------------------------------------------
 * Class representing a universal epoch time at nanosecond granularity
 * -------------------------------------------------------------------
 * 
 */
class Timespec : public timespec
{
public:
  Timespec() {
    tv_sec = 0;
    tv_nsec = 0;
  };

  /* Why is seconds not time_t, as in the two-argument constructor? */
  Timespec(const int seconds) {
     tv_sec = seconds;
     tv_nsec = 0;
  };

  Timespec(const struct timespec& ts) {
    tv_sec = ts.tv_sec;
    tv_nsec = ts.tv_nsec;
  };

  Timespec(const time_t sec, const long nsec) {
    tv_sec = sec;
    tv_nsec = nsec;
  }

  inline Timespec& add_seconds(const unsigned long seconds) {
    tv_sec += seconds;
    return *this;
  }

  inline Timespec& sub_seconds(const unsigned long seconds) {
    tv_sec -= seconds;
    return *this;
  }

  inline bool operator==(const Timespec& ts) {
    return ((tv_sec == ts.tv_sec) &&
	    (tv_nsec == ts.tv_nsec));
  }

  inline bool operator<(const Timespec& ts) {
    if(tv_sec < ts.tv_sec) return true;
    else if(tv_sec > ts.tv_sec) return false;
    else return tv_nsec < ts.tv_nsec;
  }

  inline bool operator<=(const Timespec& ts) {
    if(tv_sec <= ts.tv_sec) return true;
    else if(tv_sec > ts.tv_sec) return false;
    else return tv_nsec <= ts.tv_nsec;
  }

  inline bool operator>(const Timespec& ts) {
    if(tv_sec > ts.tv_sec) return true;
    else if(tv_sec < ts.tv_sec) return false;
    else return tv_nsec > ts.tv_nsec;
  }

  inline bool operator>=(const Timespec& ts) {
    if(tv_sec < ts.tv_sec) return false;
    else if(tv_sec == ts.tv_sec)
      return tv_nsec >= ts.tv_nsec;
    else return true;
  }  
  
  inline uint64_t seconds() const {
    return (tv_sec & LONG_MAX);
  }

  inline bool is_nil() const {
    return (tv_sec == 0) && (tv_nsec == 0);
  }

  Timepoint to_timepoint() const;
  
  inline bool is_defined() const {
    return tv_sec == 0 && tv_nsec == 0;
  }

  std::string str() const {
    std::stringstream ss;
    ss << tv_sec << ":" << tv_nsec;
    return ss.str();
  }
};

static const Timespec Epoch_nil;

/* epoch time (nanosecond resolution) */
using epoch_time_t = common::Timespec;

inline epoch_time_t epoch_now() {
  epoch_time_t ts;
  if(clock_gettime(CLOCK_REALTIME, &ts) != 0)
    throw General_exception("clock_gettime failed");
  assert(ts.tv_sec > 0);
  return ts;
}

inline epoch_time_t operator-(const epoch_time_t& t1,
			      const epoch_time_t& t2)
{
  return timespec_subtract(t1, t2);
}

inline epoch_time_t operator+(const epoch_time_t& t1,
			      const epoch_time_t& t2)
{
  return timespec_add(t1, t2);
}


/**
 * --------------------------------------------
 * Timestamp class based on std C++ time_points
 * --------------------------------------------
 * 
 */
class Timepoint : public std::chrono::time_point<std::chrono::system_clock>
{
private:
  using time_point_t = std::chrono::time_point<std::chrono::system_clock>;
  
public:

  /** 
   * Default constructor (will set timestamp to current rdtsc)
   * 
   */
  Timepoint();

  /** 
   * Construct timestamp from epoch
   * 
   * @param epoch 
   */
  Timepoint(const common::epoch_time_t& epoch);

  /** 
   * Cast operator to raw value
   * 
   * 
   * @return Unsigned long raw
   */
  operator unsigned long() const { return raw(); }

  /** 
   * Get raw counter value
   * 
   * 
   * @return 64bit raw value
   */
  unsigned long raw() const;

  /** 
   * Update timestamp
   * 
   */
  void update();

  /** 
   * Convert timestamp to epoch form
   * 
   * 
   * @return Converted time
   */
  common::epoch_time_t to_epoch() const;
};

/**
 * ------------------------------------
 * Timestamp class based on h/w TSC
 * ------------------------------------
 * 
 */
class Tsc
{
public:

  /** 
   * Default constructor (will set timestamp to current rdtsc)
   * 
   */
  Tsc();


  /** 
   * Construct timestamp from epoch
   * 
   * @param epoch 
   */
  Tsc(const common::epoch_time_t& epoch);

  /** 
   * Cast operator to raw value
   * 
   * 
   * @return Unsigned long raw
   */
  operator unsigned long() const { return raw(); }

  /** 
   * Get raw counter value
   * 
   * 
   * @return 64bit raw value
   */
  unsigned long raw() const;

  /** 
   * Update timestamp
   * 
   */
  void update();

  /** 
   * Convert timestamp to epoch form
   * 
   * 
   * @return Converted time
   */
  common::epoch_time_t to_epoch() const;

private:
  uint64_t _tsc;
};  


/** 
 * Class hooks the Timestamp class to an implementation
 * 
 */
template<class Base_timestamp>
class Timestamp : public Base_timestamp
{
public:
  Timestamp(const common::epoch_time_t& epoch) : Base_timestamp(epoch) {}
  Timestamp() : Base_timestamp() {}
};

/* timestamp time using rdtsc directly */
//using tsc_time_t = Timestamp<Tsc>;

/* timestamp time using Std C++ Timepoint which is optimized to not system call */
using tsc_time_t = Timestamp<Timepoint>;


}



#endif // __COMMON_TIME_H__
