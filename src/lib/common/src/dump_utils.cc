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
  Copyright (C) 2014, 2020 Daniel G. Waddington <daniel.waddington@acm.org>
*/

#include "common/dump_utils.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <execinfo.h>
#include <unistd.h>

void hexdump(const void *data, const size_t len) {
  printf("HEXDUMP-(%p)---------------------------------------------", data);
  assert(len > 0);
  const uint8_t *d = static_cast<const uint8_t *>(data);
  for (size_t i = 0; i < len; i++) {
    if (i % 16 == 0) {
      printf("\n0x%zx:\t", i);
    }
    printf("%x%x ", 0xf & (d[i] >> 4), 0xf & d[i]);
  }
  printf("\n");
  fflush(0);
}

void asciidump(const void *data, const size_t len) {
  printf("ASCIIDUMP-(%p)---------------------------------------------", data);
  assert(len > 0);
  const uint8_t *d = static_cast<const uint8_t *>(data);
  for (size_t i = 0; i < len; i++) {
    if (i % 16 == 0) {
      printf("\n0x%zx:\t", i);
    }
    printf("%c%c ", 0xf & (d[i] >> 4), 0xf & d[i]);
  }
  printf("\n");
}

void dump_backtrace()
{
  void *array[32];
  
  // get void*'s for all entries on the stack
  auto size = backtrace(array, 32);

  // print out all the frames to stderr
  backtrace_symbols_fd(array, size, STDERR_FILENO);
}
