/*
   Copyright [2017-2019] [IBM Corporation]
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

/*
  Authors:
  Copyright (C) 2019-2020, Daniel G. Waddington <daniel.waddington@ibm.com>
*/

#ifndef __ERRORS_H__
#define __ERRORS_H__

#ifndef ERROR_ENUMS
#define ERROR_ENUMS                                                     \
  enum {                                                                \
    S_OK = 0,                                                           \
    S_OK_CREATED = 1,                                                   \
    S_MORE = 2,                                                         \
    S_USER0 = 3,                                                        \
    S_USER1 = 4,                                                        \
    E_FAIL = -1,                                                        \
    E_INVALID_REQUEST = -2,                                             \
    E_INVAL = -2,                                                       \
    E_INSUFFICIENT_QUOTA = -3,                                          \
    E_NOT_FOUND = -4,                                                   \
    E_INSUFFICIENT_RESOURCES = -5,                                      \
    E_NO_RESOURCES = -6,                                                \
    E_INSUFFICIENT_SPACE = -7,                                          \
    E_INSUFFICIENT_BUFFER = -7,                                         \
    E_BUSY = -9,                                                        \
    E_TAKEN = -10,                                                      \
    E_LENGTH_EXCEEDED = -11,                                            \
    E_BAD_OFFSET = -12,                                                 \
    E_BAD_PARAM = -13,                                                  \
    E_NO_MEM = -14,                                                     \
    E_NOT_SUPPORTED = -15,                                              \
    E_OUT_OF_BOUNDS = -16,                                              \
    E_NOT_INITIALIZED = -17,                                            \
    E_NOT_IMPL = -18,                                                   \
    E_NOT_ENABLED = -19,                                                \
    E_SEND_TIMEOUT = -20,                                               \
    E_RECV_TIMEOUT = -21,                                               \
    E_BAD_FILE = -22,                                                   \
    E_FULL = -23,                                                       \
    E_EMPTY = -24,                                                      \
    E_INVALID_ARG = -25,                                                \
    E_BAD_SEMANTICS = -26,                                              \
    E_EOF = -27,                                                        \
    E_ALREADY = -28,                                                    \
    E_ALREADY_EXISTS = -28,                                             \
    E_NO_RESPONSE = -29,                                                \
    E_TIMEOUT = -30,                                                    \
    E_MAX_REACHED = -31,                                                \
    E_NO_INDEX = -32,                                                   \
    E_LOCKED = -33,                                                     \
    E_ITERATOR_DISTURBED = -34,                                         \
    E_ERROR_BASE = -50,                                                 \
  }
#endif

ERROR_ENUMS; /* add to global namespace also */

#endif
