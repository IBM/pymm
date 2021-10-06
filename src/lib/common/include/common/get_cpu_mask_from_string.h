#ifndef __COMMON_GET_CPU_MASK_FROM_STRING__
#define __COMMON_GET_CPU_MASK_FROM_STRING__

#include <common/common.h>
#include <common/get_vector_from_string.h>

#include <common/logging.h> /* PERR */
#include <common/cpu.h> /* cpu_mask_t */

#include <string>

namespace common
{
  cpu_mask_t get_cpu_mask_from_string(std::string core_string);
}

#endif
