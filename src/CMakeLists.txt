cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

add_subdirectory(lib)

if(BOOTSTRAP_DEPS)
else()
add_subdirectory(mm)
add_subdirectory(components)
add_subdirectory(python)
endif()
