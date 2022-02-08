cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(libnupm-test CXX)

set(CMAKE_CXX_STANDARD 14)

include_directories(${CMAKE_SOURCE_DIR}/src/lib/libnupm/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libpmem/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libccpm/include)
include_directories(../src)

link_directories(${CMAKE_SOURCE_DIR}/src/lib/libnupm/)
link_directories(${CMAKE_SOURCE_DIR}/src/lib/libpmem/)
link_directories(${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}) # ndctl
link_directories(${CMAKE_INSTALL_PREFIX}/lib)
link_directories(${CMAKE_INSTALL_PREFIX}/lib64)

add_compile_options(-g -fPIC)

set(GTEST_LIB "gtest$<$<CONFIG:Debug>:d>")

add_executable(libnupm-test test1.cpp)
target_link_libraries(libnupm-test ${ASAN_LIB} common numa ndctl ${GTEST_LIB} pthread dl nupm pmem z gcov)  # add profiler for google profiler

add_executable(libnupm-test2 test2.cpp)
target_compile_options(libnupm-test2 PUBLIC $<$<CONFIG:Debug>:-O0> -g -pedantic -Wall -Werror -Wextra -Wcast-align -Wcast-qual -Wconversion -Weffc++ -Wold-style-cast -Wredundant-decls -Wshadow -Wtype-limits -Wunused-parameter -Wwrite-strings -Wformat=2)
target_link_libraries(libnupm-test2 ${ASAN_LIB} ${GTEST_LIB} pthread dl nupm gcov) # add profiler for google profiler
