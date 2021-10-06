cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(libccpm-test CXX)

find_library(PROFILER_LIB NAMES profiler PATHS /usr/local/lib,/usr/local/lib64,/usr/lib,/usr/lib64)
if ("${PROFILER_LIB}" STREQUAL "PROFILER_LIB-NOTFOUND")
set(PROFILER "")
else()
set(PROFILER "profiler")
add_definitions(-DHAS_PROFILER=1)
endif()

include_directories(${CMAKE_SOURCE_DIR}/src/lib/libnupm/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libpmem/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libccpm/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/EASTL/include) # test6
include_directories(${CMAKE_SOURCE_DIR}/src/lib/EASTL/include) # immutable_allocator
include_directories(${CMAKE_SOURCE_DIR}/src/lib/EASTL/test/packages/EABase/include/Common) # test5
include_directories(${CMAKE_INSTALL_PREFIX}/include) # gtest

link_directories(${CMAKE_SOURCE_DIR}/src/lib/libnupm/)
link_directories(${CMAKE_SOURCE_DIR}/src/lib/libpmem/)
link_directories(${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}) # ndctl
link_directories(${CMAKE_INSTALL_PREFIX}/lib)
link_directories(${CMAKE_INSTALL_PREFIX}/lib64) # gtest

if (CMAKE_BUILD_TYPE STREQUAL "Release")
  set(GTEST_LIB gtest)
else()
  set(GTEST_LIB gtestd)
endif()

add_executable(libccpm-test1 test1.cpp)
add_executable(libccpm-test2 test2.cpp)
add_executable(libccpm-test3 test3.cpp)
add_executable(libccpm-test5 test5.cpp store_map.cpp)
add_executable(libccpm-test6 test6.cpp store_map.cpp)
add_executable(libccpm-test7 test7.cpp store_map.cpp)

target_compile_options(libccpm-test1 PUBLIC "$<$<CONFIG:Debug>:-O0>")
target_compile_options(libccpm-test2 PUBLIC "$<$<CONFIG:Debug>:-O0>")
target_compile_options(libccpm-test3 PUBLIC "$<$<CONFIG:Debug>:-O0>")
target_compile_options(libccpm-test5 PUBLIC "$<$<CONFIG:Debug>:-O0>")
target_compile_options(libccpm-test6 PUBLIC "$<$<CONFIG:Debug>:-O0>")
target_compile_options(libccpm-test7 PUBLIC "$<$<CONFIG:Debug>:-O0>")

target_link_libraries(libccpm-test1 ${ASAN_LIB} ${GTEST_LIB} nupm) # add profiler for google profiler
target_link_libraries(libccpm-test2 ${ASAN_LIB} ${GTEST_LIB} ccpm pmem) # add profiler for google profiler
target_link_libraries(libccpm-test3 ${ASAN_LIB} ${GTEST_LIB} ccpm pmem) # add profiler for google profiler
target_link_libraries(libccpm-test5 ${ASAN_LIB} ${GTEST_LIB} ccpm pmem)
target_link_libraries(libccpm-test6 ${ASAN_LIB} ${GTEST_LIB} ccpm pmem profiler)
target_link_libraries(libccpm-test7 ${ASAN_LIB} ${GTEST_LIB} ccpm pmem)
