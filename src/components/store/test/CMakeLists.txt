cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(store-tests CXX)

include_directories(../../)
include_directories(../../../)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libnupm/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/common/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/GSL/include)
include_directories(${CMAKE_SOURCE_DIR}/src/components)
include_directories(${CMAKE_INSTALL_PREFIX}/include)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)
link_directories(${CMAKE_INSTALL_PREFIX}/lib64)
set(GTEST_LIB "gtest$<$<CONFIG:Debug>:d>")

add_definitions(-DCONFIG_DEBUG)
add_definitions(-DDM_REGION_LOG_GRAIN_SIZE=${DM_REGION_LOG_GRAIN_SIZE})
add_executable(allocator_efficiency src/allocator_efficiency.cpp)
target_compile_options(allocator_efficiency PUBLIC "$<$<CONFIG:Debug>:-O0>")
target_link_libraries(allocator_efficiency ${ASAN_LIB} common numa pthread dl boost_program_options)

add_executable(dax_address_overlap src/dax_address_overlap.cpp)
target_compile_options(dax_address_overlap PUBLIC "$<$<CONFIG:Debug>:-O0>")
target_link_libraries(dax_address_overlap ${ASAN_LIB} common numa pthread dl boost_program_options)

add_executable(store-test1 src/test1.cpp src/pool_iterator.cpp src/pool_opened.cpp src/pool_instance.cpp)
target_compile_options(store-test1 PUBLIC "$<$<CONFIG:Debug>:-O0>")
target_link_libraries(store-test1 ${ASAN_LIB} common numa ${GTEST_LIB} pthread dl boost_program_options)
