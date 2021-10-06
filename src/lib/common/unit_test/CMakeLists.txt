cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(libcommon-test CXX)

include_directories(${CMAKE_SOURCE_DIR}/src/lib/common/include)
include_directories(${CMAKE_INSTALL_PREFIX}/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/googletest/googletest/include)

link_directories(${CMAKE_BINARY_DIR}/lib)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)
link_directories(${CMAKE_INSTALL_PREFIX}/lib64)

add_definitions(-DCONFIG_DEBUG)

add_executable(libcommon-test1 test1.cpp)

target_compile_options(libcommon-test1 PUBLIC $<$<CONFIG:Debug>:-O0> -g -pedantic -Wall -Werror -Wextra -Wcast-align -Wcast-qual -Weffc++ -Wold-style-cast -Wredundant-decls -Wshadow -Wtype-limits -Wunused-parameter -Wwrite-strings -Wformat=2) # -Wconversion

if (CMAKE_BUILD_TYPE STREQUAL "Release")
  set(GTEST_LIB gtest)
else()
  set(GTEST_LIB gtestd)
endif()

target_link_libraries(libcommon-test1 ${GTEST_LIB} common boost_system pthread dl numa gcov )  # add profiler for google profiler
