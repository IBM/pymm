cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)

project(mm-plugin-ccpm C CXX)

add_compile_options("-fvisibility=hidden")

include_directories(${CMAKE_SOURCE_DIR}/src/lib/common/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libpmem/include)
include_directories(${CMAKE_SOURCE_DIR}/src/components)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libccpm/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libmm/src)
#include_directories(${CMAKE_SOURCE_DIR}/src/mm/ccpm/include)
include_directories(${CMAKE_INSTALL_PREFIX}/include)

#link_directories(${CMAKE_CURRENT_BINARY_DIR}/jemalloc-source/build/lib)
link_directories(${CMAKE_INSTALL_PREFIX}/lib/)


file(GLOB SOURCES
  # ../src/lib/libccpm/src/area_ctl.cpp
  # ../src/lib/libccpm/src/cca.cpp
  # ../src/lib/libccpm/src/doubt.cpp 
  # ../src/lib/libccpm/src/tracker_log.cpp 
  # ../src/lib/libccpm/src/area_top.cpp 
  # ../src/lib/libccpm/src/ccpm.cpp 
  # ../src/lib/libccpm/src/log.cpp 
  # ../src/lib/libccpm/src/value_tracked.cpp 
  # ../src/lib/libccpm/src/atomic_word.cpp 
  # ../src/lib/libccpm/src/container_cc.cpp
  src/mm_plugin.cpp)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_compile_options(${PROJECT_NAME} PUBLIC "-fPIC")

target_link_libraries(${PROJECT_NAME} -static-libgcc -static-libstdc++ common-static ccpm-static pmem-static dl pthread)

# set the linkage in the install/lib
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)


