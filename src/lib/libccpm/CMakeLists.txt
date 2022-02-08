cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(ccpm CXX C)

set(CMAKE_CXX_STANDARD 17)
include(${CMAKE_SOURCE_DIR}/mk/avoid_boost_type_to_string_bug.cmake)

add_compile_options(-g -pedantic -Wall -Werror -Wextra -Wcast-align -Wcast-qual -Wconversion -Weffc++ -Wold-style-cast -Wredundant-decls -Wshadow -Wtype-limits -Wunused-parameter -Wwrite-strings -Wformat=2)
add_compile_options("$<$<CONFIG:Debug>:-O0>")

include(../../../mk/clang-dev-tools.cmake)

# find_library(LIB_JEMALLOC libjemalloc.so HINTS ${CMAKE_INSTALL_PREFIX}/lib)
#find_library(LIB_NDCTL libndctl.so HINTS ${CMAKE_INSTALL_PREFIX}/lib)

include_directories(${CMAKE_SOURCE_DIR}/src/components) # immutable_allocator
include_directories(${CMAKE_SOURCE_DIR}/src/lib/common/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libccpm/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/GSL/include/)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/EASTL/include) # immutable_allocator

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fopenmp")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

add_definitions(-DCONFIG_DEBUG) # P{LOG,DEG,INF,WRN,ERR} control

set(SOURCES src/area_ctl.cpp src/area_top.cpp src/atomic_word.cpp src/cca.cpp src/ccpm.cpp src/doubt.cpp src/log.cpp src/tracker_log.cpp src/value_tracked.cpp src/container_cc.cpp)
add_library(${PROJECT_NAME} SHARED ${SOURCES})
add_library(${PROJECT_NAME}-static STATIC ${SOURCES})

target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Debug>:-O0>")
target_compile_options(${PROJECT_NAME}-static PUBLIC "$<$<CONFIG:Debug>:-O0>" "-fPIC")
target_link_libraries(${PROJECT_NAME} dl)

add_subdirectory(unit_test)

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
install(TARGETS ${PROJECT_NAME}-static ARCHIVE DESTINATION lib)
install(DIRECTORY "src/" DESTINATION "include/ccpm" FILES_MATCHING PATTERN "./include/ccpm/*")

# # force early install because others depend on this.  this seems a hack. is there
# # a better way?
# #
# add_custom_command(
#   TARGET ${PROJECT_NAME} POST_BUILD
#   COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/include/ccpm
#   COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/lib
#   COMMAND cp ${PROJECT_SOURCE_DIR}/src/*.h ${CMAKE_INSTALL_PREFIX}/include/ccpm
#   COMMAND cp -r ${PROJECT_SOURCE_DIR}/src/x86_64 ${CMAKE_INSTALL_PREFIX}/include/ccpm
#   COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/libccpm.so ${CMAKE_INSTALL_PREFIX}/lib
#   )

