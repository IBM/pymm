cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(component-hstore CXX)

include(${CMAKE_SOURCE_DIR}/mk/avoid_boost_type_to_string_bug.cmake)

add_compile_options("$<$<CONFIG:Debug>:-O0>")
add_compile_options("$<$<CONFIG:Release>:-DNDEBUG>")

add_subdirectory(./unit_test)

add_compile_options("-fPIC")
add_definitions(-DCONFIG_DEBUG) # P{LOG,DEG,INF,WRN,ERR} control
add_definitions(-DDM_REGION_LOG_GRAIN_SIZE=${DM_REGION_LOG_GRAIN_SIZE})

set(SOURCES_HEADER_CHECK
	src/allocator_cc.cpp
	src/allocator_co.cpp
	src/alloc_key.cpp
	src/as_pin.cpp
	src/bits_to_ints.cpp
	src/bucket_aligned.cpp
	src/bucket_control.cpp
	src/bucket_control_unlocked.cpp
	src/bucket_mutexes.cpp
	src/bucket_ref.cpp
	src/bucket_shared_lock.cpp
	src/bucket_unique_lock.cpp
	src/cond_print.cpp
	src/construction_mode.cpp
	src/content.cpp
	src/cptr.cpp
	src/definite_lock.cpp
	src/dummy_shared_mutex.cpp
	src/fixed_string.cpp
	src/hash_bucket.cpp
	src/heap_co.cpp
	src/heap_access.cpp
	src/histogram_log2.cpp
	src/hop_hash.cpp
	src/hop_hash_base.cpp
	src/hop_hash_debug.cpp
	src/hstore_alloc_type.cpp
	src/hstore_config.cpp
	src/hstore_kv_types.cpp
	src/hstore_nupm.cpp
	src/hstore_nupm_types.cpp
	src/hstore_open_pool.cpp
	src/is_locked.cpp
	src/key_not_found.cpp
	src/lock_impl.cpp
	src/lock_result.cpp
	src/logging.cpp
	src/mod_control.cpp
	src/monitor_emplace.cpp
	src/monitor_pin.cpp
	src/persist_fixed_string.cpp
	src/persist_atomic.cpp
	src/persist_atomic_controller.cpp
	src/persist_map.cpp
	src/persist_map_controller.cpp
	src/persist_data.cpp
	src/persistent.cpp
	src/persister_cc.cpp
	src/persister_nupm.cpp
	src/pobj_bad_alloc.cpp
	src/pointer_pobj.cpp
	src/pool_iterator.cpp
	src/pool_path.cpp
	src/pool_pobj.cpp
	src/pstr_equal.cpp
	src/pstr_hash.cpp
	src/rc_alloc_wrapper_lb.cpp
	src/region.cpp
	src/segment_and_bucket.cpp
	src/segment_count.cpp
	src/segment_layout.cpp
	src/session.cpp
	src/session_base.cpp
	src/size_control.cpp
	src/store_root.cpp
	src/test_flags.cpp
	src/trace_flags.cpp
	src/value_unstable.cpp
)

set(SOURCES
	src/as_emplace.cpp
	src/as_extend.cpp
	src/bad_alloc_cc.cpp
	src/dax_manager.cpp
	src/heap.cpp
	src/hop_hash_exceptions.cpp
	src/hop_hash_log.cpp
	src/hstore.cpp
	src/hstore_factory.cpp
	src/owner.cpp
	src/perishable.cpp
	src/perishable_expiry.cpp
	src/pool_error.cpp
	src/pool_manager.cpp
)

set(SOURCES_CC
	src/heap_cc.cpp
	src/heap_cc_ephemeral.cpp
)

set(SOURCES_RC
	src/heap_rc.cpp
	src/heap_rc_ephemeral.cpp
	src/tracked_header.cpp
)

set(SOURCES_MM
	src/heap_mm.cpp
	src/heap_mc_ephemeral.cpp
	src/heap_mc_shim.cpp
	src/heap_mm_ephemeral.cpp
	src/heap_mr_ephemeral.cpp
	src/heap_mr_shim.cpp
	src/heap_mm_ephemeral.cpp
	src/tracked_header.cpp
)

#add_custom_target(format-hstore
#  COMMAND /usr/bin/clang-format
#  -style=file
#  -i
#  ${ALL_SOURCES})

# if(TEST_HSTORE_PERISHABLE)
#  add_compile_options(-DTEST_HSTORE_PERISHABLE=1)
# endif()

get_property(incdirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
set(sysincludes "-I/usr/include/c++/5 -I/usr/include/x86_64-linux-gnu/c++/5/ -I/usr/include/linux")
# add -I prefix
string(REGEX REPLACE "([^;]+)" "-I\\1" istring "${incdirs}")

include_directories(${CMAKE_SOURCE_DIR}/src/components)
include_directories(${CMAKE_SOURCE_DIR}/src/components/api)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libpmem/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libnupm/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/common/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libccpm/include)
include_directories(${CMAKE_SOURCE_DIR}/src/mm) # pluggable memory manager
include_directories(${CMAKE_INSTALL_PREFIX}/include)
include_directories(../../)

link_directories(${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}) # cityhash
link_directories(${CMAKE_INSTALL_PREFIX}/lib) # cityhash

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

# header checks
add_library(${PROJECT_NAME}-header-check SHARED ${SOURCES_HEADER_CHECK} src/pool_error.cpp)
target_compile_options(${PROJECT_NAME}-header-check PUBLIC "$<$<BOOL:${TEST_HSTORE_PERISHABLE}>:-DMCAS_HSTORE_TEST_PERISHABLE=1>")
set_target_properties(${PROJECT_NAME}-header-check PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_INSTALL_PREFIX}/lib)

# common version
add_library(${PROJECT_NAME} SHARED ${SOURCES} ${SOURCES_RC})
target_compile_definitions(${PROJECT_NAME} PUBLIC "$<$<BOOL:${TEST_HSTORE_PERISHABLE}>:MCAS_HSTORE_TEST_PERISHABLE=1>")
target_link_libraries(${PROJECT_NAME} common pthread numa dl rt boost_system boost_filesystem nupm cityhash gcov)
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_INSTALL_PREFIX}/lib)

# no timestamp version
add_library(${PROJECT_NAME}-nt SHARED ${SOURCES} ${SOURCES_RC})
target_compile_definitions(${PROJECT_NAME}-nt PUBLIC "$<$<BOOL:${TEST_HSTORE_PERISHABLE}>:MCAS_HSTORE_TEST_PERISHABLE=1>" "ENABLE_TIMESTAMP=0")
target_link_libraries(${PROJECT_NAME}-nt common pthread numa dl rt boost_system boost_filesystem nupm cityhash)
set_target_properties(${PROJECT_NAME}-nt PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_INSTALL_PREFIX}/lib)

# crash-consistent allocator version
add_library(${PROJECT_NAME}-cc SHARED ${SOURCES} ${SOURCES_CC})
target_compile_definitions(${PROJECT_NAME}-cc PUBLIC "$<$<BOOL:${TEST_HSTORE_PERISHABLE}>:MCAS_HSTORE_TEST_PERISHABLE=1>" "MCAS_HSTORE_USE_CC_HEAP=4")
target_link_libraries(${PROJECT_NAME}-cc common pthread numa dl rt boost_system boost_filesystem nupm cityhash ccpm)
set_target_properties(${PROJECT_NAME}-cc PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_INSTALL_PREFIX}/lib)

# crash-consistent with perishable expiry
add_library(${PROJECT_NAME}-cc-pe SHARED ${SOURCES} ${SOURCES_CC})
target_compile_definitions(${PROJECT_NAME}-cc-pe PUBLIC "MCAS_HSTORE_TEST_PERISHABLE=1" "MCAS_HSTORE_USE_CC_HEAP=4")
target_link_libraries(${PROJECT_NAME}-cc-pe common pthread numa dl rt boost_system boost_filesystem nupm cityhash ccpm)
set_target_properties(${PROJECT_NAME}-cc-pe PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_INSTALL_PREFIX}/lib)

# test that all trace options compile
add_library(${PROJECT_NAME}-cc-pe-tr SHARED ${SOURCES} ${SOURCES_CC})
target_compile_definitions(${PROJECT_NAME}-cc-pe-tr PUBLIC "MCAS_HSTORE_TEST_PERISHABLE=1" "MCAS_HSTORE_USE_CC_HEAP=4" "HSTORE_TRACE_ALL=1")
target_link_libraries(${PROJECT_NAME}-cc-pe-tr common pthread numa dl rt boost_system boost_filesystem nupm cityhash ccpm)
set_target_properties(${PROJECT_NAME}-cc-pe-tr PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_INSTALL_PREFIX}/lib)

# pluggable (crash-consistent (ccpm), or reconstituting (Rca_LB)) allocator version
add_library(${PROJECT_NAME}-mm SHARED ${SOURCES} src/heap_mm.cpp ${SOURCES_MM})
target_compile_definitions(${PROJECT_NAME}-mm PUBLIC "$<$<BOOL:${TEST_HSTORE_PERISHABLE}>:MCAS_HSTORE_TEST_PERISHABLE=1>" "MCAS_HSTORE_USE_CC_HEAP=7")
target_link_libraries(${PROJECT_NAME}-mm common pthread numa dl rt boost_system boost_filesystem nupm cityhash)
set_target_properties(${PROJECT_NAME}-mm PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_INSTALL_PREFIX}/lib)

# pluggable allocator, perishable version
add_library(${PROJECT_NAME}-mm-pe SHARED ${SOURCES} ${SOURCES_MM})
target_compile_definitions(${PROJECT_NAME}-mm-pe PUBLIC "$<$<BOOL:${TEST_HSTORE_PERISHABLE}>:MCAS_HSTORE_TEST_PERISHABLE=1>" "MCAS_HSTORE_USE_CC_HEAP=7")
target_link_libraries(${PROJECT_NAME}-mm-pe common pthread numa dl rt boost_system boost_filesystem nupm cityhash)
set_target_properties(${PROJECT_NAME}-mm-pe PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_INSTALL_PREFIX}/lib)

# multi-thread capable pluggable allocator version
add_library(${PROJECT_NAME}-mt SHARED ${SOURCES} src/heap_mm.cpp ${SOURCES_MM})
target_compile_definitions(${PROJECT_NAME}-mt PUBLIC "$<$<BOOL:${TEST_HSTORE_PERISHABLE}>:MCAS_HSTORE_TEST_PERISHABLE=1>" "MCAS_HSTORE_USE_CC_HEAP=7" "THREAD_SAFE_HASH=1")
target_link_libraries(${PROJECT_NAME}-mt common pthread numa dl rt boost_system boost_filesystem nupm cityhash)
set_target_properties(${PROJECT_NAME}-mt PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_INSTALL_PREFIX}/lib)

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
install(TARGETS ${PROJECT_NAME}-nt LIBRARY DESTINATION lib)
install(TARGETS ${PROJECT_NAME}-cc LIBRARY DESTINATION lib)
install(TARGETS ${PROJECT_NAME}-cc-pe LIBRARY DESTINATION lib)
install(TARGETS ${PROJECT_NAME}-mm LIBRARY DESTINATION lib)
install(TARGETS ${PROJECT_NAME}-mm-pe LIBRARY DESTINATION lib)
install(TARGETS ${PROJECT_NAME}-mt LIBRARY DESTINATION lib)
