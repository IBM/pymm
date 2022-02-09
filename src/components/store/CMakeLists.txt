
cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

execute_process( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )

# x86 only components
if( ${ARCHITECTURE} STREQUAL "x86_64" )
  add_subdirectory(hstore)
endif()

add_subdirectory(mapstore)

# broken in pymm distribution
#add_subdirectory(test)
