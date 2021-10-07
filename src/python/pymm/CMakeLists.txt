cmake_minimum_required(VERSION 3.5) # may work with earlier versions

project(pymm)

option(BUILD_PYMM_NOPACKAGE "do not build a python package - turn ON for development mode" OFF)
option(BUILD_PYMM_VALGRIND "build support for valgrind triggers" OFF)

# set PYMM_DEV=1 in your environment to turn this on
if ($ENV{PYMM_DEV})
  set(BUILD_PYMM_NOPACKAGE ON)
endif()

if (BUILD_PYMM_VALGRIND)
  set(EXTRA_FLAGS "BUILD_PYMM_VALGRIND=1")
endif()

# this is the directory for where the package will be constructed
#
set(PKG_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/pymm")

find_program(PYTHON NAMES "python3.9" "python3.8" "python3.7" "python3.6" "python3.5")

get_filename_component(PYTHON_FILENAME ${PYTHON} NAME)
# get python locations
execute_process ( COMMAND ${PYTHON} -m site --user-site OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process ( COMMAND ${PYTHON} -c "from sysconfig import get_paths as gp; print(gp()['include'])" OUTPUT_VARIABLE PYTHON_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process ( COMMAND ${PYTHON} -c "from sysconfig import get_config_vars as gc; print(gc()['LIBPL'])" OUTPUT_VARIABLE PYTHON_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process ( COMMAND ${PYTHON} -c "from sysconfig import get_config_vars as gc; print(gc()['BLDLIBRARY'])" OUTPUT_VARIABLE PYTHON_BLDLIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)

message("-- Python    site : ${PYTHON_SITE_PACKAGES}")
message("--         stdlib : ${PYTHON_LIB_DIR}")
message("--        include : ${PYTHON_INCLUDE_DIR}")

message("-- Building Python support.")
set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(SETUP_PY_DEPS common pmem ccpm)
set(DEPS "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py ${CMAKE_CURRENT_SOURCE_DIR}/pickled.py")
set(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/timestamp")

string(REGEX REPLACE  "^.*python" "" PYTHON_VERSION ${PYTHON})

message("-- Python version ${PYTHON_VERSION}")
set(LIBDIR "lib.linux-x86_64-${PYTHON_VERSION}")

execute_process ( COMMAND ${PYTHON} -c "import numpy; print(numpy.__path__[0])" OUTPUT_VARIABLE NUMPY_SITE_PACKAGE OUTPUT_STRIP_TRAILING_WHITESPACE)
message("-- Numpy     site : ${NUMPY_SITE_PACKAGE}")

set(PYTHON_SITE_NUMPY_INCLUDES "${NUMPY_SITE_PACKAGE}/core/include")

configure_file(${SETUP_PY_IN} ${SETUP_PY})
configure_file(core/pymm_config.h.in core/pymm_config.h)

file(GLOB ALL_PACKAGE_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.py 
  ${CMAKE_CURRENT_SOURCE_DIR}/core/*.h 
  ${CMAKE_CURRENT_SOURCE_DIR}/core/*.c* )

file(GLOB PACKAGE_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
# copy python files to build

file(MAKE_DIRECTORY ${PKG_SOURCE_DIR})
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pymm/PyMM/Meta)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE DESTINATION ${PKG_SOURCE_DIR})
file(COPY ${PACKAGE_SOURCE_FILES} DESTINATION ${PKG_SOURCE_DIR})

if(BUILD_PYMM_VALGRIND)
  message("-- Building Pymm with Valgrind support")
else()
  message("-- Building Pymm without Valgrind support")
endif()

if(BUILD_PYMM_NOPACKAGE)
  message("-- Building Pymm WITHOUT package (development only)")
  add_custom_target(${PROJECT_NAME}-np ALL
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/*.py ${PKG_SOURCE_DIR}
    COMMAND ${EXTRA_FLAGS} ${PYTHON} ${SETUP_PY} build
    )
else()
  message("-- Building Pymm with site package")
  add_custom_command(
    OUTPUT build.stamp
    COMMAND ${EXTRA_FLAGS} ${PYTHON} ${SETUP_PY} bdist # build wheel binary distribution
    COMMAND cmake -E touch build.stamp
    DEPENDS ${PROJECT_NAME}-meta ${ALL_PACKAGE_SOURCE_FILES} ${SETUP_PY_DEPS}
    )

  # install as a site-package
  install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install --user COMMAND_ERROR_IS_FATAL ANY)")
endif()

unset(BUILD_PYMM_NOPACKAGE)



