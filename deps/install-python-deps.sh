#!/bin/bash
#
# PLEASE RUN THIS AS NON-ROOT
#
# PP uses .site local installs
#
# Python3 should be installed
#
PIP="python3 -m pip"

${PIP} install numpy --user -I

if [[ -z "${TRAVIS_BUILD}" ]]; then
${PIP} install matplotlib --user -I
${PIP} install scikit-image --user -I
${PIP} install torch --user -I
${PIP} install torchvision --user -I
${PIP} install flatbuffers --user -I
${PIP} install parallel_sort --user -I
${PIP} install cython --user -I
${PIP} install chardet --user -I
fi

#
# To support transient memory mode, we use a custom version of numpy
# that allows overloading of memory allocators this is temporary until
# new features enter the mainline
#
# You will need to build this custom version and install as user-site local
#
# Modified NumPy source at https://github.com/dwaddington/numpy
#
# python3.9 setup.py install --user
#
# Alternatively here is the patch:
#
# diff --git a/numpy/core/src/multiarray/alloc.c b/numpy/core/src/multiarray/alloc.c
# index adb4ae128..05ddc70eb 100644
# --- a/numpy/core/src/multiarray/alloc.c
# +++ b/numpy/core/src/multiarray/alloc.c
# @@ -236,7 +236,7 @@ PyDataMem_NEW(size_t size)
#      void *result;
# 
#      assert(size != 0);
# -    result = malloc(size);
# +    result = PyMem_RawMalloc(size);
#      if (_PyDataMem_eventhook != NULL) {
#          NPY_ALLOW_C_API_DEF
#          NPY_ALLOW_C_API
# @@ -258,7 +258,7 @@ PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
#  {
#      void *result;
# 
# -    result = calloc(size, elsize);
# +    result = PyMem_RawCalloc(size, elsize);
#      if (_PyDataMem_eventhook != NULL) {
#          NPY_ALLOW_C_API_DEF
#          NPY_ALLOW_C_API
# @@ -279,7 +279,7 @@ NPY_NO_EXPORT void
#  PyDataMem_FREE(void *ptr)
#  {
#      PyTraceMalloc_Untrack(NPY_TRACE_DOMAIN, (npy_uintp)ptr);
# -    free(ptr);
# +    PyMem_RawFree(ptr);
#      if (_PyDataMem_eventhook != NULL) {
#          NPY_ALLOW_C_API_DEF
#          NPY_ALLOW_C_API
