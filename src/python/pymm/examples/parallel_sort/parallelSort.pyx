# cython: wraparound = False
# cython: boundscheck = False
import numpy as np
cimport numpy as np
import cython
cimport cython 

ctypedef fused real:
    cython.char
    cython.uchar
    cython.short
    cython.ushort
    cython.int
    cython.uint
    cython.long
    cython.ulong
    cython.longlong
    cython.ulonglong
    cython.float
    cython.double

cdef extern from "<parallel/algorithm>" namespace "__gnu_parallel":
    cdef void sort[T](T first, T last) nogil 

def numpyParallelSort(real[:] a):
    "In-place parallel sort for numpy types"
    sort(&a[0], &a[a.shape[0]])
