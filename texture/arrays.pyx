from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np


np.import_array()

cdef class ArrayFinalizer:
    """Object whose sole purpose is to de-allocate the memory
    of the array of data it is pointing to. Used by `set_base`
    to properly clean up numpy arrays created by C arrays.
    """
    cdef void* data_ptr
    def __dealloc__(self):
        if self.data_ptr is not NULL:
            free(self.data_ptr)


cdef void set_base(np.ndarray np_array, void *c_array):
    """set the base of a newly allocated numpy array to
    the underlying C array, so that it is properly cleaned
    up by python (handled by the finalizer class)."""
    cdef ArrayFinalizer finalizer = ArrayFinalizer()
    finalizer.data_ptr = <void*>c_array
    np.set_array_base(np_array, finalizer)


cdef np.ndarray[Int_t, ndim=1] _to_ndarray(Int_t* c_array, Int_t n_elements):
    """Creates a numpy ndarray from a C array that will clean up after itself
    during destruction."""
    cdef Int_t[:] mv = <Int_t[:n_elements]>c_array
    cdef np.ndarray[Int_t, ndim=1] np_array = np.asarray(mv)
    set_base(np_array, c_array)
    return np_array



cdef np.ndarray[Int_t, ndim=1] to_ndarray(vector[Int_t]* vector):
    """Convert a C++ vector into a numpy ndarray that will clean
    up after itself during destruction."""
    return _to_ndarray(vector.data(), vector.size())


cdef Int_t* copy_array(Int_t* array, int n_elements) nogil:
    """creates a new array (`array_copy`) and copies elements of
    `array` into it.
    """
    cdef Int_t* array_copy = <Int_t*>malloc(sizeof(Int_t) * n_elements)
    memcpy(array_copy, array, sizeof(Int_t) * n_elements)
    return array_copy
