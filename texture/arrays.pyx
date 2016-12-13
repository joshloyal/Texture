from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from cpython cimport PyObject, Py_INCREF

import numpy as np
cimport numpy as np


np.import_array()

cdef class _finalizer:
    """Object whose sole purpose is to de-allocate the memory
    of the array of data it is pointing to. Used by `set_base`
    to properly clean up numpy arrays created by C arrays.
    """
    cdef void* _data
    def __dealloc__(self):
        if self._data is not NULL:
            free(self._data)


cdef void set_base(np.ndarray np_array, void *c_array):
    """set the base of a newly allocated numpy array to
    the underlying C array, so that it is properly cleaned
    up by python (handled by the finalizer class).
    """
    cdef _finalizer f = _finalizer()
    f._data = <void*>c_array
    np.set_array_base(np_array, f)


#cdef double *copy_array(double *array, int n_elements) nogil:
#    """creates a new array (`array_copy`) and copies elements of
#    `array` into it.
#    """
#    cdef double *array_copy = <double*>malloc(sizeof(double) * n_elements)
#    memcpy(array_copy, array, sizeof(double) * n_elements)
#    return array_copy

cdef np.ndarray[Int_t, ndim=1] to_ndarray(Int_t* c_array, Int_t n_elements):
    """Creates a numpy ndarray from a C array that will clean up after itself
    during destruction.
    """
    cdef Int_t[:] mv = <Int_t[:n_elements]>c_array
    cdef np.ndarray[Int_t, ndim=1] np_array = np.asarray(mv)
    #set_base(np_array, c_array)
    return np_array

#cdef class ArrayWrapper:
#    cdef void* data_ptr
#    cdef int size
#
#    cdef set_data(self, int size, void* data_ptr):
#        """ Set the data of the array
#
#        This cannot be done in the constructor as it must recieve C-level
#        arguments.
#
#        Parameters:
#        -----------
#        size: int
#            Length of the array.
#        data_ptr: void*
#            Pointer to the data
#
#        """
#        self.data_ptr = data_ptr
#        self.size = size
#
#    def __array__(self):
#        """ Here we use the __array__ method, that is called when numpy
#            tries to get an array from the object."""
#        cdef np.npy_intp shape[1]
#        shape[0] = <np.npy_intp> self.size
#        # Create a 1D array, of length 'size'
#        ndarray = np.PyArray_SimpleNewFromData(1, shape,
#                                               np.NPY_INT32, self.data_ptr)
#        return ndarray
#
#    def __dealloc__(self):
#        """ Frees the array. This is called by Python when all the
#        references to the object are gone. """
#        free(<void*>self.data_ptr)
#
#
#cdef to_ndarray(int size, Int_t* array):
#    """ Python binding of the 'compute' function in 'c_code.c' that does
#        not copy the data allocated in C.
#    """
#    cdef np.ndarray ndarray
#
#    array_wrapper = ArrayWrapper()
#    array_wrapper.set_data(size, <void*> array)
#    ndarray = np.array(array_wrapper, copy=False)
#    # Assign our object to the 'base' of the ndarray object
#    ndarray.base = <PyObject*> array_wrapper
#    # Increment the reference count, as the above assignement was done in
#    # C, and Python does not know that there is this additional reference
#    Py_INCREF(array_wrapper)
#
#
#    return ndarray
