import numpy as np
cimport numpy as np

from texture.typedefs cimport Int_t


cdef np.ndarray[Int_t, ndim=1] to_ndarray(Int_t* c_array, Int_t n_elements)
