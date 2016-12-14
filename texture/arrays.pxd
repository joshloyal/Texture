from libcpp.vector cimport vector

import numpy as np
cimport numpy as np

from texture.typedefs cimport Int_t


cdef np.ndarray[Int_t, ndim=1] to_ndarray(vector[Int_t]* vector)
