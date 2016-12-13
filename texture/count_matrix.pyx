from libcpp.vector cimport vector
#from libcpp.unordered_map cimport unordered_map as cpp_map
from libcpp.map cimport map as cpp_map

import numpy as np
cimport numpy as np
from preshed.counter cimport PreshCounter
from spacy.tokens.doc cimport Doc


ctypedef np.int32_t INT_t
ctypedef cpp_map[INT_t, INT_t].iterator vocab_iter


cdef class Corpus:
    """A cython class using spacy to efficiently build corpora statistics."""
    pass


def count(Doc doc):
    cdef PreshCounter counts = PreshCounter()
    cdef int i
    cdef vector[int] j_indices
    cdef INT_t orth
    cdef cpp_map[INT_t, INT_t] vocab = cpp_map[INT_t, INT_t]()
    cdef vocab_iter it
    cdef INT_t vocab_counter = 0

    for i in range(doc.length):
        orth = doc.c[i].lex.orth
        counts.inc(orth, 1)

        # keep track of vocab seen to build up sparse matrix
        if it == vocab.end():
            vocab[orth] = vocab_counter
            vcoab_counter += 1

        j_indices.push_back(vocab[orth])

    return counts
