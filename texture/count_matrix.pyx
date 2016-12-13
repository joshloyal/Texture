from libcpp.vector cimport vector
#from libcpp.unordered_map cimport unordered_map as cpp_map
from libcpp.map cimport map as cpp_map

import numpy as np
cimport numpy as np

from preshed.counter cimport PreshCounter
from spacy.tokens.doc cimport Doc

from texture.typedefs cimport Int_t
from texture.arrays cimport to_ndarray

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()


ctypedef cpp_map[Int_t, Int_t] vocab_map
ctypedef cpp_map[Int_t, Int_t].iterator vocab_iter


cdef struct Vocabulary:
    vocab_map index
    Int_t length



cdef void count(Doc doc, Vocabulary* vocab, PreshCounter counts, vector[Int_t]& j_indices):
    cdef int i
    cdef Int_t orth
    cdef vocab_iter it
    for i in range(doc.length):
        orth = doc.c[i].lex.orth
        counts.inc(orth, 1)

        # keep track of vocab seen to build up sparse matrix
        it = vocab.index.find(orth)
        if it == vocab.index.end():
            vocab.index[orth] = vocab.length
            vocab.length += 1

        j_indices.push_back(vocab.index[orth])


def document_matrix(Doc doc):
    cdef vector[Int_t] j_indices
    cdef vector[Int_t] indptr
    cdef Vocabulary vocab
    cdef PreshCounter counts = PreshCounter()

    vocab.index = vocab_map()
    vocab.length = 0

    count(doc, &vocab, counts, j_indices)
    indptr.push_back(j_indices.size())

    return to_ndarray(j_indices.data(), j_indices.size())
