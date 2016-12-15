from libcpp.vector cimport vector
#from libcpp.unordered_map cimport unordered_map as cpp_map
from libcpp.map cimport map as cpp_map

import numpy as np
cimport numpy as np

from scipy import sparse

from preshed.counter cimport PreshCounter
from spacy.tokens.doc cimport Doc
from spacy.structs cimport LexemeC
from spacy.attrs cimport attr_id_t
from spacy.attrs cimport IS_PUNCT
from spacy.typedefs cimport flags_t
from spacy.tokenizer cimport Tokenizer

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



cdef inline bint c_check_flag(const LexemeC* lexeme, attr_id_t flag_id) nogil:
    """Taken from Spacy Lexeme.pxd"""
    cdef flags_t one = 1
    if lexeme.flags & (one << flag_id):
        return True
    else:
        return False


cdef void count(Doc doc, Vocabulary* vocab, PreshCounter counts, vector[Int_t]* j_indices):
    cdef int i
    cdef Int_t orth
    cdef vocab_iter it
    cdef const LexemeC* lex

    for i in range(doc.length):
        lex = doc.c[i].lex

        if c_check_flag(lex, IS_PUNCT):
            continue

        orth = lex.orth
        counts.inc(orth, 1)

        # keep track of vocab seen to build up sparse matrix
        it = vocab.index.find(orth)
        if it == vocab.index.end():
            vocab.index[orth] = vocab.length
            vocab.length += 1

        j_indices.push_back(vocab.index[orth])


def _document_matrix(list docs, Tokenizer tokenizer):
    cdef Vocabulary vocab
    cdef PreshCounter counts = PreshCounter()
    cdef vector[Int_t]* j_indices = new vector[Int_t]()
    cdef vector[Int_t]* indptr = new vector[Int_t]()

    vocab.index = vocab_map()
    vocab.length = 0

    indptr.push_back(0)
    for doc in docs:
        count(tokenizer(doc), &vocab, counts, j_indices)
        indptr.push_back(j_indices.size())

    return to_ndarray(j_indices), to_ndarray(indptr), counts


def document_matrix(list docs, Tokenizer tokenizer, dtype='int32'):
    j_indices, indptr, counts = _document_matrix(docs, tokenizer)

    #j_indices, indptr, counts = _document_matrix(doc)
    values = np.ones(len(j_indices))
    X = sparse.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr) - 1, len(counts)),
                          dtype=dtype)
    X.sum_duplicates()
    return X, counts
