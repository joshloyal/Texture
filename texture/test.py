from __future__ import unicode_literals

import spacy

import texture

from sklearn import datasets

nlp = spacy.load('en')


texts = datasets.fetch_20newsgroups(subset='train').data
X, counts = texture.document_matrix(texts[:5], nlp.tokenizer)
#for doc in nlp.pipe(texts, batch_size=10000, n_threads=4):
#    pass
#    #X, counts = texture.document_matrix(doc)
