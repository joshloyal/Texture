from __future__ import unicode_literals

import spacy

from texture import count_matrix

nlp = spacy.load('en')

doc = nlp('Hello, world. Here are two two sentences.')

counts = count_matrix.document_matrix(doc)
