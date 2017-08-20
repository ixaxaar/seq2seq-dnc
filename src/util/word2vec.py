#!/usr/bin/env python3

import gensim
import numpy as np

from util import *


def load_glove(path):
    word2vec = gensim.models.\
        word2vec.Word2Vec.\
        load_word2vec_format(
            path,
            binary=False,
            unicode_errors='ignore'
        )
    return word2vec


def load_numberbatch(path):
    return gensim.models.KeyedVectors.load_word2vec_format(path)


def vec(sentence, w2v, lang='en', unk=np.zeros(300)):
    return [np.array(n[u(lang, k)]) if u(lang, k) in n else unk for k in normalize(sentence).split(' ')]
