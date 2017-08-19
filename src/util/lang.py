#!/usr/bin/env python3

import pickle

from .logs import log
from .special import *

class Lang:
    """Langauge dictionary

    Creates a frequency-sorted dictionary of a corpus.
    Usage:
    Build a lang:
    ```python
    l = Lang('german')

    with open('corpus.txt') as i:
        # index every sentence form the corpus
        for sentence in i:
            l.index(i)

    # save the lang dict
    l.save(l.name+'.lang')
    ```

    Load a lang:
    ```python
    l = Lang('german')
    l.load('german.lang')
    ```
    """
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = default_dict
        self.n_words = 7  # Count default tokens

    def index(self, sentence):
        """Index a sentence

        Arguments:
            sentence {string} -- Sentence to be indexed
        """
        for word in sentence.split(' '):
            self._index_word(word)

    def _index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        """Trim vocabulary

        Remove words below a certain count threshold

        Arguments:
            min_count {Cutoff frequency} -- Words with frequency less than this will be removed

        Returns:
            list -- List of retained words
        """
        # TODO: why this?
        # if self.trimmed:
        #     return
        # self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        log.info('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(
                keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = default_dict
        self.n_words = 7  # Count default tokens

        for word in keep_words:
            self._index_word(word)
        return keep_words

    def save(self, path):
        """Save a lang

        Arguments:
            path {string} -- File name with full path
        """
        with open(path, 'wb') as out:
            data = {
                'name': self.name,
                'trimmed': self.trimmed,
                'word2index': self.word2index,
                'word2count': self.word2count,
                'index2word': self.index2word,
                'n_words': self.n_words
            }
            pickle.dump(data, out, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        """Load a saved lang

        Arguments:
            path {string} -- File name with full path
        """
        with open(path, 'rb') as inp:
            data = pickle.load(inp)
            self.name = data['name']
            self.trimmed = data['trimmed']
            self.word2index = data['word2index']
            self.word2count = data['word2count']
            self.index2word = data['index2word']
            self.n_words = data['n_words']

