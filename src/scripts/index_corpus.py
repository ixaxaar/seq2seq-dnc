#!/usr/bin/env python3

import argparse

from util import *

parser = argparse.ArgumentParser(description='''
  Language distionary creator
  Creates a frequency-counted dictionary from a corpus
''')
parser.add_argument('-corpus', required=True,
                    help='Path to the text corpus to index')
parser.add_argument('-name', required=True, help='Name of this lang')
opts = parser.parse_args()


def index_corpus(name, corpus):
    '''Index a corpus to form a dictionary

    Arguments:
        name {string} -- Name of this language / corpus
        corpus {string} -- Path to the text corpus to index
    '''
    l = Lang(name)
    with open(corpus) as i:
        for line in i:
            l.index(normalize(line))
    l.save(name + '.lang')

if __name__ == '__main__':
    index_corpus(opts.name, opts.corpus)
