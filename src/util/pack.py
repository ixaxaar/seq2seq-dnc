#!/usr/bin/env python3

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack

from util.logs import log
from util.cuda import *

from .special import *


def pad_seq(seq, max_len=MAX_LENGTH):
    '''Pad a sequence

    Pad a sequence upto a maximum length

    Arguments:
        seq {numpy.array} -- A 1D Numpy ndarray

    Keyword Arguments:
        max_len {int} -- Maximum length of the sequence (default: {MAX_LENGTH})

    Returns:
        numpy.ndarray -- A 1D Numpy ndarray
    '''
    pad = max_len - seq.shape[0]
    if pad > 0:
        return np.concatenate([seq, np.zeros((pad))])
    else:
        return seq[0:max_len]


def mk_sentence(sentence, lang):
    '''Construct a padded vectorized sentence

    Construct a vectorized sentence,
    append with an `EOS`
    prepend with a `SOS`
    replace OOVs with `UNK`

    Arguments:
        sentence {list} -- A tokenized sentence
        lang {util.Lang} -- Language dict

    Returns:
        numpy.ndarray --
    '''
    return pad_seq(np.array([SOS] + [lang.word2index[w] if w in lang.word2index else UNK for w in sentence] + [EOS], dtype='float32'))


def mk_target(target, idx):
    '''Sort the target sentences according to given sortig order

    Arguments:
        target {list} -- Target
        idx {list} -- List of indexes
    '''
    return [target[i] for i in idx]


def mk_batch(source, target, source_lang, target_lang):
    '''Make a batch

    Sort source and target sentences according to size
    Vectorize source and target sentences

    Arguments:
        source {numpy.ndarray} -- Source batch list of sentences
        target {list} -- Target batch list of sentences
        source_lang {Lang} -- Souce language dictionary
        target_lang {Lang} -- Target language dictionary

    Returns:
        tuple(list(int), numpy.ndarray, numpy.ndarray) -- Tuple consisting of

        1. List of sorted indices
        2. Vectorized source batch
        3. Vectorized target batch
    '''
    sorted_source = sorted(list(enumerate(source)),
                           key=lambda x: len(x[1]), reverse=True)
    idxs = [s[0] for s in sorted_source]
    source = [s[1] for s in sorted_source]
    target = mk_target(target, idxs)
    return idxs, \
        np.array([ mk_sentence(s, source_lang) for s in source ], dtype=np.float32), \
        np.array([mk_sentence(t, target_lang)
                  for t in target], dtype=np.float32)


def pack_batch(pairs, source_lang, target_lang, cuda_device=0):
    '''Pack a batch into vectorized form

    Arguments:
        pairs {list(tuple(int,int))} -- list of source-target sentence pairs
        source_lang {Lang} -- Souce language dictionary
        target_lang {Lang} -- Target language dictionary

    Returns:
        tuple(list(int), torch.Tensor, torch.Tensor, list(int), list(int)) -- tuple consisting:

        1. List of sorted indices
        2. Vectorized source batch
        3. Vectorized target batch
        4. Source batch lengths
        5. Target batch lengths
    '''
    # split the sentences
    source = [p[0].split(' ') for p in pairs]
    target = [p[1].split(' ') for p in pairs]
    # initial lengths
    source_lengths = [len(s) + 2 for s in source]
    target_lengths = [len(s) + 2 for s in target]
    # vectorize the batch
    idxs, source, target = mk_batch(source, target, source_lang, target_lang)
    # sorted lengths
    source_lengths = [source_lengths[i] for i in idxs]
    target_lengths = [target_lengths[i] for i in idxs]

    return (idxs,
        cudalong(source, gpu_id=cuda_device),
        cudalong(target, gpu_id=cuda_device),
        source_lengths,
        target_lengths
    )
