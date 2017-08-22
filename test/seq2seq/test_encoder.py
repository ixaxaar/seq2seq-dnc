#!/usr/bin/env python3

import pytest
import numpy as np

import sys
import os
sys.path.append('./src/')

from util import *
from seq2seq import *
from scripts.index_corpus import index_corpus

from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack


def test_encoder():
    index_corpus('en-test', './data/multi30k/train.en')
    index_corpus('de-test', './data/multi30k/train.de')
    en_lang = Lang('./en-test.lang')
    de_lang = Lang('./de-test.lang')
    en_lang.load('./en-test.lang')
    de_lang.load('./de-test.lang')

    x = ['lol that was very very very very very very very very funny',
         'sure it was', 'well indeed'] * 100
    y = ['lol das war sehr sehr sehr sehr sehr sehr sehr sehr lustig',
         'sicher war es', 'ja gut'] * 100
    idx, batch, tgt, slen, tlen = pack_batch(
        list(zip(x, y)), en_lang, de_lang, -1)

    source_vocab_size = en_lang.n_words
    target_vocab_size = de_lang.n_words
    e = Encoder(1024, 2, vocab_size=source_vocab_size)
    o, h = e(batch, slen)
    y, lengths = pad(o, batch_first=True)

    assert y.size() == T.Size([300, 14, 1024]), \
        'Outputs should have size nr_batches * max_len * nr_hidden'
    assert h.size() == T.Size([2, 300, 1024]), \
        'Hidden should have size  nr_layers * nr_batches * nr_hidden'


