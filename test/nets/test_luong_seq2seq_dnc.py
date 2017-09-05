#!/usr/bin/env python3

import pytest
import numpy as np

import sys
import os
sys.path.append('./src/')

from util import *
from nets import *
from scripts.index_corpus import index_corpus

from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack


def test_luong_seq2seq_dnc():
  index_corpus('en-test', './data/multi30k/train.en')
  index_corpus('de-test', './data/multi30k/train.de')
  en_lang = Lang('./en-test.lang')
  de_lang = Lang('./de-test.lang')
  en_lang.load('./en-test.lang')
  de_lang.load('./de-test.lang')

  x = ['lol that was very very very very very very very very funny',
       'sure it was', 'well indeed'] * 10
  y = ['lol das war sehr sehr sehr sehr sehr sehr sehr sehr lustig',
       'sicher war es', 'ja gut'] * 10
  idx, src, targ, slen, tlen = pack_batch(
      list(zip(x, y)), en_lang, de_lang, -1)

  source_vocab_size = en_lang.n_words
  target_vocab_size = de_lang.n_words

  model = LuongSeq2SeqDNC(en_lang, de_lang, 4, 1024, 0.1, 'general', -1)
  o, a = model(src, targ, slen, tlen)

  assert o.size() == T.Size([30, 14, de_lang.n_words])
  # assert a.shape == (14, 30, 1, 14)
