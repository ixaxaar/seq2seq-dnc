#!/usr/bin/env python3

import pytest
import numpy as np

import sys
import os
sys.path.append('./src/')

from util import *
from nets import *
from scripts.wmt import processWMT


def test_luongseq2seqtrainer():
  where = './data/multi30k'
  which = 'train'
  src = 'en'
  targ = 'de'
  shard_size = 5120
  gpu_id = 0
  processWMT(which, where, src, targ, shard_size=shard_size, gpu_id=gpu_id)

  trainer = LuongSeq2SeqDNCTrainer(
      where, src, targ,
      where + '/' + src + '.lang',
      where + '/' + targ + '.lang',
      2, 256, 0.2, 'dot', 0.001, 10.0, gpu_id
  )

  losses, last_attention = trainer(0, batch_size=128)

  print(losses)
  assert len(losses) == ((5120 / 128) + 1)
