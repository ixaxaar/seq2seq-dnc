#!/usr/bin/env python3

import pytest
import numpy as np

import sys
import os
sys.path.append('./src/')

from util import *
from nets import *
from scripts.wmt import processWMT


def test_seq2seqdnctrainer():
  where = './data/multi30k'
  which = 'train'
  src = 'en'
  targ = 'de'
  shard_size = 5120
  gpu_id = -1
  processWMT(which, where, src, targ, shard_size=shard_size, gpu_id=gpu_id)

  trainer = Seq2seqDNCTrainer(
      where, src, targ,
      where + '/' + src + '.lang',
      where + '/' + targ + '.lang',
      n_layers=2,
      hidden_size=256,
      teacher_forcing_ratio=0.2,
      attention_type='general',
      learning_rate=1.0,
      gradient_clip=10.0,
      gpu_id=-1,
      optimizer='sgd',
      bidirectional_encoder=True,
      bidirectional_decoder=False,
      mem_size=50,
      cell_size=256,
      read_heads=4,
      dropout=0.3
  )

  losses, last_attention = trainer(0, batch_size=128)

  print(losses)
  assert len(losses) == ((5120 / 128) + 1)
