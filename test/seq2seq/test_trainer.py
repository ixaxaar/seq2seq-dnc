#!/usr/bin/env python3

import pytest
import numpy as np

import sys
import os
sys.path.append('./src/')

from util import *
from seq2seq import *
from scripts.wmt import processWMT

def test_luongseq2seqtrainer():
    where = './data/multi30k'
    which = 'train'
    src = 'en'
    targ = 'de'
    shard_size = 5120
    vectorize_gpu = -1
    processWMT(which, where, src, targ, shard_size, vectorize_gpu)

    trainer = LuongSeq2SeqTrainer(
        where, src, targ,
        where+'/'+src+'.lang',
        where+'/'+targ+'.lang',
        2, 1024, 0.2, 'dot', 0.001, 10.0, -1
    )

    losses, last_attention = trainer(0, batch_size=128)

    print(losses)
    assert len(losses) == ((512/128) + 1)



