#!/usr/bin/env python3

import numpy as np

from util import *
from seq2seq import *
from scripts.wmt import processWMT

def train_wmt_multimodal():
    where = '../data/multi30k'
    which = 'train'
    src = 'en'
    targ = 'de'
    shard_size = 10000
    vectorize_gpu = 0
    nr_shards = 29000/shard_size

    epochs = 13
    gpu_id = 0

    # processWMT(which, where, src, targ, shard_size, vectorize_gpu)

    trainer = LuongSeq2SeqTrainer(
        where, src, targ,
        where+'/'+src+'.lang',
        where+'/'+targ+'.lang',
        2, 1024, 0.2, 'dot', 0.001, 10.0, gpu_id
    )

    losses = []
    attns = []
    for epoch in range(epochs):
        for nr_shard in range(int(nr_shards)):
            l, last_attention = trainer(nr_shard, batch_size=128)
            print(l)
            losses.append(l)
            attns.append(last_attention)

if __name__ == '__main__':
    train_wmt_multimodal()

