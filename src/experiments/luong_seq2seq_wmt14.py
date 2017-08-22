#!/usr/bin/env python3

import numpy as np

import torch as T

from util import *
from seq2seq import *
from scripts.wmt import processWMT


def train_wmt_multimodal():
    where = '../data/multi30k'
    train = 'train'
    valid = 'val'
    test = 'test'
    src = 'en'
    targ = 'de'
    shard_size = 10000
    nr_shards = 29000 / shard_size

    epochs = 50
    gpu_id = 0

    s, t = processWMT(train, where, src, targ, shard_size=shard_size, gpu_id=gpu_id)
    processWMT(valid, where, src, targ, s, t, shard_size=shard_size, gpu_id=gpu_id)
    processWMT(test, where, src, targ, s, t, shard_size=shard_size, gpu_id=gpu_id)

    trainer = LuongSeq2SeqTrainer(
        where, src, targ,
        where+'/'+src+'.lang',
        where+'/'+targ+'.lang',
        4, 1024, 0.2, 'dot', 0.001, 10.0, gpu_id
    )

    # trainer = T.load(where+'/luong-seq2seq-epoch-12-dot-loss-2.7417213916778564.model')

    losses = []
    attns = []
    for epoch in range(1, epochs):
        for nr_shard in range(int(nr_shards)):
            log.info('Training epoch ' + str(epoch) +
                     ' shard ' + str(nr_shard))
            l, last_attention = trainer(nr_shard, batch_size=100)
            print(l)
            losses.append(l)
            attns.append(last_attention)

        T.save(trainer, where + '/' + 'luong-seq2seq-epoch-' + str(epoch) + '-' +
               'dot' + '-loss-' + str(trainer.last_loss) + '.model')
        T.save(trainer.state_dict(), where + '/' + 'luong-seq2seq-epoch-' + str(epoch) + '-' +
               'dot' + '-loss-' + str(trainer.last_loss) + '-state-dict.model')

if __name__ == '__main__':
    train_wmt_multimodal()
