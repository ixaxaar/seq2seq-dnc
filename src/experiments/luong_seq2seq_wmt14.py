#!/usr/bin/env python3

import numpy as np

import torch as T
from random import shuffle

from util import *
from nets import *
from seq2seq import *
from scripts.wmt import processWMT

import time


def train_wmt_multimodal():
  where = '../data/multi30k'
  train = 'train'
  valid = 'val'
  test = 'test'
  src = 'en'
  targ = 'de'
  shard_size = 5000
  nr_shards = math.ceil(29000 / shard_size)
  n_layers = 2
  n_hidden = 500
  attention_type = 'general'
  learning_rate = 1.0
  clip = 5.0
  teacher_forcing_ratio = 0.0
  batch_size = 128
  optim = 'sgd'
  bidirectional_encoder = True
  bidirectional_decoder = False

  epochs = 50
  gpu_id = 0

  # s = Lang('en')
  # t = Lang('de')
  # s.load('../data/multi30k/en.lang')
  # t.load('../data/multi30k/de.lang')

  s, t = processWMT(train, where, src, targ,
                    shard_size=shard_size, gpu_id=gpu_id)
  processWMT(valid, where, src, targ, s, t,
             shard_size=1000, gpu_id=gpu_id)
  processWMT(test, where, src, targ, s, t,
             shard_size=1000, gpu_id=gpu_id)
  # processWMT('supersmall', where, src, targ, s, t, shard_size=10, gpu_id=gpu_id)

  trainer = LuongSeq2SeqTrainer(
      where, src, targ,
      s, t,
      n_layers, n_hidden, teacher_forcing_ratio, attention_type,
      learning_rate, clip, gpu_id, optim,
      bidirectional_encoder=bidirectional_encoder,
      bidirectional_decoder=bidirectional_decoder
  )

  # trainer = T.load(where+'/50epochs/luong-seq2seq-epoch-49-dot-loss-0.20151005685329437.model')
  # trainer.model.encoder.gru.flatten_parameters()
  # trainer.model.decoder.gru.flatten_parameters()

  # trainer.evaluate(where, valid, save=True)
  # return

  # print(trainer.targ_lang.index2word[6352], 'glatzenansatz')
  # trainer.evaluate(where, valid, save=True)
  # trainer.evaluate(where, test, save=True)
  # trainer.evaluate(where, 'supersmall', save=True, shard_size=10, batch_size=10)

  losses = []
  attns = []
  bleus = []
  for epoch in range(1, epochs):
    log.info('=====================================================')
    log.info('Epoch ' + str(epoch))
    log.info('=====================================================')

    # shuffle shard order
    shuffled_shards = shuffle(list(range(int(nr_shards))))
    for nr_shard in shuffled_shards:

      log.info('Training epoch ' + str(epoch) +
               ' shard ' + str(nr_shard))

      l, last_attention = trainer(nr_shard, batch_size=batch_size)
      trainer.evaluate(where, valid, save=True)
      bl = bleu(where,
                valid + '-predicted.txt',
                valid + '-reference.txt')

      log.info(str(bl))
      log.info('Total loss: ' + str(sum(l)))

      bleus.append(parse_bleu_output(bl))
      losses.append(l)
      attns.append(last_attention)

    try:
      T.save(trainer, where + '/' + 'luong-seq2seq-epoch-' + str(epoch) + '-' +
             'dot' + '-loss-' + str(trainer.last_loss) + '.model')
      T.save(trainer.state_dict(), where + '/' + 'luong-seq2seq-epoch-' + str(epoch) + '-' +
             'dot' + '-loss-' + str(trainer.last_loss) + '-state-dict.model')
    except Exception as e:
      print(e)

    trainer.learning_rate_decay(sum(bleus) / len(bleus))

if __name__ == '__main__':
  train_wmt_multimodal()
