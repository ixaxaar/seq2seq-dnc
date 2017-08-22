#!/usr/bin/env python3

import math, re

import numpy as np
import torch.nn as nn
import torch as T
import torch.optim as optim
from torch.autograd import Variable as var
import numpy as np

from util import *
from .luong_seq2seq import LuongSeq2Seq

from loss import MaskedCrossEntropy


class LuongSeq2SeqTrainer(nn.Module):

    def __init__(
        self,
        data_path,
        src,
        targ,
        src_lang_path,
        targ_lang_path,
        n_layers=4,
        hidden_size=1024,
        teacher_forcing_ratio=0.2,
        attention_type='general',
        learning_rate=0.001,
        gradient_clip=10.0,
        gpu_id=-1
    ):
        super(LuongSeq2SeqTrainer, self).__init__()

        self.path = data_path
        self.src = src
        self.targ = targ
        self.src_lang = self._load_lang('src', src_lang_path)
        self.targ_lang = self._load_lang('targ', targ_lang_path)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.attention_type = attention_type
        self.gpu_id = gpu_id
        self.learning_rate = learning_rate
        self.gradient_clip = gradient_clip
        self.last_loss = 0

        self.model = LuongSeq2Seq(self.src_lang, self.targ_lang, self.n_layers, self.hidden_size,
                                  self.teacher_forcing_ratio, self.attention_type, self.gpu_id)

        self.loss = MaskedCrossEntropy(self.gpu_id)
        self.encoder_optimizer = \
            optim.Adam(self.model.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = \
            optim.Adam(self.model.decoder.parameters(), lr=learning_rate)

    def _load_shard(self, nr_shard, which='train'):
        with open(self.path + '/sentence-pairs-' + which + '-' + self.src + '-' + self.targ + '-shard-' + str(nr_shard) + '.t7', 'rb') as pairs:
            return T.load(pairs)

    def _load_lang(self, name, path):
        l = Lang(name)
        l.load(path)
        return l

    def evaluate(self, where, which, batch_size=64, shard_size=1000, save=False):
        shard = self._load_shard(0, which)
        indexes = shard['indexes']
        source = shard['source']
        target = shard['target']
        source_lengths = shard['source_lengths']
        target_lengths = shard['target_lengths']

        self.model.eval()

        batches = math.ceil(shard_size / batch_size)
        predicted_sentences = []

        for b in range(batches):
            log.debug('Evalutaing ' + which + ' batch ' + str(b))
            # try:
            # prepare batch
            sort_order = indexes[b:b + batch_size]
            s_packed = source[b:b + batch_size]
            t_packed = target[b:b + batch_size]
            s_lens = source_lengths[b:b + batch_size]
            t_lens = target_lengths[b:b + batch_size]

            predicted, attn = self.model(
                s_packed, t_packed, s_lens, t_lens)

            # scores
            _, predicted = predicted.topk(1)
            predicted = predicted.data.cpu().numpy()

            # Top 1 prediction
            for x in range(predicted.shape[0]):
                s = []
                for y in predicted[x]:
                    s.append( self.targ_lang.index2word[y[0]] )
                predicted_sentences.append(' '.join(s))

        predicted_sentences = [ re.sub('\s+', ' ', p.replace('SOS', ' ').replace('EOS', ' ')) for p in predicted_sentences ]

        # Remove special tokens and write into a file
        if save:
            with open(where+'/'+which+'-predicted.txt', 'w') as out:
                for p in predicted_sentences:
                    out.write(p)
                    out.write('\n')

        self.model.train()
        return predicted_sentences

    def forward(self, nr_shard, batch_size=64):
        losses = [0.0]
        last_attn = None

        # load the shard
        shard = self._load_shard(nr_shard)
        indexes = shard['indexes']
        source = shard['source']
        target = shard['target']
        source_lengths = shard['source_lengths']
        target_lengths = shard['target_lengths']

        batches = math.ceil(len(indexes) / batch_size)
        for b in range(batches):
            log.debug('Training batch ' + str(b))
            # try:
            # prepare batch
            sort_order = indexes[b:b + batch_size]
            s_packed = source[b:b + batch_size]
            t_packed = target[b:b + batch_size]
            s_lens = source_lengths[b:b + batch_size]
            t_lens = target_lengths[b:b + batch_size]
            # reset gradients
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            # forward pass
            last_attn = None
            predicted, last_attn = self.model(
                s_packed, t_packed, s_lens, t_lens)
            # evaluate
            loss = self.loss(
                predicted.contiguous(),
                t_packed[:, :predicted.size()[1]].contiguous(),
                t_lens
            )
            # T.cuda.synchronize()
            # backpropagate
            loss.backward()
            # clip gradient norms
            nn.utils.clip_grad_norm(
                self.model.encoder.parameters(), self.gradient_clip)
            nn.utils.clip_grad_norm(
                self.model.decoder.parameters(), self.gradient_clip)
            # update parameters
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            losses.append(float(loss.data.cpu().numpy()[0]))
            self.last_loss = losses[len(losses) - 1]
            # except Exception as e:
            #     print('Exception occured')
            #     print(e)
            #     pass

        return losses, last_attn
