#!/usr/bin/env python3

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack

from util import *


class ColumnarEncoder(nn.Module):

    def __init__(self, hidden_size, n_columns=2, n_layers=1, dropout_p=0.2, vocab_size=50000):
        super(ColumnarEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout_p
        self.vocab_size = vocab_size
        self.n_columns = n_columns

        self.λ = hidden_size / n_columns # column size
        self.embedding = nn.Embedding(vocab_size, hidden_size, PAD)
        self.columns = []
        for c in range(n_columns):
            gru = nn.GRU(
                self.λ,
                self.λ,
                num_layers=n_layers,
                dropout=dropout_p,
                batch_first=True
            )
            self.columns.append(gru)

    def forward(self, source, source_lengths, hidden=None):
        '''
        source: nr_batches * max_len
        source_lengths: nr_batches
        hidden: nr_layers * nr_batches * nr_hidden
        '''
        batch_size = self.source.size()[0]
        embedded = self.embedding(source)
        outputs = T.zeros(batch_size, max(source_lengths), self.nr_hidden)

        for c in range(self.columns):
            e = embedded[ :, :, self.λ*c:(self.λ+1)*c ]
            packed = pack(e, source_lengths, batch_first=True)
            h = None if not hidden else hidden[:, :, self.λ*c:(self.λ+1)*c]
            o, h = self.gru(packed, h)
            o, _ = pad(packed, batch_first=True)
            hidden[:, :, self.λ*c:(self.λ+1)*c] = h
            outputs[:, :, self.λ*c:(self.λ+1)*c] = o

        return outputs, hidden
