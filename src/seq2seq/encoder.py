#!/usr/bin/env python3

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack

from util import *


class Encoder(nn.Module):

    def __init__(self, hidden_size, n_layers=1, dropout_p=0.2, vocab_size=50000):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout_p
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, hidden_size, PAD)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=n_layers,
            dropout=dropout_p,
            batch_first=True
        )
        self.gru.flatten_parameters()

    def forward(self, source, source_lengths, hidden=None):
        embedded = self.embedding(source)
        packed = pack(embedded, source_lengths, batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        return outputs, hidden
