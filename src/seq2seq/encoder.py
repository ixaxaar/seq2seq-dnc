#!/usr/bin/env python3

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack

from util import *


class Encoder(nn.Module):

  def __init__(self, hidden_size, n_layers=1, dropout_p=0.3, vocab_size=50000, bidirectional=False):
    super(Encoder, self).__init__()
    self.hidden_size = hidden_size
    self.n_layers = n_layers
    self.dropout = dropout_p
    self.vocab_size = vocab_size
    self.bidirectional = bidirectional

    self.embedding = nn.Embedding(vocab_size, hidden_size, PAD)
    self.rnn = nn.LSTM(
        self.hidden_size,
        self.hidden_size,
        num_layers=self.n_layers,
        dropout=self.dropout,
        batch_first=True,
        bidirectional=self.bidirectional
    )
    self.rnn.flatten_parameters()

  def forward(self, source, source_lengths, hidden=None):
    embedded = self.embedding(source)
    packed = pack(embedded, source_lengths, batch_first=True)
    outputs, hidden = self.rnn(packed, hidden)
    outputs, _ = pad(outputs, batch_first=True)

    # sum the bidirectional outputs
    if self.bidirectional:
      outputs = outputs[:, :, :self.hidden_size] + \
          outputs[:, :, self.hidden_size:]

    return outputs, hidden
