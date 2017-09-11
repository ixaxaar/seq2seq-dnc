#!/usr/bin/env python3

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack

from util import *
from dnc import *


class DNCEncoder(nn.Module):

  def __init__(
      self,
      hidden_size,
      n_layers=1,
      dropout_p=0.3,
      vocab_size=50000,
      bidirectional=False,
      mem_size=5,
      read_heads=2,
      gpu_id=-1
  ):
    super(DNCEncoder, self).__init__()
    self.hidden_size = hidden_size
    self.n_layers = n_layers
    self.dropout = dropout_p
    self.vocab_size = vocab_size
    self.bidirectional = bidirectional
    self.gpu_id = gpu_id
    self.mem_size = mem_size
    self.read_heads = read_heads

    self.embedding = nn.Embedding(vocab_size, hidden_size, PAD)
    self.rnn = DNC(
        'LSTM',
        self.hidden_size,
        num_layers=self.n_layers,
        dropout=self.dropout,
        nr_cells=self.mem_size,
        read_heads=self.read_heads,
        batch_first=True,
        gpu_id=self.gpu_id,
        clip=self.vocab_size
    )

  def forward(self, source, source_lengths, hidden=None):
    embedded = self.embedding(source)
    if np.isnan(embedded.sum().cpu().data.numpy()[0]):
      raise Exception('nan detected in embedding')
    outputs, hidden = self.rnn(embedded, hidden)

    # sum the bidirectional outputs
    # if self.bidirectional:
    #   outputs = outputs[:, :, :self.hidden_size] + \
    #       outputs[:, :, self.hidden_size:]

    return outputs, hidden
