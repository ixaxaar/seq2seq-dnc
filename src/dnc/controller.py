#!/usr/bin/env python3

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence

from util import *
from seq2seq import Encoder


class LSTM_DNC(nn.Module):

  def __init__(self, hidden_size, memory, encoder, n_layers=1, dropout_p=0.3, bidirectional=False, gpu_id=-1):
    super(LSTMController, self).__init__()

    self.hidden_size = hidden_size
    self.memory = memory
    self.encoder = encoder
    self.n_layers = n_layers
    self.dropout = dropout_p
    self.gpu_id = gpu_id

    self.input_size = (self.memory.r * self.memory.w) + \
        self.hidden_size
    self.interface_size = (self.memory.w * self.memory.r) + \
        (3 * self.memory.w) + (5 * self.memory.r) + 3

    self.rnn = nn.LSTM(
        self.input_size,
        self.interface_size,
        num_layers=self.n_layers,
        dropout=self.dropout_p,
        batch_first=True,
        bidirectional=self.bidirectional
    )
    self.rnn.flatten_parameters()

  # input: b * s * w
  # read_vectors: b * s * (w*r)
  def forward(self, input, hidden):
    batch_size = input.size()[0]

    (encoder_hidden, interface_hidden, memory_hidden) = hidden
    encoded, encoder_hidden = self.encoder(input, encoder_hidden)

    # nothing read in first time step (b * 1 * (w*r))
    read_vectors = cuda(T.zeros(batch_size, 1, self.memory.r * self.memory.w), gpu_id=gpu_id)
    dnc_encoded = cuda(T.zeros(encoded.size()), gpu_id=gpu_id)

    # unroll the rnn for each time step
    for x in range(encoded.size()[1]):
      b = encoded[:, x, :].unsqueeze(1)
      input = T.cat(b, read_vectors, 2)
      out, interface_hidden = self.rnn(input, interface_hidden)
      dnc_encoded[:, x, :] = out[:, :, :self.hidden_size]
      interface_vector = out[:, :, self.hidden_size:]
      read_vectors, memory_hidden = self.memory(interface_vector, memory_hidden)

    return dnc_encoded, (encoder_hidden, interface_hidden, memory_hidden)
