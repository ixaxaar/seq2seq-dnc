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


class MemoryController(nn.Module):

  def __init__(self,
               hidden_size,
               memory,
               encoder,
               n_layers=1,
               dropout_p=0.3,
               bidirectional=False,
               gpu_id=-1
               ):
    super(MemoryController, self).__init__()

    self.hidden_size = hidden_size
    self.memory = memory
    self.encoder = encoder
    self.n_layers = n_layers
    self.dropout = dropout_p
    self.gpu_id = gpu_id
    self.bidirectional = bidirectional

    self.w = self.memory.cell_size
    self.r = self.memory.read_heads

    assert self.hidden_size == self.w, 'RNN hidden size should match memory cell size'
    assert self.encoder.hidden_size == self.w, 'Encoder hidden size should match memory cell size'

    self.input_size = (self.r + 1) * self.w
    self.interface_size = (self.w * self.r) + (3 * self.w) + (5 * self.r) + 3
    self.output_size = self.w

    self.rnn = nn.LSTM(
        self.input_size,
        self.output_size,
        num_layers=self.n_layers,
        dropout=self.dropout,
        batch_first=True,
        bidirectional=self.bidirectional
    )
    self.mem_out = nn.Linear(self.input_size, self.output_size)
    self.rnn.flatten_parameters()

  def forward(self, input, source_lengths, hidden=(None, None, None)):
    batch_size = len(source_lengths)
    if not hidden:
      hidden = (None, None, None)
    (encoder_hidden, interface_hidden, mem_hidden) = hidden

    # encode
    encoded, encoder_hidden = self.encoder(input, source_lengths, encoder_hidden)

    # reset working memory
    mem_hidden = self.memory.reset(batch_size, mem_hidden)

    # nothing read in first time step (b*w*r)
    read_vectors = cuda(T.zeros(batch_size, self.w, self.r), gpu_id=self.gpu_id)
    dnc_encoded = cuda(T.zeros(encoded.size()), gpu_id=self.gpu_id)

    # unroll the rnn for each time step
    for x in range(max(source_lengths)):
      # concat the input and stuff read from memory in last time step
      b = encoded[:, x, :].unsqueeze(2)  # b * w * 1
      input = T.cat((b, read_vectors), 2).view(batch_size, -1, self.input_size)  # b * 1 * ((r+1)*w)

      # pass it through an RNN
      input = pack(input, [1] * batch_size, batch_first=True)

      out, interface_hidden = self.rnn(input, interface_hidden)
      out, _ = pad(out, batch_first=True)
      ξ = out.squeeze(1)

      # forward pass through memory
      read_vectors, mem_hidden = self.memory(ξ, mem_hidden)

      # final output, todo: differs from deepmind's implementation
      # where they concat and then pass through a Linear
      read_vecs = read_vectors.view(-1, self.w * self.r)
      mem_encoded = T.cat([ξ, read_vecs], 1)
      dnc_encoded[:, x, :] = self.mem_out(mem_encoded)

    return dnc_encoded, (encoder_hidden, interface_hidden, mem_hidden)
