#!/usr/bin/env python3

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence

from util import *
from .working_memory import *


class DNC(nn.Module):

  def __init__(
      self,
      mode,
      hidden_size,
      num_layers=1,
      bias=True,
      batch_first=True,
      dropout=0,
      bidirectional=False,
      mem_size=5,
      read_heads=2,
      nonlinearity='tanh',
      gpu_id=-1
  ):
    super(DNC, self).__init__()
    # todo: separate weights and RNNs for the interface and output vectors

    self.cell_size = hidden_size

    self.mode = mode
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.bias = bias
    self.batch_first = batch_first
    self.dropout = dropout
    self.bidirectional = bidirectional
    self.mem_size = mem_size
    self.read_heads = read_heads
    self.nonlinearity = nonlinearity
    self.gpu_id = gpu_id

    self.w = self.cell_size
    self.r = self.read_heads

    self.input_size = (self.r + 1) * self.w
    # self.interface_size = (self.w * self.r) + (3 * self.w) + (5 * self.r) + 3
    self.output_size = self.hidden_size

    self.rnns = []
    self.memories = []

    for layer in range(self.num_layers):
      if self.mode == 'RNN':
        self.rnns.append(nn.RNNCell(self.input_size, self.output_size, bias=self.bias, nonlinearity=self.nonlinearity))
      elif self.mode == 'GRU':
        self.rnns.append(nn.GRUCell(self.input_size, self.output_size, bias=self.bias))
      elif self.mode == 'LSTM':
        self.rnns.append(nn.LSTMCell(self.input_size, self.output_size, bias=self.bias))

      self.memories.append(
          Memory(
              input_size=self.output_size,
              mem_size=self.mem_size,
              cell_size=self.w,
              read_heads=self.r,
              gpu_id=self.gpu_id
          )
      )

    self.mem_out = nn.Linear(self.input_size, self.output_size)

  def forward(self, input, hx=(None, None, None)):
    # handle packed data
    is_packed = type(input) is PackedSequence
    if is_packed:
      input, lengths = pad(input)
      max_length = lengths[0]
    else:
      max_length = input.size(1) if self.batch_first else input.size(0)
      lengths = [input.size(1)] * max_length if self.batch_first else [input.size(0)] * max_length

    nr_batches = input.size(0) if self.batch_first else input.size(1)

    # make the data batch-first
    if not self.batch_first:
      input = input.transpose(0, 1)

    # create empty hidden states if not provided
    if hx is None:
      hx = (None, None, None)
    (controller_hidden, mem_hidden, last_read) = hx

    # initialize hidden state of the controller RNN
    if controller_hidden is None:
      controller_hidden = var(input.data.new(
          nr_batches, self.output_size).zero_(), requires_grad=False)
      if self.mode == 'LSTM':
        controller_hidden = (controller_hidden, controller_hidden)

    # Last read vectors
    if last_read is None:
      last_read = var(input.data.new(nr_batches, self.w * self.r).zero_())

    # memory states
    if mem_hidden is None:
      mem_hidden = [m.reset(nr_batches) for m in self.memories]
    else:
      mem_hidden = [m.reset(nr_batches, h) for m, h in zip(self.memories, mem_hidden)]

    # batched forward pass per element / word / etc
    outputs = []
    for i in range(max_length):
      inp = T.cat([input[:, i, :], last_read], 1)

      for l in range(self.num_layers):
        # concat input and last read vectors
        controller_hidden = self.rnns[l](inp, controller_hidden)
        if self.mode == 'LSTM':
          (out, cells) = controller_hidden
        else:
          out = controller_hidden

        # pass through memory module
        read_vectors, mem_hidden[l] = self.memories[l](out, mem_hidden[l])
        last_read = read_vectors.view(-1, self.w * self.r)

        # get the final output
        mem_encoded = T.cat([out, last_read], 1)
        # output of this layer goes to next layer as input
        inp = mem_encoded

      outputs.append(self.mem_out(inp))
    outputs = T.cat([o.unsqueeze(1) for o in outputs], 1)

    if not self.batch_first:
      outputs = outputs.transpose(0, 1)
    if is_packed:
      outputs = pack(output, lengths)

    return outputs, (controller_hidden, mem_hidden, last_read)
