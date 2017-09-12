#!/usr/bin/env python3

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence

from util import *
from .memory import *


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
      nr_cells=5,
      read_heads=2,
      cell_size=10,
      nonlinearity='tanh',
      gpu_id=-1,
      clip=20,
      use_linear=True
  ):
    super(DNC, self).__init__()
    # todo: separate weights and RNNs for the interface and output vectors

    self.mode = mode
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.bias = bias
    self.batch_first = batch_first
    self.dropout = dropout
    self.bidirectional = bidirectional
    self.nr_cells = nr_cells
    self.read_heads = read_heads
    self.cell_size = cell_size
    self.nonlinearity = nonlinearity
    self.gpu_id = gpu_id
    self.clip = clip
    self.use_linear_memory_transforms = use_linear

    self.w = self.cell_size
    self.r = self.read_heads

    self.input_size = self.r * self.w + self.hidden_size
    self.interface_size = (self.w * self.r) + (3 * self.w) + (5 * self.r) + 3

    # whether to use linear transforms or asymmetric RNN
    if self.use_linear_memory_transforms:
      self.output_size = self.hidden_size
    else:
      self.output_size = self.hidden_size + self.interface_size

    self.rnns = []
    self.memories = []

    for layer in range(self.num_layers):
      # controllers for each layer
      if self.mode == 'RNN':
        self.rnns.append(nn.RNNCell(self.input_size, self.output_size, bias=self.bias, nonlinearity=self.nonlinearity))
      elif self.mode == 'GRU':
        self.rnns.append(nn.GRUCell(self.input_size, self.output_size, bias=self.bias))
      elif self.mode == 'LSTM':
        self.rnns.append(nn.LSTMCell(self.input_size, self.output_size, bias=self.bias))

      # memories for each layer
      self.memories.append(
          Memory(
              input_size=self.output_size,
              mem_size=self.nr_cells,
              cell_size=self.w,
              read_heads=self.r,
              gpu_id=self.gpu_id,
              use_linear=self.use_linear_memory_transforms
          )
      )

    for layer in range(self.num_layers):
      setattr(self, 'rnn_layer_' + str(layer), self.rnns[layer])
      setattr(self, 'rnn_layer_memory_' + str(layer), self.memories[layer])

    # final output layer
    self.mem_out = nn.Linear(self.input_size, self.hidden_size)

    if self.gpu_id != -1:
      [x.cuda(self.gpu_id) for x in self.rnns]
      [x.cuda(self.gpu_id) for x in self.memories]
      self.mem_out.cuda(self.gpu_id)

  def _init_hidden(self, hx, batch_size):
    # create empty hidden states if not provided
    if hx is None:
      hx = (None, None, None)
    (chx, mhx, last_read) = hx

    # initialize hidden state of the controller RNN
    if chx is None:
      chx = cuda(T.zeros(self.num_layers, batch_size, self.output_size), gpu_id=self.gpu_id)
      if self.mode == 'LSTM':
        chx = (chx, chx)

    # Last read vectors
    if last_read is None:
      last_read = cuda(T.zeros(batch_size, self.w * self.r), gpu_id=self.gpu_id)

    # memory states
    if mhx is None:
      mhx = [m.reset(batch_size) for m in self.memories]
    else:
      mhx = [m.reset(batch_size, h) for m, h in zip(self.memories, mhx)]

    return chx, mhx, last_read

  def _layer_forward(self, input, layer, hx=(None, None)):
    (chx, mhx) = hx
    max_length = len(input)
    outs = [0] * max_length
    read_vectors = [0] * max_length

    for time in range(max_length):
      # pass through controller
      chx = self.rnns[layer](input[time], chx)
      out = chx[0] if self.mode == 'LSTM' else chx

      # separate output and interface vectors
      if self.use_linear_memory_transforms:
        ξ = out
      else:
        ξ = out[:, :self.hidden_size]

      # pass through memory
      read_vecs, mhx = self.memories[layer](ξ, mhx)
      read_vectors[time] = read_vecs.view(-1, self.w * self.r)

      # get the final output for this time step
      outs[time] = self.mem_out(T.cat([out, read_vectors[time]], 1))
      # outs[time] = T.clamp(outs[time], -self.clip, self.clip)

    return outs, read_vectors, (chx, mhx)

  def forward(self, input, hx=(None, None, None)):
    # handle packed data
    is_packed = type(input) is PackedSequence
    if is_packed:
      input, lengths = pad(input)
      max_length = lengths[0]
    else:
      max_length = input.size(1) if self.batch_first else input.size(0)
      lengths = [input.size(1)] * max_length if self.batch_first else [input.size(0)] * max_length

    batch_size = input.size(0) if self.batch_first else input.size(1)

    # make the data batch-first
    if not self.batch_first:
      input = input.transpose(0, 1)

    controller_hidden, mem_hidden, last_read = self._init_hidden(hx, batch_size)

    # batched forward pass per element / word / etc
    outputs = None
    chxs = []
    read_vectors = [last_read] * max_length
    outs = [T.cat([input[:, x, :], last_read], 1) for x in range(max_length)]

    for layer in range(self.num_layers):
      # this layer's hidden states
      chx = [x[layer] for x in controller_hidden] if self.mode == 'LSTM' else controller_hidden[layer]
      # pass through controller
      outs, read_vectors, (chx, mem_hidden[layer]) = self._layer_forward(
          outs,
          layer,
          (chx, mem_hidden[layer])
      )
      chxs.append(chx)

      if layer == self.num_layers - 1:
        # final outputs
        outputs = T.stack(outs, 1)
      else:
        # the controller output + read vectors go into next layer
        outs = [T.cat([o, r], 1) for o, r in zip(outs, read_vectors)]

    # final hidden values
    if self.mode == 'LSTM':
      h = T.stack([x[0] for x in chxs], 0)
      c = T.stack([x[1] for x in chxs], 0)
      controller_hidden = (h, c)
    else:
      controller_hidden = T.stack(chxs, 0)

    if not self.batch_first:
      outputs = outputs.transpose(0, 1)
    if is_packed:
      outputs = pack(output, lengths)

    return outputs, (controller_hidden, mem_hidden, read_vectors[-1])
