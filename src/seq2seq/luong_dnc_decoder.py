#!/usr/bin/env python3

import torch.nn as nn
import torch as T
import torch.nn.functional as F
from torch.autograd import Variable as var
from torch.nn.utils.rnn import PackedSequence

import numpy as np

from util import *
from .attention import Attn

from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack

from util import *
from dnc import *


class LuongAttnDecoderDNC(nn.Module):

  def __init__(
      self,
      attn_model,
      hidden_size,
      n_layers=1,
      dropout_p=0.3,
      vocab_size=50000,
      bidirectional=False,
      mem_size=5,
      cell_size=10,
      read_heads=2,
      gpu_id=-1
  ):
    super(LuongAttnDecoderDNC, self).__init__()
    self.attn_model = attn_model
    self.hidden_size = hidden_size
    self.dropout = dropout_p
    self.n_layers = n_layers
    self.output_size = vocab_size
    self.gpu_id = gpu_id
    self.bidirectional = bidirectional
    self.mem_size = mem_size
    self.read_heads = read_heads
    self.cell_size = cell_size

    self.embedding = nn.Embedding(vocab_size, hidden_size, PAD)
    self.embedding_dropout = nn.Dropout(dropout_p)
    self.rnn = DNC(
        'LSTM',
        self.hidden_size,
        num_layers=self.n_layers,
        dropout=self.dropout,
        nr_cells=self.mem_size,
        cell_size=self.cell_size,
        read_heads=self.read_heads,
        batch_first=True,
        gpu_id=self.gpu_id,
        clip=vocab_size
    )

    self.attn = Attn(attn_model, hidden_size, gpu_id=self.gpu_id)
    self.concat = nn.Linear(hidden_size * 2, hidden_size)
    self.output = nn.Linear(hidden_size, vocab_size)

  def forward(self, input, encoder_outputs, hidden=None):
    batch_size = input.size()[0]
    embedded = self.embedding(input)
    if np.isnan(embedded.sum().cpu().data.numpy()[0]):
      raise Exception('nan detected in embedding')
    embedded = self.embedding_dropout(embedded)

    # batch_size * 1 * hidden_size
    dec_outs, hidden = self.rnn(embedded, hidden)

    # sum the bidirectional outputs
    if self.bidirectional:
      dec_outs = dec_outs[:, :, :self.hidden_size] + \
          dec_outs[:, :, self.hidden_size:]

    # calculate the attention context vector
    attn_weights = self.attn(dec_outs, encoder_outputs)
    context = attn_weights.bmm(encoder_outputs)

    # Finally concatenate and pass through a linear layer
    concated_input = T.cat((dec_outs, context), 2)
    concated_out = self.concat(concated_input.squeeze(1)).unsqueeze(1)

    concat_output = self.output(F.tanh(dec_outs))

    return (concat_output, hidden, attn_weights)
