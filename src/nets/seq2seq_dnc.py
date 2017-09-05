#!/usr/bin/env python3

import numpy as np
import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import numpy as np

from util import *
from dnc import *
from seq2seq import *

from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack


class Seq2SeqDNC(nn.Module):

  def __init__(
      self,
      src_lang,
      targ_lang,
      n_layers=2,
      hidden_size=256,
      teacher_forcing_ratio=0.2,
      attention_type='general',
      gpu_id=-1,
      bidirectional_encoder=True,
      bidirectional_decoder=False,
      mem_size=50,
      cell_size=256,
      read_heads=4,
      dropout=0.3
  ):
    super(Seq2SeqDNC, self).__init__()
    self.src_lang = src_lang
    self.targ_lang = targ_lang
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.teacher_forcing_ratio = teacher_forcing_ratio
    self.attention_type = attention_type
    self.gpu_id = gpu_id
    self.bidirectional_encoder = bidirectional_encoder
    self.bidirectional_decoder = bidirectional_decoder
    self.mem_size = mem_size
    self.cell_size = cell_size
    self.read_heads = read_heads
    self.dropout = dropout

    self.encoder = Encoder(
        hidden_size=self.hidden_size,
        n_layers=self.n_layers,
        dropout_p=self.dropout,
        vocab_size=src_lang.n_words,
        bidirectional=self.bidirectional_encoder
    )
    self.memory = Memory(
        self.hidden_size,
        mem_size=self.mem_size,
        cell_size=self.cell_size,
        read_heads=self.read_heads,
        gpu_id=self.gpu_id
    )
    self.controller = MemoryController(
        hidden_size=self.hidden_size,
        memory=self.memory,
        encoder=self.encoder,
        n_layers=self.n_layers,
        dropout_p=self.dropout,
        bidirectional=False,  # todo: not yet supported
        gpu_id=self.gpu_id
    )
    self.decoder = LuongAttnDecoderRNN(
        attention_type,
        hidden_size,
        n_layers,
        vocab_size=targ_lang.n_words,
        gpu_id=self.gpu_id,
        dropout_p=self.dropout,
        bidirectional=self.bidirectional_decoder
    )

    if self.gpu_id != -1:
      self.encoder.cuda(self.gpu_id)
      self.decoder.cuda(self.gpu_id)
      self.memory.cuda(self.gpu_id)
      self.controller.cuda(self.gpu_id)

  def _teacher_force(self):
    return np.random.choice([False, True], p=[1 - self.teacher_forcing_ratio, self.teacher_forcing_ratio])

  def forward(self, source, target, source_lengths, target_lengths, hidden=None):
    attentions = []

    encoded, controller_hidden = self.controller(source, source_lengths, hidden)

    # the encoder LSTM's last hidden layer
    hidden = tuple([h[:self.decoder.n_layers] for h in controller_hidden[0]])
    batch_size = len(source)

    outputs = cuda(
        T.zeros(batch_size, max(target_lengths), self.decoder.output_size),
        gpu_id=self.gpu_id
    )
    input = cuda(T.LongTensor([SOS] * batch_size), gpu_id=self.gpu_id).unsqueeze(1)

    # manually unrolled
    for x in range(max(target_lengths)):
      o, hidden, att = self.decoder(input, encoded, hidden)
      outputs[:, x, :] = o
      # attentions.append(att.data.cpu().numpy())

      if self._teacher_force():
        input = target[:, x].unsqueeze(1).long()
      else:
        input = var(o.data.topk(1)[0].squeeze(1).long())

    return outputs, np.array(attentions), controller_hidden
