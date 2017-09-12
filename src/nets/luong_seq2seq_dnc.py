#!/usr/bin/env python3

import numpy as np
import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import numpy as np

from util import *
from seq2seq import *

from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack


class LuongSeq2SeqDNC(nn.Module):

  def __init__(
      self,
      src_lang,
      targ_lang,
      n_layers=4,
      hidden_size=1024,
      teacher_forcing_ratio=0.2,
      attention_type='general',
      gpu_id=-1,
      bidirectional_encoder=True,
      bidirectional_decoder=False,
      mem_size=5,
      cell_size=10,
      read_heads=2,
      encoder_type='dnc',
      decoder_type='lstm'
  ):
    super(LuongSeq2SeqDNC, self).__init__()

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
    self.encoder_type = encoder_type
    self.decoder_type = decoder_type

    if self.encoder_type == 'dnc':
      self.encoder = DNCEncoder(
          hidden_size,
          n_layers,
          vocab_size=src_lang.n_words,
          bidirectional=self.bidirectional_encoder,
          mem_size=self.mem_size,
          cell_size=self.cell_size,
          read_heads=self.read_heads,
          gpu_id=self.gpu_id
      )
    else:
      self.encoder = Encoder(
          hidden_size,
          n_layers,
          vocab_size=src_lang.n_words,
          bidirectional=self.bidirectional_encoder
      )
    if self.decoder_type == 'dnc':
      self.decoder = LuongAttnDecoderDNC(
          attention_type,
          hidden_size,
          n_layers,
          vocab_size=targ_lang.n_words,
          gpu_id=gpu_id,
          bidirectional=self.bidirectional_decoder,
          mem_size=self.mem_size,
          cell_size=self.cell_size,
          read_heads=self.read_heads
      )
    else:
      self.decoder = LuongAttnDecoderRNN(
          attention_type,
          hidden_size,
          n_layers,
          vocab_size=targ_lang.n_words,
          gpu_id=gpu_id,
          bidirectional=self.bidirectional_decoder
      )
    if gpu_id != -1:
      self.encoder.cuda(gpu_id)
      self.decoder.cuda(gpu_id)

  def _teacher_force(self):
    return np.random.choice([False, True], p=[1 - self.teacher_forcing_ratio, self.teacher_forcing_ratio])

  def forward(self, source, target, source_lengths, target_lengths):
    attentions = []
    # pass through the encoder and get hidden state of its last layer
    if self.encoder_type == 'dnc':
      encoded, (controller_hidden, mem_hidden, last_read) = self.encoder(source, source_lengths)
      if self.decoder_type == 'dnc':
        # hidden = (controller_hidden, mem_hidden, last_read)
        hidden = None
      else:
        hidden = controller_hidden
    else:
      encoded, controller_hidden = self.encoder(source, source_lengths)
      if self.decoder_type == 'dnc':
        hidden = (controller_hidden, None, None)
      elif self.decoder_type == self.encoder_type:
        hidden = tuple([h[:self.decoder.n_layers] for h in controller_hidden])

    # uncomment for dummy encoder input
    # encoded = cuda(T.zeros(source.size(0), source.size(1), self.hidden_size), gpu_id=self.gpu_id)
    batch_size = len(source)

    outputs = cuda(
        T.zeros(batch_size, max(target_lengths), self.decoder.output_size),
        gpu_id=self.gpu_id
    )
    # todo: use tensor instead of numpy
    input = cudavec(
        np.array([SOS] * batch_size, dtype=np.long),
        gpu_id=self.gpu_id
    ).unsqueeze(1)

    # manually unrolled
    for x in range(max(target_lengths)):
      o, hidden, att = self.decoder(input, encoded, hidden)
      outputs[:, x, :] = o
      # attentions.append(att.data.cpu().numpy())

      # if self._teacher_force():
      #   input = target[:, x].unsqueeze(1).long()
      # else:
      input = var(o.data.topk(1)[0].squeeze(1).long())

    return outputs, np.array(attentions)
