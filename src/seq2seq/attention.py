#!/usr/bin/env python3

import torch.nn as nn
import torch as T
import torch.nn.functional as F
from torch.autograd import Variable as var
from torch.nn.utils.rnn import PackedSequence

import numpy as np

from util import *

from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack


class Attn(nn.Module):

    def __init__(self, method, hidden_size, max_length=MAX_LENGTH, gpu_id=-1):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.gpu_id = gpu_id

        if self.method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.other = nn.Parameter(T.zeros(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        lengths = None
        if type(encoder_outputs) is PackedSequence:
            encoder_outputs, lengths = pad(
                encoder_outputs, batch_first=True)
        else:
            lengths = [len(x) for x in encoder_outputs]

        batch_size = encoder_outputs.size()[0]
        attns = cuda(T.zeros(batch_size, max(lengths)), gpu_id=self.gpu_id)
        lengths = cuda(T.zeros(max(lengths), 1), gpu_id=self.gpu_id)

        if self.method == 'dot':
            attns = T.baddbmm(
                lengths,
                encoder_outputs,
                hidden.transpose(2, 1)
            ).squeeze(2)

        elif self.method == 'general':
            attended = self.attn(encoder_outputs)
            attns = T.baddbmm(
                lengths,
                attended,
                hidden.transpose(2, 1)
            ).squeeze(2)

        elif self.method == 'concat':
            concated = T.cat(
                (hidden.expand_as(encoder_outputs), encoder_outputs), 2)
            energy = self.attn(concated)
            expanded = self.other.unsqueeze(0).expand(
                batch_size, 1,
                self.hidden_size
            )
            attns = T.baddbmm(
                lengths,
                energy,
                expanded.transpose(2, 1)
            ).squeeze(2)

        return F.softmax(attns).unsqueeze(1)
