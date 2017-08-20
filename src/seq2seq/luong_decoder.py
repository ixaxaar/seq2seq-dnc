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

class LuongAttnDecoderRNN(nn.Module):

    def __init__(self, attn_model, hidden_size, n_layers=1, dropout_p=0.1, vocab_size=50000):
        super(LuongAttnDecoderRNN, self).__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.n_layers = n_layers
        self.output_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, hidden_size, PAD)
        self.embedding_dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            dropout=dropout_p, batch_first=True
        )
        self.attn = Attn(attn_model, hidden_size)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, encoder_outputs, hidden=None):
        batch_size = input.size()[0]
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)

        dec_outs, hidden = self.gru(
            pack(embedded, [1] * batch_size, batch_first=True),
            hidden
        )

        # batch_size * 1 * hidden_size
        dec_outs, lengths = pad(dec_outs, batch_first=True)

        attn_weights = self.attn(dec_outs, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs)

        concated_input = T.cat((dec_outs, context), 2)
        concated_out = self.concat(concated_input.squeeze(1)).unsqueeze(1)
        concat_output = self.output(F.tanh(concated_out))

        return (concat_output, hidden, attn_weights)
