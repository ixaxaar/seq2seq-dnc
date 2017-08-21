#!/usr/bin/env python3

import numpy as np
import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import numpy as np

from util import *
from .attention import Attn
from .encoder import Encoder
from .luong_decoder import LuongAttnDecoderRNN

from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack


class LuongSeq2Seq(nn.Module):

    def __init__(
        self,
        src_lang,
        targ_lang,
        n_layers=4,
        hidden_size=1024,
        teacher_forcing_ratio=0.2,
        attention_type='general',
        cuda_device=-1
    ):
        super(LuongSeq2Seq, self).__init__()

        self.src_lang = src_lang
        self.targ_lang = targ_lang
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.attention_type = attention_type
        self.cuda_device = cuda_device

        self.encoder = Encoder(
            hidden_size,
            n_layers,
            vocab_size=src_lang.n_words
        )
        self.decoder = LuongAttnDecoderRNN(
            attention_type,
            hidden_size,
            n_layers,
            vocab_size=targ_lang.n_words
        )
        if cuda_device != -1:
            self.encoder.cuda(cuda_device)
            self.decoder.cuda(cuda_device)

    def _teacher_force(self):
        return np.random.choice([False, True], p=[1 - self.teacher_forcing_ratio, self.teacher_forcing_ratio])

    def forward(self, source, target, source_lengths, target_lengths):
        attentions = []
        encoded, hidden = self.encoder(source, source_lengths)
        hidden = hidden[:self.decoder.n_layers]
        batch_size = len(source)

        encoded, _ = pad(encoded, batch_first=True)
        outputs = cuda(
            T.zeros(batch_size, max(target_lengths), self.decoder.output_size),
            gpu_id=self.cuda_device
        )
        input = cudavec(
            np.array([SOS] * batch_size, dtype=np.long),
            gpu_id=self.cuda_device
        ).unsqueeze(1)

        # manually unrolled
        for x in range(max(target_lengths)):
            o, hidden, att = self.decoder(input, encoded, hidden)
            outputs[:, x, :] = o
            attentions.append(att.data.cpu().numpy())

            if self._teacher_force():
                input = target[:, x].unsqueeze(1).long()
            else:
                input = var(o.data.topk(1)[0].squeeze(1).long())

        return outputs, np.array(attentions)
