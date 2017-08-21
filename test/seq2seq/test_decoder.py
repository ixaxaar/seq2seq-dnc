# # !/usr/bin/env python3

# import pytest
# import numpy as np

# import sys
# import os
# sys.path.append('./src/')

# from util import *
# from seq2seq import *
# from scripts.index_corpus import index_corpus

# from torch.nn.utils.rnn import pad_packed_sequence as pad
# from torch.nn.utils.rnn import pack_padded_sequence as pack


# def test_luong_decoder():
#     index_corpus('en-test', './data/multi30k/train.en')
#     index_corpus('de-test', './data/multi30k/train.de')
#     en_lang = Lang('./en-test.lang')
#     de_lang = Lang('./de-test.lang')
#     en_lang.load('./en-test.lang')
#     de_lang.load('./de-test.lang')

#     x = ['lol that was very very very very very very very very funny',
#          'sure it was', 'well indeed'] * 100
#     y = ['lol das war sehr sehr sehr sehr sehr sehr sehr sehr lustig',
#          'sicher war es', 'ja gut'] * 100
#     idx, batch, tgt, slen, tlen = pack_batch(
#         list(zip(x, y)), en_lang, de_lang, -1)

#     source_vocab_size = en_lang.n_words
#     target_vocab_size = de_lang.n_words

#     e = Encoder(1024, 2, vocab_size=source_vocab_size)
#     o, h = e(batch, slen)
#     y, lengths = pad(o, batch_first=True)

#     dec = LuongAttnDecoderRNN('general', 1024, 2, vocab_size=target_vocab_size)
#     print(dec)
#     hidden = h
#     # sentences = []
#     input = cudavec(np.array([SOS] * 300, dtype=np.long)).unsqueeze(1)
#     for x in range(input.size()[1]):
#         o, hidden, att = dec(input, y, hidden)
#         assert o.size() == T.Size([300,1,target_vocab_size]), \
#             'Output has size batch_size * 1 * target_vocab_size'
#         assert hidden.size() == T.Size([2,300,1024]), \
#             'Hidden has size nr_layers * batch_size * hidden_size'
#         assert att.size() == T.Size([300,1,14]), \
#             'Attention has size batch_size * 1 * max_len'
