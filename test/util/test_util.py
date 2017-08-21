# #!/usr/bin/env python3

# import pytest
# import numpy as np

# import sys
# import os
# sys.path.append('./src/')

# from util import *
# from scripts.index_corpus import index_corpus


# def test_indexing():
#     index_corpus('en-test', './data/multi30k/train.en')
#     en_lang = Lang('en-test')
#     en_lang.load('./en-test.lang')

#     assert os.path.isfile('en-test.lang'), \
#         'Should create a dictionary for the ./data/multi30k/train.en corpus'

#     assert en_lang.n_words == 9792


# def test_pack_batch():
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

#     assert len(idx) == len(x)
#     assert len(slen) == len(x)
#     assert len(tlen) == len(x)
#     assert type(batch) is var
#     assert batch.size() == T.Size([300, 100])

#     # w_s = en_lang.trim(3)
#     # w_t = de_lang.trim(3)
#     # assert
