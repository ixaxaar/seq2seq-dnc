#!/usr/bin/env python3


def test_encoder():
    assert 1 == 2

# en_lang = pickle.load(open('./mt/en-de/vectors/wmt-vectors-en-lang-train.dict', 'rb'))
# de_lang = pickle.load(open('./mt/en-de/vectors/wmt-vectors-de-lang-train.dict', 'rb'))
# w_s = en_lang.trim(30)
# w_t = de_lang.trim(30)
# source_vocab_size = len(w_s)
# target_vocab_size = len(w_t)
# e = EncoderRNN(1024, 2, vocab_size=source_vocab_size).cuda(gpu_id)
# print(e)
# x = ["lol that was very very very very very very very very funny", "sure it was", "well indeed"] * 100
# idx, batch, tgt, slen, tlen = pack_batch(list(zip(x, x)), en_lang, en_lang)
# o, h = e(batch, slen)
# y, lengths = pad_packed_sequence(o, batch_first=True)
# print('Outputs should have size nr_batches * max_len * nr_hidden: ', y.size() == T.Size([300, 12, 1024]))
# print('Hidden should have size  nr_layers * nr_batches * nr_hidden: ', h.size() == T.Size([2, 300, 1024]))

