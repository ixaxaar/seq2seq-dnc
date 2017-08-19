#!/usr/bin/env python3

# dec = AttnDecoderRNN('general', 1024, 2, vocab_size=target_vocab_size).cuda(gpu_id)
# print(dec)
# hidden = h
# hidden = None
# sentences = []
# input = cudavec(np.array([SOS] * 300, dtype=np.long)).unsqueeze(1)
# for x in range(input.size()[1]):
#     o, hidden, att = dec(input, y, hidden)
#     print('Output has size batch_size * 1 * target_vocab_size', o.size() == T.Size([300,1,target_vocab_size]), '\n',
#           'Hidden has size nr_layers * batch_size * hidden_size', hidden.size() == T.Size([2,300,1024]), '\n',
#           'Attention has size batch_size * 1 * max_len', att.size() == T.Size([300,1,12]))
#     break
