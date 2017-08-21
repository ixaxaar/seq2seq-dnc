# #!/usr/bin/env python3

# import pytest
# import numpy as np

# import sys
# import os
# sys.path.append('./src/')

# from util import *
# from seq2seq import *


# def test_mse_loss()
# adversarial = [ n[u('de', 'je_nachdem')] ] * 9
# fake_predicted = [
#     adversarial,
#     vec('das ist falsch aber lasst uns sein echtes eh annehmen', 'de'),
#     vec('this is fake but lets assume its real eh'),
#     vec('this is fake but lets not assume its real'),
#     vec('this is fake but lets assume its much much much longer eh'),
#     vec('some short stuff'),
#     vec('rabble rabble rabble , but we dont know what else to do mayor , rabble rabble rabble')
# ]

# fake_target = [ vec('das ist falsch, aber lasst uns sein echtes eh annehmen', 'de') ] * len(fake_predicted)

# fake_batch = [ list(x) for x in zip(fake_predicted, fake_target) ]
# fake_i, fake_s, fake_t, fake_s_lens, fake_t_lens = pack_batch(fake_batch)
# fake_s, _ = pad_packed_sequence(fake_s, batch_first=True)
# fake_t, _ = pad_packed_sequence(fake_t, batch_first=True)

# for x in range(len(fake_s)):
#     print('Loss for sentence ' + str(fake_i[x]), criterion(fake_s[x].unsqueeze(0), fake_t[x].unsqueeze(0), [fake_s_lens[x]], [fake_t_lens[x]]).data[0] )
# print('Overall ', criterion(fake_s, fake_t, fake_s_lens, fake_t_lens).data[0])



# def test_mse_loss()
# adversarial = [ n[u('de', 'je_nachdem')] ] * 9
# fake_predicted = [
#     adversarial,
#     vec('das ist falsch aber lasst uns sein echtes eh annehmen', 'de'),
#     vec('this is fake but lets assume its real eh'),
#     vec('this is fake but lets assume its somewhat real eh'),
#     vec('this is fake but lets not assume its real'),
#     vec('this is fake but lets assume its much much much longer eh'),
#     vec('some short stuff'),
#     vec('rabble rabble rabble , but we dont know what else to do mayor , rabble rabble rabble')
# ]

# fake_target = [ vec('das ist falsch, aber lasst uns sein echtes eh annehmen', 'de') ] * len(fake_predicted)

# fake_batch = [ list(x) for x in zip(fake_predicted, fake_target) ]
# fake_i, fake_s, fake_t, fake_s_lens, fake_t_lens = pack_batch(fake_batch)
# fake_s, _ = pad_packed_sequence(fake_s, batch_first=True)
# fake_t, _ = pad_packed_sequence(fake_t, batch_first=True)

# for x in range(len(fake_s)):
#     print('Loss for sentence ' + str(fake_i[x]), criterion(fake_s[x].unsqueeze(0), fake_t[x].unsqueeze(0), [fake_s_lens[x]], [fake_t_lens[x]]).data[0] )
# print('Overall ', criterion(fake_s, fake_t, fake_s_lens, fake_t_lens).data[0])
