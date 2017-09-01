# #!/usr/bin/env python3

# import torch.nn as nn
# import torch as T
# from torch.autograd import Variable as var
# import numpy as np
# import time

# from torch.nn.utils.rnn import pad_packed_sequence as pad
# from torch.nn.utils.rnn import pack_padded_sequence as pack

# from util import *


# class DNCParams(object):
#     def __init__(self):
#         self.use_cuda    = T.cuda.is_available()
#         self.dtype       = T.cuda.FloatTensor if T.cuda.is_available() else T.FloatTensor

#         self.batch_size     = None
#         self.input_dim      = None  # set after env
#         self.read_vec_dim   = None  # num_read_heads x mem_wid
#         self.output_dim     = None  # set after env

#         self.hidden_dim      = 64
#         self.num_write_heads = 1
#         self.num_read_heads  = 4
#         self.mem_hei         = 16
#         self.mem_wid         = 16
#         self.clip_value      = 20.

#         self.controller_params = ControllerParams()
#         self.accessor_params   = AccessorParams()


# class DNCEncoder(nn.Module):

#     def __init__(self,
#         hidden_size,
#         read_vec_dim,
#         output_dim,
#         n_layers=1,
#         dropout_p=0.3,
#         vocab_size=50000,
#         hidden_dim=64,
#         num_write_heads=1,
#         num_read_heads=4,
#         mem_hei=16,
#         mem_wid=16,
#         clip_value=20.
#     ):
#         super(DNCEncoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.n_layers = n_layers
#         self.dropout = dropout_p
#         self.vocab_size = vocab_size

#         self.embedding = nn.Embedding(vocab_size, hidden_size, PAD)

#         self.dnc_params = DNCParams()
#         self.dnc_params.batch_size     = batch_size
#         self.dnc_params.input_dim      = self.hidden_size
#         self.dnc_params.read_vec_dim   = read_vec_dim
#         self.dnc_params.output_dim     = output_dim
#         self.dnc_params.hidden_dim      = hidden_dim
#         self.dnc_params.num_write_heads = num_write_heads
#         self.dnc_params.num_read_heads  = num_read_heads
#         self.dnc_params.mem_hei         = mem_hei
#         self.dnc_params.mem_wid         = mem_wid
#         self.dnc_params.clip_value      = clip_value

#         self.rnn = DNCCircuit(self.dnc_params)
#         self.rnn.flatten_parameters()

#     def forward(self, source):
#         self.rnn._reset_states()
#         embedded = self.embedding(source)
#         outputs, hidden = self.rnn(embedded)
#         return outputs, hidden
