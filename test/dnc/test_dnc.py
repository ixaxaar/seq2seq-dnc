#!/usr/bin/env python3

import pytest
import numpy as np

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import numpy as np

import sys
import os
sys.path.append('./src/')

from util import *
from dnc import *


def test_dnc_lstm():
  input = var(T.randn(10, 50, 64))

  model = DNC('LSTM', 64, num_layers=1, mem_size=5,
              read_heads=2, gpu_id=-1)

  o, h = model(input)

  assert o.size() == input.size()


def test_dnc_gru():
  input = var(T.randn(10, 50, 64))
  model = DNC('GRU', 64, num_layers=1, mem_size=5,
              read_heads=2, gpu_id=-1)

  o, h = model(input)

  assert o.size() == input.size()


def test_dnc_rnn():
  input = var(T.randn(10, 50, 64))
  model = DNC('RNN', 64, num_layers=1, mem_size=5,
              read_heads=2, gpu_id=-1)

  o, h = model(input)

  assert o.size() == input.size()
