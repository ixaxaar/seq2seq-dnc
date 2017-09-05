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


def test_memory():
  ξ = var(T.randn(64, 32))
  # ξ = var(T.randn(64, (32 * 4) + (3 * 32) + (5 * 4) + 3))
  n = WorkingMemory(32)
  hx = n.reset(64)
  v, hx = n(ξ, hx)

  assert v.size() == T.Size([64, 32, 4])
  v1, hx = n(ξ, hx)

  assert v1.size() == T.Size([64, 32, 4])
