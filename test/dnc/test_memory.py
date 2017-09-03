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
  ξ = var(T.randn(64, (32 * 4) + (3 * 32) + (5 * 4) + 3))
  n = WorkingMemory()
  n.reset()
  v, _ = n(ξ)

  assert v.size() == T.Size([64, 32, 4])
  v1 = n(ξ)

  assert v1.size() == T.Size([64, 32, 4])
