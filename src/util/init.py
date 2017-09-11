#!/usr/bin/env python3

import numpy as np
import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
from torch.nn.init import xavier_uniform, xavier_normal, orthogonal, normal, uniform

import numpy as np

# currying functions are so fucking horrible in python
# fuck python and this typeless wilderness


def init(f):
  def _init(m):
    if isinstance(m, nn.Linear):
      f(m.weight.data)

    if isinstance(m, nn.Parameter):
      f(m.data)

  return _init
