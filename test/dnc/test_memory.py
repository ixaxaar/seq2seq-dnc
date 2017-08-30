#!/usr/bin/env python3

import pytest
import numpy as np

import sys
import os
sys.path.append('./src/')

from util import *
from dnc import *


def test_memory():
  m = WorkingMemory()

  b = m.batch_size
  w = m.w
  r = m.r

  ξ = T.randn(m.batch_size, m.w * m.r + 3 * m.w + 5 * m.r)

  # r read keys (b * r * w)
  read_keys = ξ[:, :r * w].contiguous().resize_(b, r, w)
  # r read strengths (b * r)
  read_strengths = oneplus(ξ[:, r * w:r * w + r].contiguous().resize_(b, r))
  # write key (b * w)
  write_key = ξ[:, r * w + r:r * w + r + w]
  # write strength (b * 1)
  write_strength = oneplus(ξ[:, r * w + r + w]).unsqueeze(1)
  # erase vector (b * w)
  erase_vector = F.sigmoid(ξ[:, r * w + r + w + 1: r * w + r + 2 * w + 1])
  # write vector (b * w)
  write_vector = ξ[:, r * w + r + 2 * w + 1: r * w + r + 3 * w + 1]
  # r free gates (b * r)
  free_gates = F.sigmoid(ξ[:, r * w + r + 3 * w + 1: r * w + 2 * r + 3 * w + 1])
  # allocation gate (b * 1)
  allocation_gate = F.sigmoid(ξ[:, r * w + 2 * r + 3 * w + 1]).unsqueeze(1)
  # write gate (b * 1)
  write_gate = F.sigmoid(ξ[:, r * w + 2 * r + 3 * w + 2]).unsqueeze(1)
  # read modes (b * 3*r)
  read_modes = σ(ξ[:, r * w + 2 * r + 3 * w + 2:], 1)
