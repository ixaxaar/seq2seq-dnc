#!/usr/bin/env python3

import re
import string

from .special import *

import torch as T
import torch.nn.functional as F


def normalize(s):
  """Normalize a string

  Perform some basic operations to normalize a string, namely:
  1. Separate special characters with a space
  2. Lowercase
  3. Strip extra whitespace

  Arguments:
      s {string} -- Input string

  Returns:
      string -- Normalized string
  """
  s = s.lower()
  s = re.sub(r"([,.!?\(\);\"\'@#-])", r" \1 ", s)
  s = re.sub(r"\s+", r" ", s).strip()
  return s


def θ(a, b, dimA=2, dimB=2, normBy=2):
  """Batchwise Cosine distance

  Cosine distance

  Arguments:
      a {Tensor} -- A 3D Tensor (b * m * w)
      b {Tensor} -- A 3D Tensor (b * r * w)

  Keyword Arguments:
      dimA {number} -- exponent value of the norm for `a` (default: {2})
      dimB {number} -- exponent value of the norm for `b` (default: {1})

  Returns:
      Tensor -- Batchwise cosine distance (b * r * m)
  """
  a_norm = T.norm(a, normBy, dimA, keepdim=True).expand_as(a) + δ
  b_norm = T.norm(b, normBy, dimB, keepdim=True).expand_as(b) + δ

  print(a.size(), b.size())
  print('T.bmm(a, b.transpose(1, 2))', T.bmm(a, b.transpose(1, 2)))

  return T.bmm(a, b.transpose(1, 2)).transpose(1, 2) / (
      T.bmm(a_norm, b_norm.transpose(1, 2)).transpose(1, 2) + δ)


def σ(input, axis=1):
  """Softmax on an axis

  Softmax on an axis

  Arguments:
      input {Tensor} -- input Tensor

  Keyword Arguments:
      axis {number} -- axis on which to take softmax on (default: {1})

  Returns:
      Tensor -- Softmax output Tensor
  """
  input_size = input.size()

  trans_input = input.transpose(axis, len(input_size) - 1)
  trans_size = trans_input.size()

  input_2d = trans_input.contiguous().view(-1, trans_size[-1])
  soft_max_2d = F.softmax(input_2d)
  soft_max_nd = soft_max_2d.view(*trans_size)
  return soft_max_nd.transpose(axis, len(input_size) - 1)


def parse_bleu_output(bleu):
  b = re.findall('BLEU\ =\ [0-9\.]*', bleu)[0].replace('BLEU = ', '')
  return float(b)


def oneplus(x): return (1 + F.relu(x))
