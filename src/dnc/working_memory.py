#!/usr/bin/env python3

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import numpy as np

from util import *
from dnc import *


class WorkingMemory(nn.Module):

  def __init__(self, mem_size=512, cell_size=32, read_heads=4, gpu_id=-1):
    super(WorkingMemory, self).__init__()

    self.mem_size = mem_size
    self.cell_size = cell_size
    self.read_heads = read_heads
    self.gpu_id = gpu_id

  def reset(self, batch_size=1, hidden=None):
    m = self.mem_size
    w = self.cell_size
    r = self.read_heads
    b = batch_size

    self.I = cuda(1 - T.eye(m).unsqueeze(0), gpu_id=self.gpu_id)  # (1 * n * n)

    if hidden is None:
      return {
          # linear memory (b * m * w)
          'memory': cuda(T.zeros(b, m, w).fill_(δ), gpu_id=self.gpu_id),
          # associative linkages (b * m * m)
          'temporal': cuda(T.zeros(b, m, m), gpu_id=self.gpu_id),
          # 'semantic' : cuda(T.zeros(b, m, m), gpu_id=self.gpu_id)
          'temporal_weights': cuda(T.zeros(b, m), gpu_id=self.gpu_id),
          # 'semantic_weights' : cuda(T.zeros(b, m), gpu_id=self.gpu_id)
          'read_weights': cuda(T.zeros(b, m, r).fill_(δ), gpu_id=self.gpu_id),
          'write_weights': cuda(T.zeros(b, m).fill_(δ), gpu_id=self.gpu_id),
          # 'read_vector' : cuda(T.zeros(b, w, r).fill_(δ), self.gpu_id=self.gpu_id)
          'usage_vector': cuda(T.zeros(b, m), gpu_id=self.gpu_id)
      }
    else:
      hidden['memory'].data.fill_(δ)
      hidden['temporal'].data.fill_(δ)
      # hidden['semantic'].data.fill_(δ)
      hidden['temporal_weights'].data.fill_(δ)
      # hidden['semantic_weights'].data.fill_(δ)
      hidden['read_weights'].data.fill_(δ)
      hidden['write_weights'].data.fill_(δ)
      # hidden['read_vector'].data.fill_(δ)
      hidden['usage_vector'].data.fill_(δ)
    return hidden

  def mem_usage(self, usage, free_gates, read_weights, write_weight):
    ψ = T.prod(1 - free_gates.unsqueeze(1).expand_as(read_weights) * read_weights, 2)
    return (usage + write_weight - usage * write_weight) * ψ

  def allocation(self, usage):
    # free list
    sorted_usage, φ = T.sort(usage, descending=False)
    # TODO: these are actually shifted cumprods
    sorted_allocation_weights = (1 - sorted_usage) * T.cumprod(sorted_usage, dim=1)
    # construct the reverse sorting index https://stackoverflow.com/questions/2483696/undo-or-reverse-argsort-python
    φ_rev = cuda(T.zeros(φ.size()), gpu_id=self.gpu_id)
    x = cuda(T.arange(0, φ_rev.size()[1], 1).unsqueeze(0).expand_as(φ), gpu_id=self.gpu_id)
    φ_rev.scatter_(1, φ, x)

    allocation_weights = sorted_allocation_weights.gather(1, φ_rev.long())
    return allocation_weights

  def write_weighting(self, memory, φ, allocation, write_gate, allocation_gate):
    ag = allocation_gate.expand_as(allocation)
    wg = write_gate.expand_as(allocation)
    return wg * (ag * allocation + (1 - ag) * φ.squeeze(2))

  def get_link_matrix(self, temporal, write_weight, temporal_weights):
    write_weight = write_weight.unsqueeze(1)  # only one head (b * 1 * m * m)
    temporal_weights = temporal_weights.unsqueeze(2)
    write_weights_i = write_weight.unsqueeze(3)
    write_weights_j = write_weight.unsqueeze(2)

    prev_scale = 1 - write_weights_i - write_weights_j
    new_temporal = write_weights_i * temporal_weights

    temporal = (prev_scale * temporal + new_temporal).squeeze(4).squeeze(1)
    # elaborate trick to delete diag elems
    return self.I.expand_as(temporal) * temporal

  def update_precedence(self, precedence, write_weight):
    return (1 - T.sum(write_weight, 1)).expand_as(precedence) * precedence + write_weight

  def write(self, write_key, write_vector, erase_vector, free_gates, read_strengths, write_strength, write_gate, allocation_gate, hidden):
    # lookup memeory with write_key
    φ = θ(hidden['memory'], write_key, dimB=2).squeeze(1)
    # get current usage
    hidden['usage_vector'] = self.mem_usage(hidden['usage_vector'], free_gates, hidden['read_weights'], write_strength)
    # get memory allocation
    alloc = self.allocation(hidden['usage_vector'])
    # get write weightings
    hidden['write_weights'] = \
        self.write_weighting(hidden['memory'], φ, alloc, write_gate, allocation_gate)

    write_weights = hidden['write_weights'].unsqueeze(2)
    write_vector = write_vector.unsqueeze(1)
    erase_vector = erase_vector.unsqueeze(1)

    # Update memory
    hidden['memory'] = hidden['memory'] * (1 - T.bmm(write_weights, erase_vector))
    hidden['memory'] = hidden['memory'] + T.bmm(write_weights, write_vector)

    # update link_matrix
    hidden['temporal'] = self.get_link_matrix(hidden['temporal'], write_weights, hidden['temporal_weights'])
    hidden['temporal_weights'] = self.update_precedence(hidden['temporal_weights'], write_weights)
    return hidden

  def content_weightings(self, memory, keys, read_strengths):
    d = θ(memory, keys, dimB=2)  # b * m * r
    return σ(d * read_strengths, 1)

  def directional_weightings(self, link_matrix, read_weights):
    f = T.bmm(link_matrix, read_weights)
    b = T.bmm(link_matrix.transpose(1, 2), read_weights)
    return f, b

  def read_weightings(self, memory, read_keys, read_weights, link_matrix, read_modes):
    c = self.content_weightings(memory, read_keys, read_weights)
    forward_weight, backward_weight = \
        self.directional_weightings(link_matrix, read_weights)

    content_mode = read_modes[:, 1, :].contiguous().unsqueeze(1).expand_as(c) * c
    backward_mode = read_modes[:, 0, :].contiguous().unsqueeze(1).expand_as(backward_weight) * backward_weight
    forward_mode = read_modes[:, 2, :].contiguous().unsqueeze(1).expand_as(forward_weight) * forward_weight

    return backward_mode + content_mode + forward_mode

  def read_vectors(self, memory, read_weights):
    return T.bmm(memory.transpose(1, 2), read_weights)

  def read(self, read_keys, read_strengths, read_modes, hidden):
    hidden['read_weights'] = self.read_weightings(
        hidden['memory'], read_keys, hidden['read_weights'], hidden['temporal'], read_modes
    )
    read_vectors = self.read_vectors(hidden['memory'], hidden['read_weights'])
    return read_vectors, hidden

  def forward(self, ξ, hidden):

    m = self.mem_size
    w = self.cell_size
    r = self.read_heads
    b = ξ.size()[0]

    # r read keys (b * r * w)
    read_keys = ξ[:, :r * w].contiguous().view(b, w, r)
    # r read strengths (b * r)
    read_strengths = oneplus(ξ[:, r * w:r * w + r].contiguous().view(b, r))
    # write key (b * w * 1)
    write_key = ξ[:, r * w + r:r * w + r + w].contiguous().view(b, w, 1)
    # write strength (b * 1)
    write_strength = oneplus(ξ[:, r * w + r + w]).unsqueeze(1)
    # erase vector (b * w)
    erase_vector = F.sigmoid(ξ[:, r * w + r + w + 1: r * w + r + 2 * w + 1])
    # write vector (b * w)
    write_vector = ξ[:, r * w + r + 2 * w + 1: r * w + r + 3 * w + 1]
    # r free gates (b * r)
    free_gates = F.sigmoid(ξ[:, r * w + r + 3 * w + 1: r * w + 2 * r + 3 * w + 1])
    # allocation gate (b * 1)
    allocation_gate = F.sigmoid(ξ[:, r * w + 2 * r + 3 * w + 1].contiguous().unsqueeze(1))
    # write gate (b * 1)
    write_gate = F.sigmoid(ξ[:, r * w + 2 * r + 3 * w + 2]).unsqueeze(1)
    # read modes (b * 3*r)
    read_modes = σ(ξ[:, r * w + 2 * r + 3 * w + 2: r * w + 5 * r + 3 * w + 2].contiguous().view(b, 3, r), 1)

    hidden = self.write(write_key, write_vector, erase_vector, free_gates,
                        read_strengths, write_strength, write_gate, allocation_gate, hidden)
    return self.read(read_keys, read_strengths, read_modes, hidden)
