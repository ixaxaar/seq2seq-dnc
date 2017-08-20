#!/usr/bin/env python3

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import numpy as np

from util import *

class AssociativeMemory(nn.Module):
    def __init__(self, size=512, cell_size=embedding_size, batch_size=batch_size, parallelism=4):
        super(AssociativeMemory, self).__init__()
        mem = {
            # linear memory size * cell_size (word vec / representation dims)
            # we take global softmax for content addressing
            'addressable': cuda(T.zeros(batch_size, size, cell_size).fill_(δ)),
            # associative linkages
            'temporal': cuda(T.zeros(batch_size, size, size)),
            'semantic': cuda(T.zeros(batch_size, size, size)),
            'weights': {
                'temporal': cuda(T.zeros(batch_size, size)), # precedence
                'semantic': cuda(T.zeros(batch_size, size)),
                'read': cuda(T.zeros(batch_size, size, parallelism).fill_(δ)),
                'write': cuda(T.zeros(batch_size, size).fill_(δ))
            },
            'vectors': {
                'read': cuda(T.zeros(batch_size, cell_size, parallelism).fill_(δ)),
                'usage': cuda(T.zeros(batch_size, size))
            }
        }
        self.get = mem

    def get_content(self, addressable, keys, strengths):
        φ = θ(addressable, keys)
        β = cuda(expand_dims(strengths, 1).expand_as(φ))
        return σ(φ * β, 1)

    def mem_usage(self, usage, free_gates, read_weights, write_weight):
        ψ = T.prod(1 - expand_dims(free_gates).expand_as(read_weights) * read_weights, 2)
        return (usage  + write_weight - usage * write_weight) * ψ

    def allocation(self, usage):
        sort_order = cuda(T.from_numpy(np.argsort(usage.cpu().data.numpy())), False)
        us = T.gather(usage, 1, sort_order).cpu().data.numpy()
        aas = []
        for batch in us:
            a = []; prev_u = [1]
            for u in batch:
                a.append((1-u) * reduce(lambda x,y: x*y, prev_u))
                prev_u.append(u)
            aas.append(a)
        return cuda(T.Tensor(aas))

    def write_weighting(self, addressable, allocation, write_key, write_strength, write_gate, allocation_gate):
        φ = θ(addressable, expand_dims(write_key, 2)).squeeze(2)
        c = σ(φ * write_strength.expand_as(φ), 1)
        ag = allocation_gate.expand_as(c)
        wg = write_gate.expand_as(c)
        return wg * ( ag * allocation + (1-ag)*c )

    def write(self, addressable, write_weight, write_vector, erase_vector):
        write_weight = expand_dims(write_weight, 2)
        write_vector = expand_dims(write_vector, 1)
        erase_vector = expand_dims(erase_vector, 1)

        erasing = addressable * (1 - torch.bmm(write_weight, erase_vector))
        writing = torch.bmm(write_weight, write_vector)
        updated_memory = erasing + writing

        return updated_memory

    def precedence(self, precedence, write_weight):
        return (1 - T.sum(write_weight, 1)).expand_as(precedence)*precedence + write_weight

    def link_matrix(self, temporal, write_weight, precedence):
        batches_of_w = write_weight.cpu().data.numpy()
        pairwise1 = [ (1-x-y) for x in w for y in w for w in batches_of_w ]
        pairwise2 = np.resize(np.array(pairwise1), (batch_size, size, size))
        with_diag = pairwise2 * temporal + T.bmm(expand_dims(write_weight, 2), expand_dims(precedence, 1))
        return (1 - T.eye(batch_size, size, size)) * with_diag

    def directional_weightings(self, temporal, read_weights):
        f = T.bmm(temporal, read_weights)
        b = T.bmm(temporal.transpose(1,2), read_weights)
        return f, b

    def read_weightings(self, addressable, read_keys, read_weights, link_matrix, read_modes):
        content_weightings = self.get_content(addressable, read_keys, read_weights)
        forward_weight, backward_weight = self.get_directional_weightings(read_weights, link_matrix)
        content_mode   = expand_dims(read_mode[:,1,:].contiguous(), 1).expand_as(content_weightings)  * content_weightings
        backward_mode = expand_dims(read_mode[:,0,:].contiguous(), 1).expand_as(backward_weight) * backward_weight
        forward_mode  = expand_dims(read_mode[:,2,:].contiguous(), 1).expand_as(forward_weight)  * forward_weight
        return backward_mode + lookup_mode + forward_mode

    def read_vectors(self, addressable, read_weightings):
        return torch.bmm(memory_matrix.transpose(1,2), read_weights)

