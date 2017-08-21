#!/usr/bin/env python3

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import numpy as np

from masked_cross_entropy import masked_cross_entropy
from util import *


class MSELoss(nn.Module):

    def __init__(self, size_average=True, gpu_id=-1):
        super(MSELoss, self).__init__()

        self.gpu_id = gpu_id
        self.size_average = size_average
        self.loss_fn = nn.MSELoss(size_average=size_average)

    def forward(self, predicted, target, predicted_lens, target_lens, batch_size=None):
        if not batch_size:
            batch_size = predicted.size()[0]
        loss = 0.0
        for i, t in enumerate(target_lens):
            target[i, t:, :].data.fill_(0)
        p = T.sum(predicted, 1)
        t = T.sum(target, 1)

        return self.loss_fn(p, t)


class CosineLoss(nn.Module):

    def __init__(self, margin=0.2, size_average=True, gpu_id=-1):
        super(CosineLoss, self).__init__()

        self.margin = margin
        self.size_average = size_average
        self.gpu_id = gpu_id
        self.loss_fn = T.nn.CosineEmbeddingLoss(
            margin=margin, size_average=size_average)

    def forward(self, predicted, target, predicted_lens, target_lens, batch_size=None):
        if not batch_size:
            batch_size = predicted.size()[0]
        loss = 0.0
        for i, t in enumerate(target_lens):
            target[i, t:, :].data.fill_(0)
        p = T.sum(predicted, 1)
        t = T.sum(target, 1)
        return loss_fn(p, t, cuda(T.ones(batch_size), gpu_id=self.gpu_id))


class MaskedCrossEntropy(nn.Module):

    def __init__(self, gpu_id=-1):
        super(MaskedCrossEntropy, self).__init__()

        self.gpu_id = gpu_id

    def forward(self, logits, target, lengths):
        lengths = cuda(T.LongTensor(lengths), gpu_id=self.gpu_id)
        return masked_cross_entropy(logits, target, lengths, self.gpu_id)
