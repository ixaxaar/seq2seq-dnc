#!/usr/bin/env python3

import numpy as np
import torch.nn as nn
import torch as T
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.autograd import Variable as var
import numpy as np

from util import *


class DecayingOptimizer:

  def __init__(self, parameters, optimizer, learning_rate, decay_kind=None, decay_param=None, γ=0.9):

    self.θ = parameters
    self.optimizer = optimizer
    self.learning_rate = learning_rate
    self.kind = decay_kind
    self.λ = decay_param
    self.γ = γ

    if self.optimizer == 'sgd':
      self.opt = optim.SGD(self.θ, lr=self.learning_rate)
    elif self.optimizer == 'adam':
      self.opt = optim.Adam(self.θ, lr=self.learning_rate)
    elif self.optimizer == 'adagrad':
      self.opt = optim.Adagrad(self.θ, lr=self.learning_rate)
    elif self.optimizer == 'adadelta':
      self.opt = optim.Adadelta(self.θ, lr=self.learning_rate)
    elif self.optimizer == 'rmsprop':
      self.opt = optim.RMSprop(self.θ, lr=self.learning_rate, eps=self.γ)

    if self.kind is None:
      self.scheduler = None
    elif self.kind == 'lambda':
      self.scheduler = LambdaLR(self.opt, self.λ)
    elif self.kind == 'step':
      self.scheduler = StepLR(self.opt, self.λ, gamma=self.γ)
    elif self.kind == 'plateau':
      self.scheduler = ReduceLROnPlateau(
          self.opt, 'max', patience=5, verbose=True, factor=self.λ)

  def step(self):
    self.opt.step()

  def zero_grad(self):
    self.opt.zero_grad()

  def decay(self, val_loss=None):
    if self.kind == 'plateau':
      self.scheduler.step(val_loss)
    elif self.scheduler:
      self.scheduler.step()
    else:
      pass
