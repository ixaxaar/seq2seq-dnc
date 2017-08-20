#!/usr/bin/env python3

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import numpy as np

def cuda(x, grad=False, gpu_id=-1):
  if gpu_id == -1:
    return var(x, requires_grad=grad)
  else:
    return var(x.pin_memory(), requires_grad=grad).cuda(gpu_id, async=True)

def cudavec(x, grad=False, gpu_id=-1):
  if gpu_id == -1:
    return var(T.from_numpy(x), requires_grad=grad)
  else:
    return var(T.from_numpy(x).pin_memory(), requires_grad=grad).cuda(gpu_id, async=True)

def cudalong(x, grad=False, gpu_id=-1):
  if gpu_id == -1:
    return var(T.from_numpy(x.astype(np.long)), requires_grad=grad)
  else:
    return var(T.from_numpy(x.astype(np.long)).pin_memory(), requires_grad=grad).cuda(gpu_id, async=True)
