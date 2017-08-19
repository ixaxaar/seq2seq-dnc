#!/usr/bin/env python3

import re
import string

import torch as T


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


def θ(a, b, normByA=2, normByB=1):
    """Batchwise Cosine distance

    Cosine distance

    Arguments:
        a {Tensor} -- A 3D Tensor
        b {Tensor} -- A 3D Tensor

    Keyword Arguments:
        normByA {number} -- exponent value of the norm for `a` (default: {2})
        normByB {number} -- exponent value of the norm for `b` (default: {1})

    Returns:
        Tensor -- Batchwise cosine distance
    """
    return T.bmm(
        T.div(a, T.norm(a, 2, normByA).expand_as(a) + δ),
        T.div(b, T.norm(b, 2, normByB).expand_as(b) + δ)
    )


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


def expand_dims(input, axis=0):
    """Add a dimention along given axis

    TODO: is this replaceable with unsqueeze?
    """
    input_shape = list(input.size())
    if axis < 0:
        axis = len(input_shape) + axis + 1
    input_shape.insert(axis, 1)
    return input.view(*input_shape)
