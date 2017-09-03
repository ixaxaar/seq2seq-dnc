#!/usr/bin/env python

import torch
import gc
import inspect

exclude = [
    "function",
    "type",
    "list",
    "dict",
    "tuple",
    "wrapper_descriptor",
    "module",
    "method_descriptor",
    "member_descriptor",
    "instancemethod",
    "builtin_function_or_method",
    "frame",
    "classmethod",
    "classmethod_descriptor",
    "_Environ",
    "MemoryError",
    "_Printer",
    "_Helper",
    "getset_descriptor",
]


def get_user_attributes(cls):
  boring = dir(type('dummy', (object,), {}))
  return [item
          for item in inspect.getmembers(cls)
          if item[0] not in boring]


ignore_torch_types = [
    'Argument',
    'Function',
    'FunctionMeta',
    'Backend',
    'Size',
    '__PrinterOptions',
    'Warning'
]


def dumpObjects():
  gc.collect()
  oo = gc.get_objects()
  for o in oo:
    try:
      if getattr(o, "__class__", None):
        # if name not in exclude:
        name = o.__class__.__name__
        filename = inspect.getabsfile(o.__class__)
        if ('/lib/' not in filename and 'venv' not in filename):
          print("Python Object:", name, " sub class of ", o.__class__.__bases__, ' in file ', filename)
          # print(o.__class__.__dict__)
          # print('attributes', get_user_attributes(o.__class__))

      if getattr(o, "__class__", None):
        # if name not in exclude:
        name = o.__class__.__name__
        filename = inspect.getabsfile(o.__class__)
        if torch.is_tensor(o) or (hasattr(o, 'data') and torch.is_tensor(o.data)):
          print("    Torch Tensor: ", name, " of size ", o.size())
        # elif 'torch' in filename:
        #   if name not in ignore_torch_types:
        #     print("    Torch object: ", name, " of type ", type(o))

    except Exception as e:
      pass
