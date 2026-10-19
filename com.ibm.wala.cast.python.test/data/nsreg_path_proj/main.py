# The PYTHONPATH is the project root then `src`. The root's `conv` has no `__init__.py`, so it is
# only a portion of a namespace package, and `src/conv` is a regular package. Python's search
# takes a regular package from a later entry over a namespace portion from an earlier one, so
# `conv.dense` is `src/conv/dense.py` (wala/ML#168).
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import tensorflow as tf
from conv.dense import make


def consume(x):
    assert x.shape == (4, 4)
    assert x.dtype == tf.float32


consume(make())
