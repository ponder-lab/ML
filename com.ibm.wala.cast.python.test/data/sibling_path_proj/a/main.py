# The PYTHONPATH is `a` then `b`, and both hold `conv/dense.py`. Python binds `conv.dense` to
# `a/conv/dense.py`, the module the first entry holds at that name, and `b/conv/dense.py` is
# shadowed: no import reaches it, though it is still code of the project (wala/ML#168).
import os
import sys

sys.path.insert(
    1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "b")
)

import tensorflow as tf
from conv.dense import make


def consume(x):
    assert x.shape == (3,)
    assert x.dtype == tf.float32


consume(make())
