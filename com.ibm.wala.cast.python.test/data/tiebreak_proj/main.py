# Two in-scope modules end with `conv/dense.py`. The PYTHONPATH is `b` then the project root, so
# Python binds `b/conv/dense.py`, the module an earlier entry holds at the name `conv/dense.py`,
# although `a/conv/dense.py` comes first in sorted order (wala/ML#168).
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "b"))

import tensorflow as tf
from conv.dense import make


def consume(x):
    assert x.shape == (4, 4)
    assert x.dtype == tf.float32


consume(make())
