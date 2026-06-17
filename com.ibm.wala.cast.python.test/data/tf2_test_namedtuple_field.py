"""Exercises tensor-type propagation through a user-defined ``NamedTuple`` field (wala/ML#579).

A tensor stored in a ``NamedTuple`` field and read back out (``b = w.tensor``) keeps its original ``(4, 8) float32`` type.
This is the minimal form of the GCN blocker in wala/ML#570, where ``GraphConvolution.call`` unwraps a ``GNNInput`` ``NamedTuple`` the same way.
"""

from typing import NamedTuple, List

import numpy as np
import tensorflow as tf


class Wrapper(NamedTuple):
    tensor: tf.Tensor
    rest: List


def consume(x):
    pass


a = tf.constant(np.ones((4, 8), dtype=np.float32))
assert a.shape == (4, 8) and a.dtype == tf.float32
w = Wrapper(a, [])
b = w.tensor
assert b.shape == (4, 8) and b.dtype == tf.float32
consume(b)
