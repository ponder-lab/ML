"""Exercises tensor-type propagation through a ``typing.Tuple``-annotated tuple-of-tensors parameter.

Mirrors the perf-eval corpus's ``deep_recommenders`` ``CIN.call(self, inputs: Tuple[tf.Tensor,
tf.Tensor])`` (an ``@tf.function``-decorated function the Hybridize tool refactors): a 2-tuple of
tensors is passed in and unpacked (``x, y = inputs``); each element should keep its original type.
This is the tuple-parameter analogue of the ``NamedTuple`` field case in wala/ML#579
(``tf2_test_namedtuple_field.py``).
"""

from typing import Tuple

import numpy as np
import tensorflow as tf


def consume(z):
    pass


def f(inputs: Tuple[tf.Tensor, tf.Tensor]):
    x, y = inputs
    assert x.shape == (4, 8) and x.dtype == tf.float32
    assert y.shape == (2, 3) and y.dtype == tf.float32
    consume(x)


a = tf.constant(np.ones((4, 8), dtype=np.float32))
assert a.shape == (4, 8) and a.dtype == tf.float32
b = tf.constant(np.ones((2, 3), dtype=np.float32))
assert b.shape == (2, 3) and b.dtype == tf.float32
f((a, b))
