"""Companion fixture for `tf2_test_expand_dims.py`: axis=-1 inserts a trailing length-1 dim.

For input shape ``(3,)``, ``tf.expand_dims(x, axis=-1)`` produces shape ``(3, 1)``.
"""

import tensorflow as tf


def f(a):
    pass


x = tf.constant([1.0, 2.0, 3.0])
y = tf.expand_dims(x, axis=-1)
assert isinstance(y, tf.Tensor)
assert y.shape == (3, 1)
assert y.dtype == tf.float32
f(y)
