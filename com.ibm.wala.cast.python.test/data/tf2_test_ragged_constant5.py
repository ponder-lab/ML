# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/GradientTape#gradient.

import tensorflow as tf


def f(a):
    pass


arg = [[], [[[]]]]
assert isinstance(arg, list)
assert len(arg) == 2
assert all(isinstance(row, list) for row in arg)
assert len(arg[0]) == 0
assert arg[0] == []
assert len(arg) == 2

x = tf.ragged.constant(arg)
assert isinstance(x, tf.RaggedTensor)
assert x.shape == (2, None, None, None)
assert x.dtype == tf.float32

f(x)
