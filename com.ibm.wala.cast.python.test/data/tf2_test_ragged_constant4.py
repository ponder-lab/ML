# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/GradientTape#gradient.

import tensorflow as tf


def f(a):
    pass


arg = [[]]
assert isinstance(arg, list)
assert len(arg) == 1
assert all(isinstance(row, list) for row in arg)
assert all(isinstance(x, int) for row in arg for x in row)
assert len(arg[0]) == 0
assert arg[0] == []
assert len(arg) == 1

x = tf.ragged.constant(arg)
assert isinstance(x, tf.RaggedTensor)
assert x.shape == (1, None)
assert x.dtype == tf.float32

f(x)
