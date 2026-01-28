# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/GradientTape#gradient.

import tensorflow as tf


def f(a):
    pass


arg = [[1], [3]]
assert isinstance(arg, list)
assert len(arg) == 2
assert all(isinstance(row, list) for row in arg)
assert all(isinstance(x, int) for row in arg for x in row)
assert len(arg[0]) == 1
assert len(arg[1]) == 1

x = tf.ragged.constant(arg)
assert isinstance(x, tf.RaggedTensor)
assert x.shape == (2, None)
assert x.dtype == tf.int32

f(x)
