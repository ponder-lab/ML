# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/GradientTape#gradient.

import tensorflow as tf


def f(a):
    pass


arg = [[1, 2], [3], [4, 5, 6]]
assert isinstance(arg, list)
assert len(arg) == 3
assert all(isinstance(row, list) for row in arg)
assert all(isinstance(x, int) for row in arg for x in row)
assert len(arg[0]) == 2
assert len(arg[1]) == 1
assert len(arg[2]) == 3

x = tf.ragged.constant(arg, tf.int32)
assert isinstance(x, tf.RaggedTensor)
assert x.shape == (3, None)
assert x.dtype == tf.int32

f(x)
