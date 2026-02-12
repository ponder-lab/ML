# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/GradientTape#gradient.

import tensorflow as tf


def f(a):
    pass


arg = [1, 2, 3, 4, 5]
assert isinstance(arg, list)
assert len(arg) == 5
assert all(isinstance(x, int) for x in arg)

x = tf.ragged.constant(arg)
assert isinstance(x, tf.Tensor)
assert x.shape == (5,)
assert x.dtype == tf.int32

f(x)
