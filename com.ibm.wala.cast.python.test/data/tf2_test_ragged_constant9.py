# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/GradientTape#gradient.

import tensorflow as tf


def f(a):
    pass


arg = [[[1], [2]], [[3]], [[4], [5], [6]]]

x = tf.ragged.constant(arg, tf.float32, 2)
assert x.shape == (3, None, None)
assert x.dtype == tf.float32

f(x)
