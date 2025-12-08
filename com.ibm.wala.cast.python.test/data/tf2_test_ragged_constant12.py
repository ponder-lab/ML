# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/GradientTape#gradient.

import tensorflow as tf


def f(a):
    pass


arg = [[[1, 2]], [[3, 4]]]

x = tf.ragged.constant(arg, tf.float32, 1)

assert x.shape == (2, None, 2)
assert x.dtype == tf.float32

f(x)
