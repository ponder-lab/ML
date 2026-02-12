# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/GradientTape#gradient.

import tensorflow as tf


def f(a):
    pass


arg = [[1.0, 2.0], [3.0]]
assert isinstance(arg, list)
assert isinstance(arg[0], list)
assert isinstance(arg[0][0], float)
assert isinstance(arg[1], list)
assert isinstance(arg[1][0], float)
assert all(isinstance(item, float) for sublist in arg for item in sublist)
assert all(isinstance(sublist, list) for sublist in arg)
assert len(arg) == 2
assert len(arg[0]) == 2
assert len(arg[1]) == 1

x = tf.ragged.constant(arg)
assert isinstance(x, tf.RaggedTensor)
assert x.shape == (2, None)
assert x.dtype == tf.float32

with tf.GradientTape() as g:
    g.watch(x)
    y = x * x
f(g.gradient(y, x))
