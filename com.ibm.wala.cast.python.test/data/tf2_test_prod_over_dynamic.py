import numpy as np
import tensorflow as tf


def consume(t):
    pass


# `np.prod` over a shape list containing a `None` (dynamic) axis: the runtime
# guards the `None` away before folding (the BERT `get_shape_list` idiom), but
# the static walk sees the unguarded shape, so the product must taint to
# *dynamic* — arithmetic over a `None` axis is itself `None` at run time — and
# not degrade to the fixed-but-unresolved marker (wala/ML#721).
inp = tf.keras.Input(shape=(4, 6))
shape = inp.shape.as_list()

assert shape == [None, 4, 6]

if shape[0] is None:
    shape[0] = 1

n = np.prod(shape)

x = tf.reshape(tf.ones((1, 4, 6)), [n])

assert x.shape == (24,)
assert x.dtype == tf.float32

consume(x)
