import numpy as np
import tensorflow as tf


def f(t):
    return t


# `x` is a `tf.constant`-wrapped `numpy.array` value. See wala/ML#598.
x = tf.constant(np.array([1.0, 2.0, 3.0]))
assert x.shape == (3,)
f(x)
