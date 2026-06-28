import numpy as np
import tensorflow as tf


def f(t):
    # `t` is a `tf.constant`-wrapped `numpy.array` value. See wala/ML#598.
    assert t.shape == (3,)
    return t


x = tf.constant(np.array([1.0, 2.0, 3.0]))
f(x)
