import numpy as np
import tensorflow as tf


def f(t):
    # `t` is a bare `numpy.array` value. See wala/ML#598.
    assert t.shape == (3,)
    return t


x = np.array([1.0, 2.0, 3.0])
f(x)
