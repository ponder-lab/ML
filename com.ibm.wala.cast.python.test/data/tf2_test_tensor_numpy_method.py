# Test for wala/ML#1105: a tensor's `numpy()` returns a NumPy array of the tensor's shape and dtype,
# so the array and its rows reach the functions they are passed to typed, and the bound method
# `t.numpy` is no tensor.
import numpy as np
import tensorflow as tf


def calc_img_shapes(meta):
    return tf.cast(meta[..., 3:5], tf.int32).numpy()


def consume_array(a):
    assert isinstance(a, np.ndarray) and a.shape == (2, 2) and a.dtype == np.int32
    return a


def consume_row(r):
    assert isinstance(r, np.ndarray) and r.shape == (2,) and r.dtype == np.int32
    return r


def consume_method(m):
    assert callable(m)
    return m


shapes = consume_array(calc_img_shapes(tf.ones([2, 11])))
for i in range(shapes.shape[0]):
    consume_row(shapes[i])
consume_method(tf.ones([2, 3]).numpy)
