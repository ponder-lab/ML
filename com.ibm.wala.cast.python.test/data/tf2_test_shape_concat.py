import tensorflow as tf


def f(a):
    pass


def get_shape(t):
    return t.shape.as_list()


# Concatenation of two shape vectors (wala/ML#708): the reshape target is
# `leading + trailing`, so its shape is the per-shape concatenation
# (4,) + (5, 6) = (4, 5, 6).
t = tf.ones((4, 5, 6))
x = tf.ones((120,))
r = tf.reshape(x, get_shape(t)[:1] + get_shape(t)[-2:])
assert isinstance(r, tf.Tensor)
assert r.shape == (4, 5, 6)
assert r.dtype == tf.float32
f(r)
