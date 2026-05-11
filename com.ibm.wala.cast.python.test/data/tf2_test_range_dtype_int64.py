import tensorflow as tf


def f(a):
    pass


r = tf.range(0, 5, dtype=tf.int64)
assert r.shape == (5,)
assert r.dtype == tf.int64
f(r)
