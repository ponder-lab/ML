import tensorflow as tf


def f(a):
    pass


# 1-positional form: tf.range(limit, dtype=...)
r = tf.range(5, dtype=tf.float32)
assert r.shape == (5,)
assert r.dtype == tf.float32
f(r)
