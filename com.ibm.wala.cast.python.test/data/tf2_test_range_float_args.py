import tensorflow as tf


def f(a):
    pass


# Float args; no explicit dtype. Runtime promotes to float32.
r = tf.range(0.0, 5.0)
assert r.shape == (5,)
assert r.dtype == tf.float32
f(r)
