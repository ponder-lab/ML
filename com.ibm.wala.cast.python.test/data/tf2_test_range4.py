import tensorflow as tf


def f(x):
    pass


# Test tf.range(start, limit) where delta defaults to 1
start = 3
limit = 8
# range: [3, 4, 5, 6, 7] -> length 5

r = tf.range(start, limit)
assert isinstance(r, tf.Tensor)
assert r.shape == (5,)
assert r.dtype == tf.int32

f(r)
