import tensorflow as tf


def f(a):
    pass


# only start keyword (limit is None, so it behaves as range(limit=start))
t = tf.range(start=5)
assert isinstance(t, tf.Tensor)
assert t.shape == (5,)
assert t.dtype == tf.int32

f(t)
