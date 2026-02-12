import tensorflow as tf


def f(a):
    pass


# start and delta keywords, limit is missing (behaves as range(limit=start, delta=delta))
t = tf.range(start=2, delta=1)
assert isinstance(t, tf.Tensor)
assert t.shape == (2,)  # [0, 1]
assert t.dtype == tf.int32

f(t)
