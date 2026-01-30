import tensorflow as tf


def f(a):
    pass


# single keyword: limit=5 (start=0, delta=1)
t1 = tf.range(limit=5)
assert isinstance(t1, tf.Tensor)
assert t1.shape == (5,)
assert t1.dtype == tf.int32

f(t1)
