import tensorflow as tf


def f(a):
    pass


# keyword: start=1, limit=5, delta=2
t2 = tf.range(start=1, limit=5, delta=2)
assert isinstance(t2, tf.Tensor)
assert t2.shape == (2,)
assert t2.dtype == tf.int32

f(t2)
