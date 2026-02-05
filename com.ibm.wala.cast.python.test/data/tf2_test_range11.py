import tensorflow as tf


def f(a):
    pass


# 1 pos arg, limit and delta as keywords
# pos 0 -> start
# start=1, limit=5, delta=2
t = tf.range(1, limit=5, delta=2)
assert isinstance(t, tf.Tensor)
assert t.shape == (2,)  # [1, 3]
assert t.dtype == tf.int32

f(t)
