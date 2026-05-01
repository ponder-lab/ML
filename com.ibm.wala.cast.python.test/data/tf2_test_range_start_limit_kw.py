import tensorflow as tf


def f(a):
    pass


# only start and limit as keywords (delta defaults to 1)
t = tf.range(start=1, limit=5)
assert isinstance(t, tf.Tensor)
assert t.shape == (4,)
assert t.dtype == tf.int32

f(t)
