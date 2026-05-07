import tensorflow as tf


def f(a):
    pass


# mixed: start=1, limit=5, delta=2
# range(start, limit, delta)
t3 = tf.range(1, 5, delta=2)
assert isinstance(t3, tf.Tensor)
assert t3.shape == (2,)
assert t3.dtype == tf.int32

f(t3)
