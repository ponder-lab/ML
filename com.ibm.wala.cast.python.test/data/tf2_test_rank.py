import tensorflow as tf


def f(a):
    pass


x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
y = tf.rank(x)
assert isinstance(y, tf.Tensor)
assert y.shape == ()
assert y.dtype == tf.int32

f(y)
