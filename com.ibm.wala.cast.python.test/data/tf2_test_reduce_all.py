import tensorflow as tf


def f(a):
    pass


x = tf.constant([[True, False], [True, True]])
y = tf.math.reduce_all(x)
assert isinstance(y, tf.Tensor)
assert y.shape == ()
assert y.dtype == tf.bool

f(y)
