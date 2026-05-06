import tensorflow as tf


def f(a):
    pass


x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
y = tf.linalg.trace(x)
assert isinstance(y, tf.Tensor)
assert y.dtype == tf.float32

f(y)
