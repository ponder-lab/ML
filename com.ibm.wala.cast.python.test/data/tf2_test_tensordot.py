import tensorflow as tf


def f(a):
    pass


a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
y = tf.linalg.tensordot(a, b, axes=1)
assert isinstance(y, tf.Tensor)
assert y.dtype == tf.float32

f(y)
